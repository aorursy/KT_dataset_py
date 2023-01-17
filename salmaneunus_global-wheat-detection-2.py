!pip install --no-deps '../input/weightedboxesfusion/' > /dev/null
import pandas as pd

import numpy as np

from sklearn.model_selection import StratifiedKFold

from torch.utils.data import Dataset,DataLoader

import albumentations as A

from albumentations.pytorch.transforms import ToTensorV2

from glob import glob

import torch

import cv2
%%time



marking = pd.read_csv('../input/global-wheat-detection/train.csv')



bboxs = np.stack(marking['bbox'].apply(lambda x: np.fromstring(x[1:-1], sep=',')))

for i, column in enumerate(['x', 'y', 'w', 'h']):

    marking[column] = bboxs[:,i]

marking.drop(columns=['bbox'], inplace=True)



marking.head()
marking.tail()
marking.describe()
marking['source'].hist(bins=20);
import seaborn as sns

import matplotlib.pyplot as plt

sns.lineplot(data=marking,x = marking.x,y=marking.y);
plt.figure(figsize=(20,10))

sns.boxplot(data=marking,x = marking.w,y=marking.h);
%%time



skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)



df_folds = marking[['image_id']].copy()

df_folds.loc[:, 'bbox_count'] = 1

df_folds = df_folds.groupby('image_id').count()

df_folds.loc[:, 'source'] = marking[['image_id', 'source']].groupby('image_id').min()['source']

df_folds.loc[:, 'stratify_group'] = np.char.add(

    df_folds['source'].values.astype(str),

    df_folds['bbox_count'].apply(lambda x: f'_{x // 15}').values.astype(str)

)

df_folds.loc[:, 'fold'] = 0



for fold_number, (train_index, val_index) in enumerate(skf.split(X=df_folds.index, y=df_folds['stratify_group'])):

    df_folds.loc[df_folds.iloc[val_index].index, 'fold'] = fold_number
df_folds.head()
def get_valid_transforms():

    return A.Compose([

            A.Resize(height=512, width=512, p=1.0),

            ToTensorV2(p=1.0),

        ], p=1.0)
DATA_ROOT_PATH = '../input/global-wheat-detection/test'



class DatasetRetriever(Dataset):



    def __init__(self, image_ids, transforms=None):

        super().__init__()

        self.image_ids = image_ids

        self.transforms = transforms



    def __getitem__(self, index: int):

        image_id = self.image_ids[index]

        image = cv2.imread(f'{DATA_ROOT_PATH}/{image_id}.jpg', cv2.IMREAD_COLOR)

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)

        image /= 255.0

        if self.transforms:

            sample = {'image': image}

            sample = self.transforms(**sample)

            image = sample['image']

        return image, image_id



    def __len__(self) -> int:

        return self.image_ids.shape[0]
dataset = DatasetRetriever(

    image_ids=np.array([path.split('/')[-1][:-4] for path in glob(f'{DATA_ROOT_PATH}/*.jpg')]),

    transforms=get_valid_transforms()

)



def collate_fn(batch):

    return tuple(zip(*batch))



data_loader = DataLoader(

    dataset,

    batch_size=2,

    shuffle=False,

    num_workers=4,

    drop_last=False,

    collate_fn=collate_fn

)
import gc

import torchvision

from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from torchvision.models.detection import FasterRCNN

from torchvision.models.detection.rpn import AnchorGenerator





def load_net(checkpoint_path):

    net = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=False, pretrained_backbone=False)

    num_classes = 2  # 1 class (wheat) + background

    # get number of input features for the classifier

    in_features = net.roi_heads.box_predictor.cls_score.in_features

    # replace the pre-trained head with a new one

    net.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)



    checkpoint = torch.load(checkpoint_path)

    net.load_state_dict(checkpoint)

    net = net.cuda()

    net.eval()



    del checkpoint

    gc.collect()

    return net



models = [

    load_net('../input/wheat-fasterrcnn-folds/fold0-best1.bin'),

    load_net('../input/wheat-fasterrcnn-folds/fold1-best1.bin'),

    load_net('../input/wheat-fasterrcnn-folds/fold2-best1.bin'),

    load_net('../input/wheat-fasterrcnn-folds/fold3-best1.bin'),

    load_net('../input/wheat-fasterrcnn-folds/fold4-best1.bin'),

]
from ensemble_boxes import *



device = torch.device('cuda:0')



def make_ensemble_predictions(images):

    images = list(image.to(device) for image in images)    

    result = []

    for net in models:

        outputs = net(images)

        result.append(outputs)

    return result



def run_wbf(predictions, image_index, image_size=512, iou_thr=0.55, skip_box_thr=0.7, weights=None):

    boxes = [prediction[image_index]['boxes'].data.cpu().numpy()/(image_size-1) for prediction in predictions]

    scores = [prediction[image_index]['scores'].data.cpu().numpy() for prediction in predictions]

    labels = [np.ones(prediction[image_index]['scores'].shape[0]) for prediction in predictions]

    boxes, scores, labels = weighted_boxes_fusion(boxes, scores, labels, weights=None, iou_thr=iou_thr, skip_box_thr=skip_box_thr)

    boxes = boxes*(image_size-1)

    return boxes, scores, labels
import matplotlib.pyplot as plt



for j, (images, image_ids) in enumerate(data_loader):

    if j > 0:

        break

predictions = make_ensemble_predictions(images)



i = 1

sample = images[i].permute(1,2,0).cpu().numpy()

boxes, scores, labels = run_wbf(predictions, image_index=i)

boxes = boxes.astype(np.int32).clip(min=0, max=511)



fig, ax = plt.subplots(1, 1, figsize=(16, 8))



for box in boxes:

    cv2.rectangle(sample,

                  (box[0], box[1]),

                  (box[2], box[3]),

                  (220, 0, 0), 2)

    

ax.set_axis_off()

ax.imshow(sample);
def format_prediction_string(boxes, scores):

    pred_strings = []

    for j in zip(scores, boxes):

        pred_strings.append("{0:.4f} {1} {2} {3} {4}".format(j[0], j[1][0], j[1][1], j[1][2], j[1][3]))

    return " ".join(pred_strings)
results = []



for images, image_ids in data_loader:

    predictions = make_ensemble_predictions(images)

    for i, image in enumerate(images):

        boxes, scores, labels = run_wbf(predictions, image_index=i)

        boxes = (boxes*2).astype(np.int32).clip(min=0, max=1023)

        image_id = image_ids[i]



        boxes[:, 2] = boxes[:, 2] - boxes[:, 0]

        boxes[:, 3] = boxes[:, 3] - boxes[:, 1]

        

        result = {

            'image_id': image_id,

            'PredictionString': format_prediction_string(boxes, scores)

        }

        results.append(result)
test_df = pd.DataFrame(results, columns=['image_id', 'PredictionString'])

test_df.to_csv('submission.csv', index=False)

test_df.head()