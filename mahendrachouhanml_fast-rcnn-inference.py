import os

import pandas as pd

import numpy as np

import pandas as pd

import torch

import torch.utils.data

import torchvision

import cv2

from PIL import Image

from matplotlib import pyplot as plt

#from pycocotools.coco import COCO

from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

import albumentations as A

from albumentations.pytorch.transforms import ToTensorV2

from torch.utils.data import DataLoader
# path to your own data and coco file

train_data_dir = "data/upload"

train_coco = "object_detaction__wheat_ml/abc.json"



Model_path = "../input/trained-model/fasterrcnn_resnet50.pth"



# Batch size

train_batch_size = 1



# Params for dataloader

train_shuffle_dl = True

num_workers_dl = 4



# Params for training



# Two classes; Only target class or background

num_classes = 2

num_epochs = 2



lr = 0.005

momentum = 0.9

weight_decay = 0.005



TEST_FOLDER = "../input/global-wheat-detection/test"



DIR_INPUT = '/kaggle/input/global-wheat-detection'

DIR_TRAIN = f'{DIR_INPUT}/train'

DIR_TEST = f'{DIR_INPUT}/test'





device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')





test_df = pd.read_csv(f'{DIR_INPUT}/sample_submission.csv')

test_df.shape
class myOwnDataset(torch.utils.data.Dataset):

    

    def __init__(self, dataframe, image_dir, transforms=None):

        super().__init__()



        self.image_ids = dataframe['image_id'].unique()

        self.df = dataframe

        self.image_dir = image_dir

        self.transforms = transforms



    def __getitem__(self, index: int):



        image_id = self.image_ids[index]

        records = self.df[self.df['image_id'] == image_id]



        image = cv2.imread(f'{self.image_dir}/{image_id}.jpg', cv2.IMREAD_COLOR)

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)

        image /= 255.0



        if self.transforms:

            sample = {

                'image': image,

            }

            sample = self.transforms(**sample)

            image = sample['image']



        return image, image_id



    def __len__(self) -> int:

        return self.image_ids.shape[0]
# In my case, just added ToTensor

def get_transform():

    return A.Compose([

        # A.Resize(512, 512),

        ToTensorV2(p=1.0)

    ])

def collate_fn(batch):

    return tuple(zip(*batch))



test_dataset = myOwnDataset(test_df, DIR_TEST, get_transform())



test_data_loader = DataLoader(

    test_dataset,

    batch_size=4,

    shuffle=False,

    num_workers=4,

    drop_last=False,

    collate_fn=collate_fn

)
# collate_fn needs for batch

def collate_fn(batch):

    return tuple(zip(*batch))





def get_model_instance_segmentation(num_classes):

    # load an instance segmentation model pre-trained pre-trained on COCO

    model =  torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=False,

                                                                  pretrained_backbone=False)

    # get number of input features for the classifier

    in_features = model.roi_heads.box_predictor.cls_score.in_features

    # replace the pre-trained head with a new one

    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)



    return model
model = get_model_instance_segmentation(num_classes)

model.load_state_dict(torch.load(Model_path))

model.eval()

x = model.to(device)
def prediction_sting(boxes, scores):

    pred_strings = []

    for j in zip(scores, boxes):

        pred_strings.append("{0:.4f} {1} {2} {3} {4}".format(j[0], j[1][0], j[1][1], j[1][2], j[1][3]))



    return " ".join(pred_strings)



def predict(filepath, image_id, detection_threshold=0.3):

    with open(filepath, mode='rb') as f:

        image = Image.open(f).convert('RGB')

    image_height, image_width = image.size

    

    trasformer = get_transform()    

    trasform_image = trasformer(image)

    images = torch.stack([trasform_image])

    #with torch.no_grad():

    results = model(images)

    boxes = results[0]['boxes'].data.cpu().numpy()

    scores = results[0]['scores'].data.cpu().numpy()

    #print(boxes)

    #print(scores)

    boxes = boxes[scores >= detection_threshold].astype(np.int32)

    scores = scores[scores >= detection_threshold]

    #image_id = image_ids[i]



    boxes[:, 2] = boxes[:, 2] - boxes[:, 0]

    boxes[:, 3] = boxes[:, 3] - boxes[:, 1]



    result = {

        'image_id': image_id,

        'PredictionString': prediction_sting(boxes, scores)

    }

    return result
detection_threshold = 0.40



results = []

for images, image_ids in test_data_loader:



    images = list(image.to(device) for image in images)

    outputs = model(images)



    for i, image in enumerate(images):



        boxes = outputs[i]['boxes'].data.cpu().numpy()

        scores = outputs[i]['scores'].data.cpu().numpy()

        

        boxes = boxes[scores >= detection_threshold].astype(np.int32)

        scores = scores[scores >= detection_threshold]

        image_id = image_ids[i]

        

        boxes[:, 2] = boxes[:, 2] - boxes[:, 0]

        boxes[:, 3] = boxes[:, 3] - boxes[:, 1]

        

        result = {

            'image_id': image_id,

            'PredictionString': prediction_sting(boxes, scores)

        }



        

        results.append(result)
all_result = []

for dirname, _, filenames in os.walk(TEST_FOLDER):

    for filename in filenames:

        image_path = os.path.join(dirname, filename)

        image_id = filename.split(".")[0]

        result = ""

        #result = predict(image_path, image_id)

        all_result.append(result)

#print(all_result)        
test_df = pd.DataFrame(results, columns=['image_id', 'PredictionString'])

test_df.to_csv('submission.csv', index=False)

test_df.head()
sample = images[1].permute(1,2,0).cpu().numpy()

boxes = outputs[1]['boxes'].data.cpu().numpy()

scores = outputs[1]['scores'].data.cpu().numpy()



boxes = boxes[scores >= detection_threshold].astype(np.int32)
fig, ax = plt.subplots(1, 1, figsize=(16, 8))



for box in boxes:

    cv2.rectangle(sample,

                  (box[0], box[1]),

                  (box[2], box[3]),

                  (220, 0, 0), 2)

    

ax.set_axis_off()

ax.imshow(sample)