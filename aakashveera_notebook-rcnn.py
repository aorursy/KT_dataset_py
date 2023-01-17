import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import matplotlib.patches as patches

import cv2

from os import listdir

from os.path import isfile, join

import warnings

warnings.filterwarnings("ignore")
import torch

import torchvision

from torch.utils.data import DataLoader, Dataset

from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from torchvision.models.detection import FasterRCNN

import albumentations as A

from albumentations.pytorch.transforms import ToTensorV2
onlyfiles = [f for f in listdir('../input/global-wheat-detection/test/') if isfile(join('../input/global-wheat-detection/test/', f))]



test_df = pd.DataFrame(onlyfiles,columns=['image_id'])
transform =  A.Compose([

        ToTensorV2(p=1.0)

    ])





def collate_fn(batch):

    return tuple(zip(*batch))
class WheatTestDataset(Dataset):



    def __init__(self, dataframe, transforms):

        super().__init__()



        self.image_ids = dataframe['image_id'].unique()

        self.df = dataframe

        self.transforms = transforms



    def __getitem__(self, index: int):



        image_id = self.image_ids[index]

        records = self.df[self.df['image_id'] == image_id]



        image = cv2.imread('../input/global-wheat-detection/test/'+image_id, cv2.IMREAD_COLOR)

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)

        image /= 255.0



        sample = {

            'image': image,

        }

        sample = self.transforms(**sample)

        image = sample['image']



        return image, image_id



    def __len__(self) -> int:

        return self.image_ids.shape[0]
test_dataset = WheatTestDataset(test_df,transform)



test_data_loader = DataLoader(

    test_dataset,

    batch_size=8,

    shuffle=False,

    num_workers=4,

    collate_fn=collate_fn

)
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=False,pretrained_backbone=False)

num_classes = 2

in_features = model.roi_heads.box_predictor.cls_score.in_features

model.roi_heads.box_predictor
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

model.roi_heads.box_predictor

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)
model.load_state_dict(torch.load('../input/global-wheat-detection-public/fasterrcnn_resnet50_fpn_best.pth'))

model.eval()
def format_prediction_string(boxes, scores):

    pred_strings = []

    for j in zip(scores, boxes):

        pred_strings.append("{0:.4f} {1} {2} {3} {4}".format(j[0], j[1][0], j[1][1], j[1][2], j[1][3]))



    return " ".join(pred_strings)
detection_threshold = 0.5

results = []

testdf_psuedo = []

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

        

        for box in boxes:

            result = {

                'image_id': 'nvnn'+image_id,

                'source': 'nvnn',

                'x': box[0],

                'y': box[1],

                'w': box[2],

                'h': box[3]

            }

            testdf_psuedo.append(result)
test_df_pseudo = pd.DataFrame(testdf_psuedo, columns=['image_id', 'source', 'x', 'y', 'w', 'h'])

test_df_pseudo.head()
import cv2

img = cv2.imread("../input/global-wheat-detection/test/348a992bb.jpg")

img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
fig,ax = plt.subplots(1)

ax.imshow(img)

for x,y,width,height in test_df_pseudo[test_df_pseudo['image_id'] == "nvnn348a992bb.jpg"][['x','y','w','h']].values:

    rect = patches.Rectangle((x,y),width,height,linewidth=1,edgecolor='r',facecolor='none')

    ax.add_patch(rect)

plt.show()
train_df = pd.read_csv("../input/global-wheat-detection/train.csv")

train_df.drop(['width','height'],axis=1,inplace=True)
train_df['x'] = train_df['bbox'].apply(lambda x: int(float(x[1:-1].split(',')[0])))

train_df['y'] = train_df['bbox'].apply(lambda x: int(float(x[1:-1].split(',')[1])))

train_df['w'] = train_df['bbox'].apply(lambda x: int(float(x[1:-1].split(',')[2])))

train_df['h'] = train_df['bbox'].apply(lambda x: int(float(x[1:-1].split(',')[3])))
train_df.drop('bbox',axis=1,inplace=True)
train_df = pd.concat([train_df,test_df_pseudo],axis=0)

train_df.reset_index(drop=True,inplace=True)
train_df
transform =  A.Compose([

        ToTensorV2(p=1.0)

    ], bbox_params={'format': 'pascal_voc', 'label_fields': ['labels']})
class WheatDataset(Dataset):



    def __init__(self, dataframe, transforms):

        super().__init__()



        self.image_ids = dataframe['image_id'].unique()

        self.df = dataframe

        self.transforms = transforms



    def __getitem__(self, index: int):



        image_id = self.image_ids[index]

        records = self.df[self.df['image_id'] == image_id]



        if 'nvnn' in image_id:

            image_id = image_id[4:]

            image = cv2.imread('../input/global-wheat-detection/test/'+image_id, cv2.IMREAD_COLOR)

        else:

            image = cv2.imread('../input/global-wheat-detection/train/'+image_id+'.jpg', cv2.IMREAD_COLOR)

        

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)

        image /= 255.0



        boxes = records[['x', 'y', 'w', 'h']].values

        boxes[:, 2] = boxes[:, 0] + boxes[:, 2]

        boxes[:, 3] = boxes[:, 1] + boxes[:, 3]

        

        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])

        area = torch.as_tensor(area, dtype=torch.float32)



        labels = torch.ones((records.shape[0],), dtype=torch.int64)       

        iscrowd = torch.zeros((records.shape[0],), dtype=torch.int64)

        

        target = {}

        target['boxes'] = boxes

        target['labels'] = labels

        target['iscrowd'] = iscrowd

        target['area'] = area

        target['image_id'] = torch.tensor([index])



        sample = {

                'image': image,

                'bboxes': target['boxes'],

                'labels': labels

            }

        sample = self.transforms(**sample)

        image = sample['image']



        target['boxes'] = torch.stack(tuple(map(torch.tensor, zip(*sample['bboxes'])))).permute(1, 0)

        return image, target, image_id



    def __len__(self) -> int:

        return self.image_ids.shape[0]
class Averager:

    def __init__(self):

        self.current_total = 0.0

        self.iterations = 0.0



    def send(self, value):

        self.current_total += value

        self.iterations += 1



    @property

    def value(self):

        if self.iterations == 0:

            return 0

        else:

            return 1.0 * self.current_total / self.iterations



    def reset(self):

        self.current_total = 0.0

        self.iterations = 0.0
train_dataset = WheatDataset(train_df,transform)
train_data_loader = DataLoader(

    train_dataset,

    batch_size=16,

    shuffle=False,

    num_workers=4,

    collate_fn=collate_fn

)
model.train()
params = [p for p in model.parameters() if p.requires_grad]

optimizer = torch.optim.SGD(params, lr=0.01, momentum=0.9, weight_decay=0.0001)
loss_hist = Averager()



num_epochs = 30
itr = 1



for epoch in range(num_epochs):

    loss_hist.reset()

    

    for images, targets, image_ids in train_data_loader:

        

        images = list(image.to(device) for image in images)

        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]



        loss_dict = model(images, targets)



        losses = sum(loss for loss in loss_dict.values())

        loss_value = losses.item()



        loss_hist.send(loss_value)



        optimizer.zero_grad()

        losses.backward()

        optimizer.step()



        if itr % 50 == 0:

            print(f"Iteration #{itr} loss: {loss_value}")



        itr += 1

    



    print(f"Epoch #{epoch} loss: {loss_hist.value}")
torch.save(model.state_dict(), 'fasterrcnn_resnet50_fpn2nd.pt')
model.eval()

detection_threshold = 0.5

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

            'PredictionString': format_prediction_string(boxes, scores)

        }



        

        results.append(result)
test_df = pd.DataFrame(results, columns=['image_id', 'PredictionString'])

test_df['image_id'] = test_df['image_id'].apply(lambda x: x.split(".")[0])

test_df.head()
test_df.to_csv('submission.csv', index=False)