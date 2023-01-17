import torch

import torchvision

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import os

import glob

import cv2

import torch.nn as nn

import torch.optim as optim

import torchvision.transforms as transforms

from PIL import Image

from torch.utils.data import dataloader,dataset

import albumentations as A

from albumentations.pytorch import ToTensor 



from torchvision.models.detection.faster_rcnn import FastRCNNPredictor, FasterRCNN

from torchvision.models.detection.backbone_utils import resnet_fpn_backbone

import warnings

warnings.filterwarnings("ignore")

test_path = '../input/global-wheat-detection/test'

model_path = '../input/modelweight'
test_df = pd.read_csv('../input/global-wheat-detection/sample_submission.csv')

test_df.shape
def format_prediction_string(boxes, scores):

    pred_strings = []

    for j in zip(scores, boxes):

        pred_strings.append("{0:.4f} {1} {2} {3} {4}".format(j[0], j[1][0], j[1][1], j[1][2], j[1][3]))



    return " ".join(pred_strings)
def collate_fn(batch):

    return tuple(zip(*batch))
class test_dataset(torch.utils.data.Dataset):

  

  def __init__(self, image_list, transforms=None):

    super().__init__()

    self.images = image_list

    self.transforms = transforms

    



  def __len__(self):

    return len(self.images)



  def __getitem__(self,item):

    image = self.images[item]

    

    base_name = os.path.basename(image)

    image_name_split = os.path.splitext(base_name)

    image_id = image_name_split[0] 

    

    image = cv2.imread(image, cv2.IMREAD_COLOR)

    image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)

    image = image/255.0

    image = np.transpose(image,(2,1,0))

    

    #if self.transforms:

      #sample = {'image':image}

    #  image = self.transforms(image)



    #image = sample['image']

    #image = self.transforms(image)



    return torch.as_tensor(image,dtype=torch.float32),image_id
images_list= glob.glob(os.path.join(test_path,'*.jpg'))
#transform = A.Compose([ToTensor()])

test_data=test_dataset(images_list)

test_loader = torch.utils.data.DataLoader(test_data,batch_size=4,shuffle=False,collate_fn=collate_fn)
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=False,pretrained_backbone=False)

num_classes = 2

in_features = model.roi_heads.box_predictor.cls_score.in_features

model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
model.load_state_dict(torch.load(os.path.join(model_path,'resnet50_GWD_6.pth')))
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

model = model.to(device)
model.eval()

detection_threshold = 0.5

results = []



for images,image_ids in test_loader:



    images_12 = list(image.to(device) for image in images)

    with torch.no_grad():

        outputs = model(images_12)

    #outputs = model(images)



        for i, image in enumerate(images_12):



            boxes = outputs[i]['boxes'].cpu().numpy()

            scores = outputs[i]['scores'].cpu().numpy()

        

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
test_df['PredictionString']=results

test_df = pd.DataFrame(results, columns=['image_id', 'PredictionString'])

test_df.head(10

            )
boxes = outputs[0]['boxes'].data.cpu().numpy()

scores = outputs[0]['scores'].data.cpu().numpy()



boxes = boxes[scores >= detection_threshold].astype(np.int32)


image = cv2.imread(os.path.join(test_path,'f5a1f0358'+'.jpg'),cv2.IMREAD_COLOR)

image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
for i in boxes:

  image = cv2.rectangle(image,(i[0],i[1]),(i[2],i[3]),(220,0,0),3)
fig,ax = plt.subplots(1,1,figsize=(15,15))

ax.imshow(image)

plt.show()
test_df.to_csv('submission.csv', index=False)