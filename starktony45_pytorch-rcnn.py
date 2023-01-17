# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        

        os.path.join(dirname, filename)



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import torch

import cv2

from torchvision import datasets,transforms

from glob import glob

import os

from PIL import Image

from matplotlib import patches

from torch.utils.data import Dataset

import torchvision

from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from torchvision.models.detection import FasterRCNN

from torchvision.models.detection.rpn import AnchorGenerator

from torch.utils.data import DataLoader

#Albumentation

import albumentations as al

from albumentations.pytorch.transforms import ToTensorV2



train_dir ='/kaggle/input/global-wheat-detection/train/'

test_dir = '/kaggle/input/global-wheat-detection/test/'

train = pd.read_csv('../input/global-wheat-detection/train.csv') 

weights = '../input/weights/fasterrcnn_resnet50_fpn (1).pth'
sample = pd.read_csv('../input/global-wheat-detection/sample_submission.csv')

sample.shape
sample
class Wheatdatasets(Dataset):

    

    def __init__(self,dataframe,image_dir,transforms = None):

        

        super().__init__()

    

        self.image_id = dataframe['image_id'].unique()

        self.df = dataframe

        self.img_dir = image_dir

        self.transforms = transforms

        

    def __getitem__(self,index:int):

        

        image_id = self.image_id[index]

        record = self.df[self.df['image_id']==image_id]

        

        image = cv2.imread(self.img_dir+image_id+'.jpg',cv2.IMREAD_COLOR)

        image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB).astype(np.float32)

        image = image/255.0

       

        if self.transforms:

            sample ={

                'image':image

            }

            sample =self.transforms(**sample)

            image = sample['image']

            

            return image,image_id

        

    def __len__(self) -> int:

        return self.image_id.shape[0]
def get_transform():

    return al.Compose([

        ToTensorV2(p=1.0)

    ])
#Create the Model

model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained = False,pretrained_backbone=False)
device = torch.device('cuda') if torch.cuda.is_available() else 'cpu'

num_class = 2 #wheats and background



in_feature = model.roi_heads.box_predictor.cls_score.in_features



model.roi_heads.box_predictor = FastRCNNPredictor(in_feature,num_class) #changin the pretrained head with a new one



model.load_state_dict(torch.load(weights))

model.eval()
x = model.to(device)
def collate_fn(batch):

    return tuple(zip(*batch))





test_dataset = Wheatdatasets(sample,test_dir,get_transform())







test_dataloader = DataLoader(

test_dataset,

batch_size=4,

shuffle =False,

num_workers =4,drop_last = False,

collate_fn = collate_fn)





def format_string(boxes,score):

    pred_strings = []

    for j in zip(score,boxes):

        

        pred_strings.append("{0:.4f} {1} {2} {3} {4}".format(j[0], j[1][0], j[1][1], j[1][2], j[1][3]))

        

    return " ".join(pred_strings)

        
detection_threshold = 0.5

results = []



for images, image_ids in test_dataloader:



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

            'PredictionString': format_string(boxes, scores)

        }



        

        results.append(result)
results[:2]




test_df = pd.DataFrame(results, columns=['image_id', 'PredictionString'])

test_df.head()
sample = images[0].permute(1,2,0).cpu().numpy()

boxes = outputs[0]['boxes'].data.cpu().numpy()

score = outputs[0]['scores'].data.cpu().numpy()



boxes = boxes[score >= detection_threshold].astype(np.int32)
fig, ax = plt.subplots(1, 1, figsize=(16, 8))



for box in boxes:

    cv2.rectangle(sample,

                  (box[0], box[1]),

                  (box[2], box[3]),

                  (220, 0, 0), 2)

    

ax.set_axis_off()

ax.imshow(sample)
test_df.to_csv('submission.csv', index=False)