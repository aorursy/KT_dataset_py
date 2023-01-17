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

        print(os.path.join(dirname, filename))



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
train_dir ='/kaggle/input/global-wheat-detection/train/'

test_dir = '../input/global-wheat-detection/test/'

train = pd.read_csv('../input/global-wheat-detection/train.csv') 
train_images = glob(train_dir +'*')

test_images = glob(test_dir +'*')

train_images[:10]
train.head()
train['bbox'] = train['bbox'].fillna('[0,0,0,0]')

bbox_items = train['bbox'].str.split(',',expand = True)

train['x'] = bbox_items[0].str.strip('[').astype(float)

train['y'] = bbox_items[1].str.strip(' ').astype(float)

train['w'] = bbox_items[2].str.strip(' ').astype(float)

train['h'] = bbox_items[3].str.strip(']').astype(float)
train.head()
img_id = train['image_id'].unique()

valid_ids = img_id[-665:]

train_ids = img_id[:-665]
valid_df = train[train['image_id'].isin(valid_ids)]

train_df = train[train['image_id'].isin(train_ids)]
valid_df.shape,train_df.shape

train['area'] = train['w']*train['h']
train
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

        

        boxes = record[['x','y','w','h']].values

        boxes[:,2] = boxes[:,2]+boxes[:,0]

        boxes[:,3] = boxes[:,3]+boxes[:,1]

        

        area = (boxes[:,2]- boxes[:,0])*(boxes[:,1]-boxes[:,3])

        area = torch.as_tensor(area,dtype=torch.float32)

        

        #There is any class

        labels = torch.ones((record.shape[0],),dtype =torch.int64)

        

        #suppose all instances are not crowd

        iscrowd = torch.zeros((record.shape[0],),dtype =torch.int64)

        

        target ={}

        target['boxes']=boxes

        target['area'] =area

        target['labels']=labels

        target['iscrowd'] = iscrowd

        target['image_id'] = torch.tensor([index])

        

        if self.transforms:

            sample ={

                'image':image,

                'bboxes':target['boxes'],

                'labels':labels

            }

            sample =self.transforms(**sample)

            image = sample['image']

                

            target['boxes'] = torch.tensor(sample['bboxes']).float()

            return image,target,image_id

        

    def __len__(self) -> int:

        return self.image_id.shape[0]

        

        
#Albumentation

import albumentations as al

from albumentations.pytorch.transforms import ToTensorV2



def train_trans():

    return al.Compose([

    al.Flip(0.5),

    al.HorizontalFlip(p=0.5),

    al.VerticalFlip(p=0.5),

    al.OneOf([al.RandomContrast(),

             al.RandomGamma(),

             al.RandomBrightness()],p=1.0),

    

    ToTensorV2(p=1.0)], bbox_params ={'format':'pascal_voc','label_fields':['labels']})







def valid_trans():

    return al.Compose([

    

    ToTensorV2(p=1.0)], bbox_params ={'format': 'pascal_voc','label_fields':['labels']})



#Create the Model

model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained = True)

num_class = 2 #wheats and background



in_feature = model.roi_heads.box_predictor.cls_score.in_features



model.roi_heads.box_predictor = FastRCNNPredictor(in_feature,num_class) #changin the pretrained head with a new one
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
def collate_fn(batch):

    return tuple(zip(*batch))



train_dataset = Wheatdatasets(train_df, train_dir,train_trans())

valid_dataset = Wheatdatasets(valid_df,train_dir,valid_trans())



indices = torch.randperm(len(train_dataset)).tolist()



train_dataloader = DataLoader(

train_dataset,

batch_size=8,

shuffle =False,

num_workers =4,

collate_fn = collate_fn)



valid_dataloader  = DataLoader(

valid_dataset,

    batch_size=8,

    shuffle=False,

    num_workers=4,

    collate_fn=collate_fn)
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

print(device)
images,target,image_id = next(iter(train_dataloader))

images = list(image.to(device) for image in images)

targets = [{k: v.to(device) for k,v in t.items()} for t in target]
boxes = targets[2]['boxes'].cpu().numpy().astype(np.int32)

sample = images[2].permute(1,2,0).cpu().numpy()
fig, ax = plt.subplots(1, 1, figsize=(16, 8))



for box in boxes:

    cv2.rectangle(sample,

                  (box[0], box[1]),

                  (box[2], box[3]),

                  (220, 0, 0), 3)

    

ax.set_axis_off()

ax.imshow(sample)
model.to(device)

params = [p for p in model.parameters() if p.requires_grad]

optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)

# lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

lr_scheduler = None



num_epochs = 2

torch.cuda.empty_cache()
torch.cuda.empty_cache()
loss_hist = Averager()

itr = 2



for epoch in range(num_epochs):

    loss_hist.reset()

    

    for images, targets, image_ids in train_dataloader:

        

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

    

    # update the learning rate

    if lr_scheduler is not None:

        lr_scheduler.step()



    print(f"Epoch #{epoch} loss: {loss_hist.value}")


images, targets, image_ids = next(iter(valid_dataloader))
             

images = list(img.to(device) for img in images)

targets = [{k: v.to(device) for k, v in t.items()} for t in targets]




boxes = targets[1]['boxes'].cpu().numpy().astype(np.int32)

sample = images[1].permute(1,2,0).cpu().numpy()
model.eval()

cpu_device = torch.device("cpu")



outputs = model(images)

outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]
fig, ax = plt.subplots(1, 1, figsize=(16, 8))



for box in boxes:

    cv2.rectangle(sample,

                  (box[0], box[1]),

                  (box[2], box[3]),

                  (220, 0, 0), 3)

    

ax.set_axis_off()

ax.imshow(sample)
torch.save(model.state_dict(), 'fasterrcnn_resnet50_fpn.pth')