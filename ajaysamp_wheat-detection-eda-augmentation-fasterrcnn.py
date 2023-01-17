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
# import required libraries

import numpy as np





import PIL

from PIL import Image



# plotly libraries

import matplotlib.pyplot as plt

import matplotlib.patches as patches

import plotly.graph_objs as go

from plotly.offline import iplot, init_notebook_mode

import plotly.express as px

import plotly.io as pio



import cv2
# initialize source paths

source_path = '/kaggle/input/global-wheat-detection/'

train_path = source_path + 'train/'

test_path = source_path + 'test/'



# read data table

train_df = pd.read_csv(source_path +'train.csv')
train_df.head()
# expand the bbox column into seprate columns

train_df[['bbox_xmin','bbox_ymin','bbox_width','bbox_height']] = train_df['bbox'].str.split(',',expand=True)

train_df['bbox_xmin'] = train_df['bbox_xmin'].str.replace('[','').astype(float)

train_df['bbox_ymin'] = train_df['bbox_ymin'].str.replace(' ','').astype(float)

train_df['bbox_width'] = train_df['bbox_width'].str.replace(' ','').astype(float)

train_df['bbox_height'] = train_df['bbox_height'].str.replace(']','').astype(float)



# add xmax, ymax, and area columns for bounding box

train_df['bbox_xmax'] = train_df['bbox_xmin'] + train_df['bbox_width']

train_df['bbox_ymax'] = train_df['bbox_ymin'] + train_df['bbox_height']

train_df['bbox_area'] = train_df['bbox_height'] * train_df['bbox_width']
# count distinct images by source

img_source_dist = train_df.groupby(['source']).agg(image_count=('image_id','nunique'),wheat_head=('image_id','size'))

img_source_dist.reset_index(inplace=True,drop=False)

fig = px.pie(img_source_dist, values='image_count', names='source', title='Spike Distribution by Source')

fig.show()
img_source_dist['Avg_Wheat_Head'] = img_source_dist['wheat_head']/img_source_dist['image_count']

img_source_dist = img_source_dist.sort_values(by='Avg_Wheat_Head', ascending=True)



fig = go.Figure(data=[

    go.Bar(name='Avg Wheat Head Count', x=img_source_dist['Avg_Wheat_Head'], y=img_source_dist['source'],

           orientation='h',marker_color='salmon')

])

# Change the bar mode

fig.update_layout(title_text='Avg Number of Spikes',

                  height=400)

fig.show()
wheat_heads_per_image = train_df.groupby('image_id').agg(head_count=('image_id','size'))

wheat_heads_per_image.reset_index(inplace=True, drop=False)



fig = px.histogram(wheat_heads_per_image, x="head_count",marginal="box")

fig.update_layout(

    xaxis = dict(

        title_text = "Spike Count"), title = 'Spike Count per Image')

fig.show()
# create list of images per the regions identified 

heads_large =  list(wheat_heads_per_image[wheat_heads_per_image.head_count > 100]['image_id'].unique())

heads_normal = list(wheat_heads_per_image[(wheat_heads_per_image.head_count >= 30) & (wheat_heads_per_image.head_count <= 30)]['image_id'].unique())

heads_small =  list(wheat_heads_per_image[wheat_heads_per_image.head_count <= 5]['image_id'].unique())
# define a function to display the images

def get_bbox(df, image_id):

    bboxes = []

    image_bbox = df[df.image_id == image_id]

    

    for _,row in image_bbox.iterrows():

        bboxes.append([row.bbox_xmin, row.bbox_ymin, row.bbox_width, row.bbox_height])

        

    return bboxes

        

def plot_image(images, title=None):

    fig = plt.figure(figsize = (20,10))

    for i in range(1,4):

        ax = fig.add_subplot(1, 3, i)

        img = np.random.choice(images)

        image_path = os.path.join(train_path,img +'.jpg')

        image = Image.open(image_path)

        ax.imshow(image)

    

        b = get_bbox(train_df,img)

    

        for bbox in b:

                    rect = patches.Rectangle((bbox[0],bbox[1]),bbox[2],bbox[3],linewidth=2,edgecolor='yellow',facecolor='none')

                    ax.add_patch(rect)

    plt.suptitle(title, fontsize=14)

    plt.tight_layout()
plot_image(heads_small,'Spike Count <= 5')
plot_image(heads_normal,'Spike Count >= 30 & <= 60')
plot_image(heads_large,'Spike Count > 100')
fig = px.histogram(train_df, x="bbox_area",marginal="box")

fig.update_layout(

    xaxis = dict(

        title_text = "Bounding Box Area"), title = 'Bounding Box Area Distribution')

fig.show()
large_area = list(train_df[train_df.bbox_area > 100000]['image_id'].unique())

small_area = list(train_df[train_df.bbox_area <= 10]['image_id'].unique())
plot_image(large_area, title='Large Bounding Boxes')
plot_image(small_area, title='Small Bounding Boxes')
import albumentations

from albumentations import (

    HorizontalFlip, IAAPerspective, ShiftScaleRotate, CLAHE, RandomRotate90,

    Transpose, ShiftScaleRotate, Blur, OpticalDistortion, GridDistortion, HueSaturationValue,

    IAAAdditiveGaussianNoise, GaussNoise, MotionBlur, MedianBlur, RandomBrightnessContrast, IAAPiecewiseAffine,

    IAASharpen, IAAEmboss, Flip, OneOf, Compose,VerticalFlip,BboxParams,Rotate, ChannelShuffle, RandomRain)
# define functions for augmentation and display



def get_aug(aug, min_area=0., min_visibility=0.):

    return Compose(aug, bbox_params=BboxParams(format='pascal_voc', min_area=min_area, 

                                               min_visibility=min_visibility, label_fields=['labels']))



BOX_COLOR = (255,255,0)

def visualize_bbox(img, bbox, color=BOX_COLOR, thickness=2):

    x_min, y_min, x_max, y_max = bbox

    x_min, x_max, y_min, y_max = int(x_min), int(x_max), int(y_min), int(y_max)

    cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color=color, thickness=thickness)   

    return img



def visualize(annotations):

    img = annotations['image'].copy()

    for idx, bbox in enumerate(annotations['bboxes']):

        img = visualize_bbox(img, bbox)

    return img 

    

def aug_plots(image_id, aug, title=None):

    img_path = os.path.join(train_path,image_id +'.jpg')

    image = cv2.imread(img_path, cv2.IMREAD_COLOR)

    bbox = train_df[train_df['image_id'] == image_id][['bbox_xmin', 'bbox_ymin', 'bbox_xmax', 'bbox_ymax']].astype(np.int32).values



    labels = np.ones((len(bbox), ))

    annotations = {'image': image, 'bboxes': bbox, 'labels': labels}

    

    aug = get_aug(aug)

    augmented = aug(**annotations)

    visualize(augmented)

    

    fig = plt.figure(figsize = (15,7))

    ax = fig.add_subplot(1, 2, 1)

    ax.imshow(visualize(annotations))

    plt.title('Original')

    

    ax = fig.add_subplot(1, 2, 2)

    ax.imshow(visualize(augmented))

    plt.title(title)
aug_plots('b6ab77fd7',[VerticalFlip(p=1)], 'Vertical Flip')
aug_plots('69fc3d3ff',[HorizontalFlip(p=1)], 'Horizontal Flip')
aug_plots('69fc3d3ff',[Blur(blur_limit= 7,p=0.5)], 'Blur')
aug_plots('69fc3d3ff',[Rotate(p=0.5)], 'Rotate')
aug_plots('69fc3d3ff',[HueSaturationValue(hue_shift_limit=20, sat_shift_limit=50, val_shift_limit=50, p=1)], 'Hue Saturation')
aug_plots('69fc3d3ff',[ChannelShuffle(p=1)], 'Channel Shuffle')
aug_plots('69fc3d3ff',[GaussNoise()], 'Gauss Noise')
aug_plots('69fc3d3ff',[RandomRain(p=1, brightness_coefficient=0.9, drop_width=1, blur_value=5)], 'Random Rain')
import albumentations as A

from albumentations.pytorch.transforms import ToTensorV2



import torch

import torchvision



from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from torchvision.models.detection import FasterRCNN

from torchvision.models.detection.rpn import AnchorGenerator



from torch.utils.data import DataLoader, Dataset

from torch.utils.data.sampler import SequentialSampler
class WheatDataset(Dataset):



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



        boxes = records[['bbox_xmin', 'bbox_ymin', 'bbox_xmax', 'bbox_ymax']].values

        

        area = records['bbox_area'].values  # i already have the area in my dataframe

        area = torch.as_tensor(area, dtype=torch.float32)



        # there is only one class - so all will be 1

        labels = torch.ones((records.shape[0],), dtype=torch.int64) 

        

        # suppose all instances are not crowd

        iscrowd = torch.zeros((records.shape[0],), dtype=torch.int64)

        

        target = {}

        target['boxes'] = boxes

        target['labels'] = labels

        target['image_id'] = torch.tensor([index])

        target['area'] = area

        target['iscrowd'] = iscrowd



        if self.transforms:

            sample = {

                'image': image,

                'bboxes': target['boxes'],

                'labels': labels

            }

            sample = self.transforms(**sample)

            image = sample['image']

            

            target['boxes'] = torch.tensor(sample['bboxes'])

            target['boxes'] = target['boxes'].type(torch.float32)



        return image, target, image_id



    def __len__(self) -> int:

        return self.image_ids.shape[0]
# define transformation functions

def get_train_transform():

    return A.Compose([

        A.Flip(0.5),

        ToTensorV2(p=1.0)

    ], bbox_params={'format': 'pascal_voc', 'label_fields': ['labels']})



def get_valid_transform():

    return A.Compose([

        ToTensorV2(p=1.0)

    ], bbox_params={'format': 'pascal_voc', 'label_fields': ['labels']})
def get_model_instance_segmentation(num_classes):

    # load an instance segmentation model pre-trained pre-trained on COCO

    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)



    # get number of input features for the classifier

    in_features = model.roi_heads.box_predictor.cls_score.in_features

    # replace the pre-trained head with a new one

    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)



    return model
image_ids = train_df['image_id'].unique()

valid_ids = image_ids[-665:]

train_ids = image_ids[:-665]



val_set = train_df[train_df['image_id'].isin(valid_ids)]

train_set = train_df[train_df['image_id'].isin(train_ids)]
def collate_fn(batch):

    return tuple(zip(*batch))



train_dataset = WheatDataset(train_df, train_path, get_train_transform())

valid_dataset = WheatDataset(valid_df, train_path, get_valid_transform())





# split the dataset in train and test set

indices = torch.randperm(len(train_dataset)).tolist()



train_data_loader = DataLoader(

    train_dataset,

    batch_size=8,

    shuffle=False,

    num_workers=4,

    collate_fn=collate_fn

)



valid_data_loader = DataLoader(

    valid_dataset,

    batch_size=8,

    shuffle=False,

    num_workers=4,

    collate_fn=collate_fn

)
# train on the GPU or on the CPU, if a GPU is not available

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')



# define the number of classes

num_classes = 2 # one for wheat and one for background



# get the model using our helper function

model = get_model_instance_segmentation(num_classes)



# move model to the right device

model.to(device)



# construct an optimizer

params = [p for p in model.parameters() if p.requires_grad]

optimizer = torch.optim.SGD(params, lr=0.005,momentum=0.9, weight_decay=0.0005) 





num_epochs = 4



#for epoch in range(num_epochs):

    # train for one epoch, printing every 10 iterations

    #train_one_epoch(model, optimizer, train_data_loader, device, epoch, print_freq=10)

    # update the learning rate

    #lr_scheduler.step()

    # evaluate on the test dataset

   # evaluate(model, valid_data_loader, device=device)
from engine import train_one_epoch