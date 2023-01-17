import os

import time

import shutil



import numpy as np

import pandas as pd



import torch

import torchvision

from torchvision import transforms

from torch.utils.data import DataLoader, Dataset

from torchvision.models.detection.faster_rcnn import FastRCNNPredictor, FasterRCNN

from torchvision.models.detection.backbone_utils import resnet_fpn_backbone

from tqdm.notebook import tqdm



import albumentations

from albumentations import *

from albumentations.pytorch.transforms import ToTensorV2

import cv2

import math

import matplotlib.pyplot as plt

from matplotlib import patches

import seaborn as sns

%matplotlib inline

print('Ready...')
DATA_DIR  = '../input/global-wheat-detection/train/'

TEST_DIR  = '../input/global-wheat-detection/test/'

train_df_path = '../input/global-wheat-detection/train.csv'

test_df_path = '../input/global-wheat-detection/sample_submission.csv'

List_Data_dir = os.listdir(DATA_DIR)
raw = pd.read_csv(train_df_path)

raw
raw.describe()



# all images have resolution 1024 x 1024
print(f'Total number of train images: {raw.image_id.nunique()}')

print(f'Total number of test images: {len(os.listdir(TEST_DIR))}')
plt.figure(figsize=(15,8))

plt.title('Wheat Distribution', fontsize= 20)

sns.countplot(x="source", data=raw)



# based on the chart, there are seven types of wheat from images data, with the most types 'ethz_1' and the least is 'inrae_1'
# Extract bbox column to xmin, ymin, width, height, then create xmax, ymax, and area columns



raw[['xmin','ymin','w','h']] = pd.DataFrame(raw.bbox.str.strip('[]').str.split(',').tolist()).astype(float)

raw['xmax'], raw['ymax'], raw['area'] = raw['xmin'] + raw['w'], raw['ymin'] + raw['h'], raw['w'] * raw['h']

raw
def show_image(image_id):

    

    fig, ax = plt.subplots(1, 2, figsize = (24, 24))

    ax = ax.flatten()

    

    bbox = raw[raw['image_id'] == image_id ]

    img_path = os.path.join(DATA_DIR, image_id + '.jpg')

    

    image = cv2.imread(img_path, cv2.IMREAD_COLOR)

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)

    image /= 255.0

    image2 = image

    

    ax[0].set_title('Original Image')

    ax[0].imshow(image)

    

    for idx, row in bbox.iterrows():

        x1 = row['xmin']

        y1 = row['ymin']

        x2 = row['xmax']

        y2 = row['ymax']

        label = row['source']

        

        cv2.rectangle(image2, (int(x1),int(y1)), (int(x2),int(y2)), (255,255,255), 2)

        font = cv2.FONT_HERSHEY_SIMPLEX

        cv2.putText(image2, label, (int(x1),int(y1-10)), font, 1, (255,255,255), 2)

    

    ax[1].set_title('Image with Bondary Box')

    ax[1].imshow(image2)



    plt.show()
show_image(raw.image_id.unique()[91])
show_image(raw.image_id.unique()[1231])
show_image(raw.image_id.unique()[3121])
def get_bboxes(bboxes, col, bbox_format = 'pascal_voc', color='white'):

    for i in range(len(bboxes)):

        x_min = bboxes[i][0]

        y_min = bboxes[i][1]

        x_max = bboxes[i][2]

        y_max = bboxes[i][3]

        width = x_max - x_min

        height = y_max - y_min

        rect = patches.Rectangle((x_min, y_min), 

                                 width, height, 

                                 linewidth=2, 

                                 edgecolor=color, 

                                 facecolor='none')

        col.add_patch(rect)
def augmented_images(image, augment):

    

    fig, ax = plt.subplots(1, 2, figsize = (24, 24))

    ax = ax.flatten()

    

    image_data = raw[raw['image_id'] == image]

    bbox = image_data[['xmin', 'ymin', 'xmax', 'ymax']].astype(np.int32).values

    labels = np.ones((len(bbox), ))



    image = cv2.imread(os.path.join(DATA_DIR + '/{}.jpg').format(image), cv2.IMREAD_COLOR)

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)

    image /= 255.0

    

    get_bboxes(bbox, ax[0], color='white')

    

    ax[0].set_title('Original Image with Bounding Boxes')

    ax[0].imshow(image)

    

    aug = albumentations.Compose([augment], 

                         bbox_params={'format': 'pascal_voc', 'label_fields':['labels']})



    

    aug_result = aug(image=image, bboxes=bbox, labels=labels)



    aug_image = aug_result['image']

    get_bboxes(aug_result['bboxes'], ax[1], color='red')

    

    ax[1].set_title('Augmented Image with Bounding Boxes')

    ax[1].imshow(aug_image)

    

    plt.show()
# HorizontalFlip Augmentation

augmented_images(raw.image_id.unique()[1230], albumentations.HorizontalFlip(p=1))
# VerticalFlip Augmentation

augmented_images(raw.image_id.unique()[2110], albumentations.VerticalFlip(p=1))
# Change Color to gray

augmented_images(raw.image_id.unique()[1212], albumentations.ToGray(p=1))
# Random Change Brightness Contrast

augmented_images(raw.image_id.unique()[1230], albumentations.RandomBrightnessContrast(p=1))
class wheatdataset_train(Dataset):

       

    def __init__(self, dataframe, data_dir, transforms=None):

        super().__init__()

        self.df = dataframe 

        self.image_list = list(self.df['image_id'].unique())

        self.image_dir = data_dir

        self.transforms = transforms

    

    def __len__(self):

        return len(self.image_list)

        

    def __getitem__(self, idx):

        

        image_id = self.image_list[idx]

        image_data = self.df.loc[self.df['image_id'] == image_id]

        boxes = torch.as_tensor(np.array(image_data[['xmin','ymin','xmax','ymax']]), 

                                dtype=torch.float32)

        area = torch.tensor(np.array(image_data['area']), dtype=torch.int64) 

        labels = torch.ones((image_data.shape[0],), dtype=torch.int64)

        iscrowd = torch.zeros((image_data.shape[0],), dtype=torch.uint8)

         

        target = {}

        target['boxes'] = boxes

        target['area'] = area

        target['labels'] = labels

        target['iscrowd'] = iscrowd

        

        image = cv2.imread((self.image_dir + '/' + image_id + '.jpg'), cv2.IMREAD_COLOR)

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)

        image /= 255.0

        

        if self.transforms:

            

            image_transforms = {

                                'image': image,

                                'bboxes': target['boxes'],

                                'labels': labels

                                 }

            

            image_transforms = self.transforms(**image_transforms)

            image = image_transforms['image']

            

            target['boxes'] = torch.as_tensor(image_transforms['bboxes'], dtype=torch.float32)

                 

        return image, target
# Albumentations



def get_train_transform():

    return albumentations.Compose([

        #albumentations.Resize(p=1, height=512, width=512),

        albumentations.ToGray(p=0.5),

        albumentations.Flip(p=0.5),

        albumentations.RandomBrightnessContrast(p=0.5),

        ToTensorV2(p=1.0)

    ], bbox_params={'format': 'pascal_voc', 'label_fields': ['labels']})





def get_test_transform():

    return albumentations.Compose([

        ToTensorV2(p=1.0)

    ])





def collate_fn(batch):

    return tuple(zip(*batch))
train_data = wheatdataset_train(raw, DATA_DIR, get_train_transform())

train_dataloader = DataLoader(train_data, batch_size=16,shuffle=True, num_workers=4,collate_fn=collate_fn)
len(train_data)
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

torch.cuda.empty_cache()

print(device)
def train_model():

    

    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)

    num_classes = 2  

    in_features = model.roi_heads.box_predictor.cls_score.in_features

    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    

    return model



def train(data_loader, epoch):

        

    model = train_model()

    model.to(device)

    params = [p for p in model.parameters() if p.requires_grad]

    optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)

    model.parameters





    total_train_loss = []

    itr = 1

    train_loss_threshold = math.inf



    for epoch in tqdm(range(epoch)):

        

        print(f'Epoch :{epoch + 1}')

        start_time = time.time()

        train_loss = []

        model.train()

        for images, targets in tqdm(data_loader):

            

            images = list(image.to(device) for image in images)

            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]



            loss_dict = model(images, targets)



            losses = sum(loss for loss in loss_dict.values())

            

            loss_value = losses.item()

            

            train_loss.append(losses.item())

            optimizer.zero_grad()

            losses.backward()

            optimizer.step()

            

            if itr % 50 == 0:

                print(f"Iteration #{itr} loss: {loss_value:.4f}")



            itr += 1

    

        

        epoch_train_loss = np.mean(train_loss)

        total_train_loss.append(epoch_train_loss)

        print(f'Epoch train loss is {epoch_train_loss:.4f}')

        time_elapsed = time.time() - start_time

        print('{:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

        if epoch_train_loss < train_loss_threshold:

            train_loss_threshold = epoch_train_loss

            torch.save(model.state_dict(), 'fasterrcnn_resnet50_fpn_{0:.3f}.pth'.format(epoch_train_loss))



    #visualize

    plt.figure(figsize=(12,6))

    plt.title('Train Loss', fontsize= 20)

    plt.plot(total_train_loss)

    plt.xlabel('epochs')

    plt.ylabel('loss')

    plt.show()
num_epochs = 17

train(train_dataloader, num_epochs)
# total = 0

#     sum_loss = 0

#     correct = 0 

#     for images_val, targets_val, image_ids_val in valid_data_loader:

#         images_val = list(image.to(device) for image in images_val)

#         targets_val = [{k: v.to(device) for k, v in t.items()} for t in targets_val]



#         loss_dict_val = model(images_val, targets_val)



#         losses_val = sum(loss for loss in loss_dict_val.values())

#         val_loss = losses_val.item()

#     print("val_loss %.5f"%(val_loss))

#     if val_loss < pre_valid_loss:

#         pre_valid_loss = val_loss

#         torch.save(model.state_dict(), 'fasterrcnn_resnet50_fpn_{0:.3f}.pth'.format(val_loss))