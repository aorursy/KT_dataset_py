### Cloning Github Repository 
!git clone https://github.com/yhenon/pytorch-retinanet.git
### Copying RetinaNet Folder to root dir so we can import it easily
!cp -r /kaggle/working/pytorch-retinanet/retinanet ./
!pip install pycocotools
import os
import re
import cv2
import time
import numpy as np
import pandas as pd


import torch
import torch.optim as optim
import torchvision.transforms as T
from torchvision.utils import make_grid 
from torch.utils.data import DataLoader, Dataset

from retinanet import model
from retinanet.dataloader import collater, Resizer, Augmenter, Normalizer, UnNormalizer

import seaborn as sns
from matplotlib import pyplot as plt
%matplotlib inline

DIR = "../input/global-wheat-detection/"
DIR_TRAIN = DIR + "train"
DIR_TEST = DIR + "test"
### Loading Dataset
df = pd.read_csv(DIR + "train.csv")
df.head()
### Converting bbox list in appropriate form

df['x'] = -1
df['y'] = -1
df['w'] = -1
df['h'] = -1

def expand_bbox(x):
    r = np.array(re.findall("([0-9]+[.]?[0-9]*)", x))
    if len(r) == 0:
        r = [-1, -1, -1, -1]
    return r

df[['x', 'y', 'w', 'h']] = np.stack(df['bbox'].apply(lambda x: expand_bbox(x)))
df.drop(columns=['bbox'], inplace=True)
df['x'] = df['x'].astype(np.float)
df['y'] = df['y'].astype(np.float)
df['w'] = df['w'].astype(np.float)
df['h'] = df['h'].astype(np.float)

df.head()

### Null Values, Unique Images, etc.

unq_values = df["image_id"].unique()
print("Total Records: ", len(df))
print("Unique Images: ",len(unq_values))

null_values = df.isnull().sum(axis = 0)
print("\n> Null Values in each column <")
print(null_values)


### Data Sources

sources = df["source"].unique()
print("Total Sources: ",len(sources))
print("\n> Sources <\n",sources)
### Visualizing Source Distribution

plt.figure(figsize=(14,8))
plt.title('Source Distribution', fontsize= 20)
sns.countplot(x = "source", data = df)
### Splitting Train Dataset into train - val (80:20)

images = df['image_id'].unique()
valid_imgs = images[-674:]
train_imgs = images[:-674]

valid_df = df[df['image_id'].isin(valid_imgs)]
train_df = df[df['image_id'].isin(train_imgs)]

### Function to plot image

def plot_img(image_name):
    
    fig, ax = plt.subplots(1, 2, figsize = (10, 10))
    ax = ax.flatten()
    
    records = df[df['image_id'] == image_name]
    img_path = os.path.join(DIR_TRAIN, image_name + ".jpg")
    
    image = cv2.imread(img_path, cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
    image /= 255.0
    image2 = image
    
    ax[0].set_title('Original Image')
    ax[0].imshow(image)
    
    for idx, row in records.iterrows():
        box = row[['x', 'y', 'w', 'h']].values
        xmin = box[0]
        ymin = box[1]
        width = box[2]
        height = box[3]
        
        cv2.rectangle(image2, (int(xmin),int(ymin)), (int(xmin + width),int(ymin + height)), (255,0,0), 3)
    
    ax[1].set_title('Image with Bondary Box')
    ax[1].imshow(image2)

    plt.show()
    
### Pass any image id as parameter

plot_img("0126b7d11")
plot_img("00333207f")
### Creating targets for model using Dataset Class

class GWD(Dataset):

    def __init__(self, dataframe, image_dir, mode = "train", transforms = None):
        
        super().__init__()
        self.image_ids = dataframe['image_id'].unique()
        self.df = dataframe
        self.image_dir = image_dir
        self.mode = mode
        self.transforms = transforms

    def __getitem__(self, index: int):

        # Retriving image id and records from df
        image_id = self.image_ids[index]
        records = self.df[self.df['image_id'] == image_id]

        # Loading Image
        image = cv2.imread(f'{self.image_dir}/{image_id}.jpg', cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        image /= 255.0

        # If mode is set to train, then only we create targets
        if self.mode == "train" or self.mode == "valid":

            # Converting xmin, ymin, w, h to x1, y1, x2, y2
            boxes = np.zeros((records.shape[0], 5))
            boxes[:, 0:4] = records[['x', 'y', 'w', 'h']].values
            boxes[:, 2] = boxes[:, 0] + boxes[:, 2]
            boxes[:, 3] = boxes[:, 1] + boxes[:, 3]
            boxes[:, 4] = 1 # This is for label, as we have only 1 class, it is always 1
            
            # Applying Transforms
            sample = {'img': image, 'annot': boxes}
                
            if self.transforms:
                sample = self.transforms(sample)

            return sample
        
        elif self.mode == "test":
            
            # We just need to apply transoforms and return image
            if self.transforms:
                
                sample = {'img' : image}
                sample = self.transforms(sample)
                
            return sample
        

    def __len__(self) -> int:
        return self.image_ids.shape[0]
    
### Preparing Datasets and Dataloaders for Training 

# Dataset Object
train_dataset = GWD(train_df, DIR_TRAIN, mode = "train", transforms = T.Compose([Augmenter(), Normalizer(), Resizer()]))
valid_dataset = GWD(valid_df, DIR_TRAIN, mode = "valid", transforms = T.Compose([Normalizer(), Resizer()]))

# DataLoaders
train_data_loader = DataLoader(
    train_dataset,
    batch_size = 8,
    shuffle = True,
    num_workers = 4,
    collate_fn = collater
)

valid_data_loader = DataLoader(
    valid_dataset,
    batch_size = 8,
    shuffle = True,
    num_workers = 4,
    collate_fn = collater
)


test_data_loader = DataLoader(
    valid_dataset,
    batch_size = 1,
    shuffle = True,
    num_workers = 4,
    collate_fn = collater
)

### Utilize GPU if available

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
torch.cuda.empty_cache()
### I am using Pre-trained Resnet50 as backbone

retinanet = model.resnet50(num_classes = 2, pretrained = True)

# Loading Pre-trained model - if you load pre-trained model, comment above line.
#retinanet = torch.load("path_to_.pt_file")
### Preparing model for training

# Defininig Optimizer
optimizer = torch.optim.Adam(retinanet.parameters(), lr = 0.0001)

# Learning Rate Scheduler
#lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = 5, gamma=0.5)

retinanet.to(device)

#No of epochs
epochs = 15

### One Epoch - Train

def train_one_epoch(epoch_num, train_data_loader):
    
    print("Epoch - {} Started".format(epoch_num))
    st = time.time()
    
    retinanet.train()
    
    epoch_loss = []

    for iter_num, data in enumerate(train_data_loader):
                
        # Reseting gradients after each iter
        optimizer.zero_grad()
            
        # Forward
        classification_loss, regression_loss = retinanet([data['img'].cuda().float(), data['annot'].cuda().float()])
                
        # Calculating Loss
        classification_loss = classification_loss.mean()
        regression_loss = regression_loss.mean()

        loss = classification_loss + regression_loss

        if bool(loss == 0):
            continue
                
        # Calculating Gradients
        loss.backward()

        # Gradient Clipping
        torch.nn.utils.clip_grad_norm_(retinanet.parameters(), 0.1)
                
        # Updating Weights
        optimizer.step()

        #Epoch Loss
        epoch_loss.append(float(loss))

            
        print(
            'Epoch: {} | Iteration: {} | Classification loss: {:1.5f} | Regression loss: {:1.5f} | Running loss: {:1.5f}'.format(
                epoch_num, iter_num, float(classification_loss), float(regression_loss), np.mean(epoch_loss)))

        del classification_loss
        del regression_loss
        
    # Update the learning rate
    #if lr_scheduler is not None:
        #lr_scheduler.step()
        
    et = time.time()
    print("\n Total Time - {}\n".format(int(et - st)))
    
        
### One Epoch - Valid

def valid_one_epoch(epoch_num, valid_data_loader):
    
    print("Epoch - {} Started".format(epoch_num))
    st = time.time()
    
    epoch_loss = []

    for iter_num, data in enumerate(valid_data_loader):
                
        with torch.no_grad():
            
            # Forward
            classification_loss, regression_loss = retinanet([data['img'].cuda().float(), data['annot'].cuda().float()])

            # Calculating Loss
            classification_loss = classification_loss.mean()
            regression_loss = regression_loss.mean()
            loss = classification_loss + regression_loss

            #Epoch Loss
            epoch_loss.append(float(loss))

            print(
                'Epoch: {} | Iteration: {} | Classification loss: {:1.5f} | Regression loss: {:1.5f} | Running loss: {:1.5f}'.format(
                    epoch_num, iter_num, float(classification_loss), float(regression_loss), np.mean(epoch_loss)))

            del classification_loss
            del regression_loss
        
    et = time.time()
    print("\n Total Time - {}\n".format(int(et - st)))
    
    # Save Model after each epoch
    torch.save(retinanet, "retinanet_gwd.pt")
    
        
### Training Loop
for epoch in range(epochs):
    
    # Call train function
    train_one_epoch(epoch, train_data_loader)
    
    # Call valid function
    valid_one_epoch(epoch, valid_data_loader)

### Sample Results
retinanet.eval()
unnormalize = UnNormalizer()

for iter_num, data in enumerate(test_data_loader):
    
    # Getting Predictions
    scores, classification, transformed_anchors = retinanet(data['img'].cuda().float())
    
    idxs = np.where(scores.cpu()>0.5)
    img = np.array(255 * unnormalize(data['img'][0, :, :, :])).copy()
    
    img[img<0] = 0
    img[img>255] = 255

    img = np.transpose(img, (1, 2, 0))

    img = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2RGB)
    
    fig, ax = plt.subplots(1, 1, figsize=(16, 8))
    
    for j in range(idxs[0].shape[0]):
        bbox = transformed_anchors[idxs[0][j], :]
        x1 = int(bbox[0])
        y1 = int(bbox[1])
        x2 = int(bbox[2])
        y2 = int(bbox[3])

        cv2.rectangle(img, (x1, y1), (x2, y2), color = (0, 0, 255), thickness = 2)
        
    ax.imshow(img)
    
    break
    
