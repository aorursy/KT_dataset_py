import pandas as pd
import numpy as np
import cv2
import os
import re

from PIL import Image

import albumentations as A #Package of transformations
from albumentations.pytorch.transforms import ToTensorV2

import torch
import torchvision

from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator

from torch.utils.data import DataLoader, Dataset #Create an efficient dataloader set to feed images to the model
from torch.utils.data.sampler import SequentialSampler

from matplotlib import pyplot as plt #Allows us to create a sample image to test the model is working correctly

DIR_INPUT = '/kaggle/input/global-wheat-detection' #Base directory for this challenge
DIR_TRAIN = f'{DIR_INPUT}/train' #Base directory where both train images and train metadata is located
DIR_TEST = f'{DIR_INPUT}/test' #Base directory where 10 test images are located. The rest is kept private by the organisers
DIR_WEIGHTS = '/kaggle/input/fasterrcnn'

WEIGHTS_FILE = f'{DIR_WEIGHTS}/fasterrcnn_resnet50_fpn_best.pth'
test_df = pd.read_csv(f'{DIR_INPUT}/sample_submission.csv')
test_df
class WheatTestDataset(Dataset):

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
# Albumentations

#Create a function to applies some list of transformations to each image with certain probabilities
    #Be mindful that these transforms are not independent, the function may apply more than one.
    #Two transforms each with 0.5 probability have a 0.5 * 0.5 probability (25%) of applying both transforms. 
def get_train_transform():
    return A.Compose([
        A.Flip(p = 0.5),
        A.Blur(p=0.5),
        ToTensorV2(p=1.0)
    ], bbox_params={'format': 'pascal_voc', 'label_fields': ['labels']})

def get_valid_transform():
    return A.Compose([
        ToTensorV2(p=1.0)
    ], bbox_params={'format': 'pascal_voc', 'label_fields': ['labels']})

def get_test_transform():
    return A.Compose([
        # A.Resize(512, 512),
        ToTensorV2(p=1.0)
    ])
# load a model; pre-trained on COCO
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=False, pretrained_backbone=False)
#Set device equal to GPU if GPU is selected in the "settings" tab on the right, under Accelerator, otherwise set device to CPU
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

num_classes = 2  # 1 class (wheat) + background

# get number of input features for the classifier
in_features = model.roi_heads.box_predictor.cls_score.in_features

# replace the pre-trained head with a new one
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

# Load the trained weights
model.load_state_dict(torch.load(WEIGHTS_FILE))
model.eval()

x = model.to(device)
def collate_fn(batch):
    return tuple(zip(*batch))
test_dataset = WheatTestDataset(test_df, DIR_TEST, get_test_transform())

test_data_loader = DataLoader(
    test_dataset,
    batch_size=4,
    shuffle=False,
    num_workers=4,
    drop_last=False,
    collate_fn=collate_fn
)
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
            #print(box)
            result = {
                'image_id': image_id,
                'width': 1024,
                'height': 1024,
                'source': 'nvnn',
                'x': box[0],
                'y': box[1],
                'w': box[2],
                'h': box[3]
            }
            testdf_psuedo.append(result)
test_df_pseudo = pd.DataFrame(testdf_psuedo, columns=['image_id', 'width', 'height', 'source', 'x', 'y', 'w', 'h'])
test_df_pseudo.head()
train_df = pd.read_csv(f'{DIR_INPUT}/train.csv') #Concatinate the base directory and file name and create a dataframe
print(train_df) #Print the dataframe to check the import worked

#Count the number of unique images in the training data
train_df.image_id.nunique()

#Add 4 new columns and set their value to -1.
    #This is just a placeholder, we could use any value. It will get replaced by the expand_bbox function
train_df['x'] = -1
train_df['y'] = -1
train_df['w'] = -1
train_df['h'] = -1

def expand_bbox(x):
    r = np.array(re.findall("([0-9]+[.]?[0-9]*)", x))
    if len(r) == 0:
        r = [-1, -1, -1, -1]
    return r

#Replace the values in the x, y, w, h columns by the values from the bbox column.
    #Lambda x is saying "for each x run the function expand_bbox"
    #We are setting "x" to be a cell in the train_df bbox column.
train_df[['x', 'y', 'w', 'h']] = np.stack(train_df['bbox'].apply(lambda x: expand_bbox(x)))

#We now have duplicated content, the original string of concatinated bounding box dimensions and the new seperated values.
#Therefore, we can drop the original bbox column and just retain the 4 new columns.
train_df.drop(columns=['bbox'], inplace=True)

#Columns x, y, w, h are "object" types. We can convert these to floats as below...
train_df['x'] = train_df['x'].astype(np.float)
train_df['y'] = train_df['y'].astype(np.float)
train_df['w'] = train_df['w'].astype(np.float)
train_df['h'] = train_df['h'].astype(np.float)
#Create a numpy array of only the unique image_ids. Most images have multiple bounding boxes so the original train_df...
#...has repeated image_ids 
image_ids = train_df['image_id'].unique()

#Seperate those image ids into a training and validation set.
    #The original dataset has 3373. 674 is approx 20%, a common split for train/valid is 80/20 as here.
valid_ids = image_ids[-674:]
train_ids = image_ids #[:-674]
train_ids.size
train_df
#For those image ids that are in valid_ids we want the full info about each image from the original full train_df.
    #This includes all the bounding boxes and the size of the image
valid_df = train_df[train_df['image_id'].isin(valid_ids)]

#In the usual case the same is done for the training portion.
train_df = train_df[train_df['image_id'].isin(train_ids)]
    #However, now we have 10 more images with Pseudo labels we can add those to the training data

frames = [train_df, test_df_pseudo]
print(frames)
train_df = pd.concat(frames)
#In this case we are using tail rather than head as we attached the test images to the end of the dataframe.
    #This allows us to check that everything was joined correctly
train_df.tail()
valid_df.shape, train_df.shape
class WheatDataset(Dataset):

    def __init__(self, dataframe, image_dir, transforms):
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
        
        #Standardise the image pixel data
        image /= 255.0

        boxes = records[['x', 'y', 'w', 'h']].values
        boxes[:, 2] = boxes[:, 0] + boxes[:, 2]
        boxes[:, 3] = boxes[:, 1] + boxes[:, 3]
        
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        area = torch.as_tensor(area, dtype=torch.float32)

        # there is only one class (wheat or not)
        labels = torch.ones((records.shape[0],), dtype=torch.int64)
        
        # suppose all instances are not crowd
        iscrowd = torch.zeros((records.shape[0],), dtype=torch.int64)
        
        target = {}
        target['boxes'] = boxes
        target['labels'] = labels
        # target['masks'] = None
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
            
            target['boxes'] = torch.stack(tuple(map(torch.tensor, zip(*sample['bboxes'])))).permute(1, 0)

        return image, target, image_id

    def __len__(self) -> int:
        return self.image_ids.shape[0]
# load a model; pre-trained on COCO
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
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
        
#Create a training dataset.
    #We pass the train_df (which now contains all information for the 80% of the images we randomly chose)...
        #...we also pass the directory for the image jpg files, and the function to apply any transformations.
train_dataset = WheatDataset(train_df, DIR_TRAIN, get_train_transform())

#The same arguments are passed to the same function but this time it is for the validation portion of the whole dataset
valid_dataset = WheatDataset(valid_df, DIR_TRAIN, get_valid_transform())


# split the dataset in train and test set
indices = torch.randperm(len(train_dataset)).tolist()

#Create a dataloader to efficiently load the data to the model. We send train_dataset that we just created to the Dataloader
train_data_loader = DataLoader(
    train_dataset,
    batch_size=16,
    shuffle=False,
    num_workers=4, #num_workers defines how many cores of the device we want to use, in this case the GPU
    collate_fn=collate_fn
)

#A dataloader is also created for the validation set.
valid_data_loader = DataLoader(
    valid_dataset,
    batch_size=8,
    shuffle=False,
    num_workers=4,
    collate_fn=collate_fn
)
#Iterate through the train_data_loader object, row by row, and for each row we unpack the information into image, targets, and image_id
    #images is a tuple where each item is a list of all the pixel values for a particular image
    #targets is a tuple where each item is a list of the x, y, w, h dimensions for each bounding box
images, targets, image_ids = next(iter(train_data_loader))

#For each image (as described above) we send the image to the device (likely CPU) and then add the values to a list.
images = list(image.to(device) for image in images)

targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
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
from tensorflow import keras

model.train()
#Send the model to the device - GPU
model.to(device)

#Create a list of parameters to be used for calculating the loss under our optimizer.
    #Only include parameters that require the gradient to be altered.
params = [p for p in model.parameters() if p.requires_grad]

#Create a Stochastic Gradient Descent optimiser (SGD).
    #all parameters can be changed but momentum is commonly 0.9.
optimizer = torch.optim.SGD(params, lr=0.0075, momentum=0.9, weight_decay=0.0005)

#Peter originally included the StepLR learning rate annealer but I added two more as optional choices for you.
    #Only the final will be used so there is no benefit in choosing multiple
    #Similar to the optimiser, these hyperparameters can be changed
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)
#lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = 1)
#lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience = 2, verbose = True)

#An epoch is one full pass through the whole dataset by your model.
    #Increasing epochs will increase accuracy, but at some point will lead to overfitting.
    #There is a balance between too many and too few epochs which can be visualised when comparing the validation loss.
    #More epochs will take longer and more importantaly use precious GPU allocation.
num_epochs = 5
loss_hist = Averager()
itr = 1

for epoch in range(num_epochs):
    loss_hist.reset()

    #For each image in train_data perform the following actions...
    for images, targets, image_ids in train_data_loader:

        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)

        losses = sum(loss for loss in loss_dict.values())
        loss_value = losses.item()

        loss_hist.send(loss_value)

        #We must zero out the gradients after each image otherwise the gradients will accumulate
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        #For every 50 iterations print a progress message
        if itr % 50 == 0:
            print(f"Iteration #{itr} loss: {loss_value}")

        itr += 1

    # update the learning rate if a learning rate annealer exists
    if lr_scheduler is not None:
        lr_scheduler.step(loss_value)

    print(f"Epoch #{epoch} loss: {loss_hist.value}")
#Replace the values of images, targets, and image_ids from containing the training data to now contain only those images in the validation set
images, targets, image_ids = next(iter(valid_data_loader))
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
#Saved the model, including its weights to a new file.
    #This allows us to transfer the model to others without the need for them to also use GPU time to attain the weights
torch.save(model.state_dict(), 'fasterrcnn_resnet50_fpn.pth')