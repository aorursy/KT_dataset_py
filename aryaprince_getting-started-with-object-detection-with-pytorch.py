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
import cv2

import os

import re

import torch

import torchvision

from torchvision import transforms 

from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from torchvision.models.detection import FasterRCNN

from torchvision.models.detection.rpn import AnchorGenerator

from torch.utils.data import DataLoader, Dataset

from torch.utils.data.sampler import SequentialSampler

from matplotlib import pyplot as plt
WEIGHTS_FILE = '/kaggle/input/fasterrcnn/fasterrcnn_resnet50_fpn_best.pth'
train_df = pd.read_csv("/kaggle/input/global-wheat-detection/train.csv")

submit = pd.read_csv("/kaggle/input/global-wheat-detection/sample_submission.csv")
train_df.head()
train_df=train_df.drop(columns=['width','height','source']) #Drop unwanted columns
train_df['image_id'].nunique() # There are total 3373 unique image in training dataset
(train_df['image_id'].value_counts()).max()  # maximum number of boxes in a single image are 116
(train_df['image_id'].value_counts()). min() # Minimum number of box in a single image is 1
train_df['x'] = -1

train_df['y'] = -1

train_df['w'] = -1

train_df['h'] = -1



def expand_bbox(x):

    r = np.array(re.findall("([0-9]+[.]?[0-9]*)", x))

    if len(r) == 0:

        r = [-1, -1, -1, -1]

    return r
train_df[['x', 'y', 'w', 'h']] = np.stack(train_df['bbox'].apply(lambda x: expand_bbox(x))) ##Lets convert the Box in 

train_df['x'] = train_df['x'].astype(np.float)                                        #in our desired formate    

train_df['y'] = train_df['y'].astype(np.float)

train_df['w'] = train_df['w'].astype(np.float)

train_df['h'] = train_df['h'].astype(np.float)
train_df.head() 
image_ids = train_df['image_id'].unique()

valid_ids = image_ids[-665:]

train_ids = image_ids[:-665]



valid_df = train_df[train_df['image_id'].isin(valid_ids)]

train_df = train_df[train_df['image_id'].isin(train_ids)]
trans = transforms.Compose([transforms.ToTensor()])   #Apply transform to image 
class WheatDataset(Dataset):



    def __init__(self, dataframe, image_dir, transforms=None,train=True):

        super().__init__()



        self.image_ids = dataframe['image_id'].unique()

        self.df = dataframe

        self.image_dir = image_dir

        self.transforms = transforms

        self.train=train



    def __len__(self) -> int:

        return self.image_ids.shape[0]



    def __getitem__(self, index: int):



        image_id = self.image_ids[index]

        image = cv2.imread(f'{self.image_dir}/{image_id}.jpg', cv2.IMREAD_COLOR)

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)

        image /= 255.0

        if self.transforms is not None:  #Apply transformation

            image = self.transforms(image)

        if(self.train==False):  # For test data

            return image, image_id

        #Else for train and validation data

        records = self.df[self.df['image_id'] == image_id]   

        boxes = records[['x', 'y', 'w', 'h']].values

        boxes[:, 2] = boxes[:, 0] + boxes[:, 2]

        boxes[:, 3] = boxes[:, 1] + boxes[:, 3]

        boxes = torch.as_tensor(boxes, dtype=torch.float32)

        

        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])

        area = torch.as_tensor(area, dtype=torch.float32)



        # there is only one class

        labels = torch.ones((records.shape[0],), dtype=torch.int64)

        

        # suppose all instances are not crowd

        iscrowd = torch.zeros((records.shape[0],), dtype=torch.int64)

        

        target = {}

        target['boxes'] = boxes

        target['labels'] = labels

        target['image_id'] = torch.tensor([index])

        target['area'] = area

        target['iscrowd'] = iscrowd



        return image, target,image_id  
train_dir = '/kaggle/input/global-wheat-detection/train'

test_dir = '/kaggle/input/global-wheat-detection/test'
class Averager:      ##Return the average loss 

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



train_dataset = WheatDataset(train_df, train_dir, trans,True)

valid_dataset = WheatDataset(valid_df, train_dir, trans,True)





# split the dataset in train and test set

indices = torch.randperm(len(train_dataset)).tolist()



train_data_loader = DataLoader(

    train_dataset,

    batch_size=16,

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



#device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
images, targets, image_ids = next(iter(train_data_loader))

images = list(image.to(device) for image in images)

targets = [{k: v.to(device) for k, v in t.items()} for t in targets]



boxes = targets[4]['boxes'].cpu().numpy().astype(np.int32)

sample = images[4].permute(1,2,0).cpu().numpy()



fig, ax = plt.subplots(1, 1, figsize=(16, 8))



for box in boxes:

    cv2.rectangle(sample,

                  (box[0], box[1]),

                  (box[2], box[3]),

                  (220, 0, 0), 3)

    

ax.set_axis_off()

ax.imshow(sample)
# load a model; pre-trained on COCO

model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=False, pretrained_backbone=False)
num_classes = 2  # 1 class (wheat) + background



# get number of input features for the classifier

in_features = model.roi_heads.box_predictor.cls_score.in_features



# replace the pre-trained head with a new one

model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)



# Load the trained weights

model.load_state_dict(torch.load(WEIGHTS_FILE))  ##Load pre trained weights

#model.eval()



#x = model.to(device)
model.train()

model.to(device)

params = [p for p in model.parameters() if p.requires_grad]

optimizer = torch.optim.SGD(params, lr=0.01, momentum=0.9, weight_decay=0.00001)

lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)

#lr_scheduler = None



num_epochs = 5



loss_hist = Averager()

itr = 1



for epoch in range(num_epochs):

    loss_hist.reset()

    

    for images, targets, image_ids in train_data_loader:

        

        images = list(image.to(device) for image in images)

        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]



        loss_dict = model(images, targets)   ##Return the loss



        losses = sum(loss for loss in loss_dict.values())

        loss_value = losses.item()



        loss_hist.send(loss_value)  #Average out the loss



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
test_dataset = WheatDataset(submit,test_dir, trans,False)
test_data_loader = DataLoader( test_dataset, batch_size=8, shuffle=False)  ##Test dataloader
detection_threshold = 0.45
def format_prediction_string(boxes, scores): ## Define the formate for storing prediction results

    pred_strings = []

    for j in zip(scores, boxes):

        pred_strings.append("{0:.4f} {1} {2} {3} {4}".format(j[0], j[1][0], j[1][1], j[1][2], j[1][3]))



    return " ".join(pred_strings)
## Lets make the prediction

results=[]

model.eval()



for images, image_ids in test_data_loader:    



    images = list(image.to(device) for image in images)

    outputs = model(images)



    for i, image in enumerate(images):



        boxes = outputs[i]['boxes'].data.cpu().numpy()    ##Formate of the output's box is [Xmin,Ymin,Xmax,Ymax]

        scores = outputs[i]['scores'].data.cpu().numpy()

        

        boxes = boxes[scores >= detection_threshold].astype(np.int32) #Compare the score of output with the threshold and

        scores = scores[scores >= detection_threshold]                    #slelect only those boxes whose score is greater

                                                                          # than threshold value

        image_id = image_ids[i]

        

        boxes[:, 2] = boxes[:, 2] - boxes[:, 0]         

        boxes[:, 3] = boxes[:, 3] - boxes[:, 1]         #Convert the box formate to [Xmin,Ymin,W,H]

        

        

            

        result = {                                     #Store the image id and boxes and scores in result dict.

            'image_id': image_id,

            'PredictionString': format_prediction_string(boxes, scores)

        }



        

        results.append(result)              #Append the result dict to Results list
test_df = pd.DataFrame(results, columns=['image_id', 'PredictionString'])

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
test_df.to_csv('submission.csv', index=False)