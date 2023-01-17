import os 

import cv2

import numpy as np

import pandas as pd 

import torch

from torch.utils.data import Dataset

from torchvision import transforms

from PIL import Image

from sklearn.model_selection import train_test_split 

import albumentations as albu

from albumentations.pytorch.transforms import ToTensor

from torchvision import models 

from tqdm import tqdm_notebook as tqdm

from torchvision import transforms 

import random 

import torch.optim as optim

import torch.nn as nn
import cv2 

import numpy as np

from glob import glob



class Detection(object):

    def __init__(self):

        pass 

    def process(self, file_name):

        image = cv2.imread(file_name)

        image = self.detect(image)

        return image



    @staticmethod

    def detect(image):

        blur = cv2.GaussianBlur(image, (3, 3), 0)

        hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)

        mask2 = cv2.inRange(hsv, np.array([2, 0, 0]), np.array([20, 255, 255]))

        kernel = np.ones((5, 5))

        dilation = cv2.dilate(mask2, kernel, iterations=1)

        erosion = cv2.erode(dilation, kernel, iterations=1)

        filtered = cv2.GaussianBlur(erosion, (5, 5), 0)

        ret, thresh = cv2.threshold(filtered, 127, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        contour = max(contours, key=lambda x: cv2.contourArea(x))

        x, y, w, h = cv2.boundingRect(contour)

        return image[y:y+h, x:x+w]







def get_transforms(phase, width=1600, height=256):

    list_transforms = []

    if phase == "train":

        list_transforms.extend(

            [

                albu.HorizontalFlip(),

                albu.OneOf([

                    albu.RandomContrast(),

                    albu.RandomGamma(),

                    albu.RandomBrightness(),

                    ], p=0.3),

                albu.OneOf([

                    albu.ElasticTransform(alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03),

                    albu.GridDistortion(),

                    albu.OpticalDistortion(distort_limit=2, shift_limit=0.5),

                    ], p=0.3),

                albu.ShiftScaleRotate(),

            ]

        )

    list_transforms.extend(

        [

            albu.Resize(width,height,always_apply=True),

            albu.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), p=1),

            ToTensor(),

        ]

    )

    list_trfms = albu.Compose(list_transforms)

    return list_trfms

class vinDataset(Dataset):

    def __init__(self, root_dir, df, phase):

        super(vinDataset, self).__init__()

        self.root_dir = root_dir

        self.df = df

        self.transforms = get_transforms(phase, width = 300, height = 400)

        self.detect = Detection()

    

    def __read_file__(self, file_name):

        df = pd.read_csv(file_name)

        return df



    def __getitem__(self, idx):

        image_id = self.df.loc[idx, 'imageName']

        image_path = os.path.join(self.root_dir, image_id)

        try:

#             img = cv2.imread(image_path)

            img = self.detect.process(image_path)

            augmented = self.transforms(image=img)

            img = augmented['image']

        except:

            print('Error')

            return self[idx+1]

        label = torch.tensor(0, dtype=torch.long)

        if self.df.loc[idx, 'gender']=='male':

            label = torch.tensor(1, dtype=torch.long) 

        return img, label



    def __len__(self):

        return len(self.df)
batch_size = 16

num_worker = 2

learning_rate = 0.001
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

train = pd.read_csv('/kaggle/input/hand-gender/data/train.csv')

train, valid = train_test_split(train ,stratify=train['gender'], test_size=0.2)

train = train.reset_index(drop=True)

valid = valid.reset_index(drop=True)

test = pd.read_csv('/kaggle/input/hand-gender/data/test.csv')

train_dataset = vinDataset(root_dir='/kaggle/input/hand-gender/data/train/', df=train, phase='train')

valid_dataset = vinDataset(root_dir='/kaggle/input/hand-gender/data/train/', df=valid, phase='valid')



test_dataset = vinDataset(root_dir='/kaggle/input/hand-gender/data/test/', df=test, phase='valid')



train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_worker)

valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=num_worker)

test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_worker)

!pip install efficientnet_pytorch

# model = models.resnet34(pretrained=False, num_classes=2)

from efficientnet_pytorch import EfficientNet

model = EfficientNet.from_pretrained('efficientnet-b1', num_classes=2)

model = model.to(device)

optimizer = optim.Adam(model.parameters(), lr=learning_rate, betas=(0.5, 0.999))

criterion = nn.CrossEntropyLoss()
def train_one_epoch(epoch):

    total_loss = 0

    model.train()

    for idx, (images, labels) in enumerate(tqdm(train_loader)):

        images = images.to(device)

        labels = labels.to(device)

        output = model(images)

        loss = criterion(output, labels)

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        total_loss+=loss.item()

    return total_loss/len(train_loader)



def evaluation(data_loader):

    model.eval()

    with torch.no_grad():

        correct = 0

        total = 0

        for images, labels in tqdm(data_loader):

            images = images.to(device)

            labels = labels.to(device)

            outputs = model(images)

            _, predicted = torch.max(outputs.data, 1)

            total += labels.size(0)

            correct += (predicted == labels).sum().item()



        return correct*100/total

num_epoch = 10

best_acc = 0

for epoch in range(num_epoch):

    train_loss = train_one_epoch(epoch)

    print('{} Epoch\tLoss: {}'.format(epoch, train_loss))

    

    acc_val = evaluation(valid_loader)

    print("[VAL] Accuracy: {}".format(acc_val))

    acc_test = evaluation(test_loader)

    print("[TEST] Accuracy: {} ".format(acc_test))

    if acc_test>best_acc:

        best_acc = acc_test

print('Best Accuracy: ', best_acc)