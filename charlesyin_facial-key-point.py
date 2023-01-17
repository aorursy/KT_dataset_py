# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
# Prepare data

# Read the data from csv and save them to npy



train_dir = '../input/training/training.csv'

data = pd.read_csv(train_dir)



num_images = data.shape[0]

images = np.zeros((num_images, 96, 96))

landmarks = np.zeros((num_images, data.shape[1] - 1))



for i in range(num_images):

    img = data['Image'][i].split(' ')

    img = np.array(img, dtype='float32').reshape(96, 96)

    images[i, :, :] = img



    ldmk = np.array(data.iloc[i, 0:-1], dtype='float32')

    landmarks[i, :] = ldmk



np.save('train_data.npy', images)

np.save('train_ldmk.npy', landmarks)



# Read and save test data



data = pd.read_csv('../input/test/test.csv')

num_images = data.shape[0]

images = np.zeros((num_images, 96, 96))



for i in range(num_images):

    img = data['Image'][i].split(' ')

    img = np.array(img, dtype='float32').reshape(96, 96)

    images[i, :, :] = img



np.save('test_data.npy', images)

# load data

import numpy as np

from torch.utils.data import Dataset

import torch.utils.data as Data

import matplotlib.pyplot as plt

from torchvision import transforms

from skimage import transform

import torch

import cv2





class Rescale(object):

    def __init__(self, output_size):

        assert isinstance(output_size, (int, tuple))

        self.output_size = output_size



    def __call__(self, sample):

        image, landmarks = sample['image'], sample['landmarks']

        h, w = image.shape[:2]

        if isinstance(self.output_size, int):

            if h > w:

                new_h, new_w = self.output_size * h / w, self.output_size

            else:

                new_h, new_w = self.output_size, self.output_size * w / h

        else:

            new_h, new_w = self.output_size



        new_h, new_w = int(new_h), int(new_w)

        img = transform.resize(image, (new_h, new_w))

        landmarks = landmarks * [new_w * 1.0 / w, new_h * 1.0 / h]



        return {'image': img, 'landmarks': landmarks}





class RandomCrop(object):

    def __init__(self, output_size):

        assert isinstance(output_size, (int, tuple))

        if isinstance(output_size, int):

            self.output_size = (output_size, output_size)

        else:

            assert len(output_size) == 2

            self.output_size = output_size



    def __call__(self, sample):

        image, landmarks = sample['image'], sample['landmarks']

        h, w = image.shape[:2]

        new_h, new_w = self.output_size

        if h > new_h:

            top = np.random.randint(0, h - new_h)

        else:

            top = 0

        if w > new_w:

            left = np.random.randint(0, w - new_w)

        else:

            left = 0

        image = image[top:top + new_h,

                left:left + new_w]

        landmarks = landmarks - [left, top]

        return {'image': image, 'landmarks': landmarks}





class RandomFlip(object):

    def __init__(self, p=0.5):

        self.p = 0.5



    def __call__(self, sample):

        image, landmarks = sample['image'], sample['landmarks']

        p = np.random.rand()

        if p > self.p:

            image = np.fliplr(image)

            landmarks = landmarks * [-1, 1] + [96, 0]



        return {'image': image, 'landmarks': landmarks}





def get_rotation_mat(img, angle_degree):

    height, width = img.shape[0:2]

    center = (width / 2.0, height / 2.0)

    angle_radians = np.radians(angle_degree)

    rot_mat = cv2.getRotationMatrix2D(center, angle_degree, scale=1)

    new_height = height * np.abs(np.cos(angle_radians)) + width * np.abs(np.sin(angle_radians))

    new_width = height * np.abs(np.sin(angle_radians)) + width * np.abs(np.cos(angle_radians))

    new_center = (new_width / 2.0, new_height / 2.0)

    dx, dy = (new_center[0] - center[0], new_center[1] - center[1])

    rot_mat[0, 2] += dx

    rot_mat[1, 2] += dy

    # img = cv2.warpAffine(img, rot_mat,(int(new_width), int(new_height)))

    return rot_mat, (int(new_width), int(new_height))





def show_sample(sample):

    img = sample['image']

    lm = sample['landmarks']

    plt.imshow(img, cmap='gray')

    plt.scatter(lm[:, 0], lm[:, 1], s=10, c='r')

    plt.show()





def test_rotation(img, lm, angle_degree):

    rot_mat, new_size = get_rotation_mat(img, angle_degree)

    new_img = cv2.warpAffine(img, rot_mat, new_size)

    lm = np.hstack((lm, np.ones((lm.shape[0], 1))))

    new_lm = np.dot(lm, rot_mat.T)

    plt.imshow(new_img, cmap='gray')

    plt.scatter(new_lm[:, 0], new_lm[:, 1], s=10, c='r')

    plt.show()





class RandomRotation(object):

    def __init__(self, angle_degree=10):

        self.angle = angle_degree



    def __call__(self, sample):

        image, landmarks = sample['image'], sample['landmarks']

        angle = np.random.uniform(-self.angle, self.angle, 1)

        rot_mat, new_size = get_rotation_mat(image, angle)



        image = cv2.warpAffine(image, rot_mat, new_size)

        landmarks = np.hstack((landmarks, np.ones((landmarks.shape[0], 1))))

        landmarks = np.dot(landmarks, rot_mat.T)



        return {'image': image, 'landmarks': landmarks}





class Normalize(object):

    def __call__(self, sample):

        image, landmarks = sample['image'], sample['landmarks']

        image = (image / 255.0 - 0.5) / 0.5

        image = image[np.newaxis, ...]

        landmarks = (landmarks / 96.0 - 0.5) / 0.5

        return {'image': image, 'landmarks': landmarks}





class ToTensor(object):

    def __call__(self, sample):

        image, landmarks = sample['image'], sample['landmarks']

        return {'image': torch.Tensor(image), 'landmarks': torch.Tensor(landmarks)}





class FaceDataset(Dataset):

    def __init__(self, pth_img, transform=None):

        self.images = np.load(pth_img)

        self.transform = transform

        self.num_imgs = len(self.images)



    def __len__(self):

        return self.num_imgs



    def __getitem__(self, idx):

        img = self.images[idx]

        lms = np.random.rand(30)

        sample = {'image': img, 'landmarks': lms.reshape(-1, 2)}

        if self.transform:

            sample = self.transform(sample)

        return sample['image']





class FaceLandmarkDataset(Dataset):

    def __init__(self, pth_img, pth_ldmk, transform=None):

        self.images = np.load(pth_img)

        self.transform = transform

        self.landmarks = np.load(pth_ldmk)

        self.num_imgs = len(self.images)



    def __len__(self):

        return self.num_imgs



    def __getitem__(self, idx):

        img = self.images[idx]

        lm = self.landmarks[idx]

        sample = {'image': img, 'landmarks': lm.reshape(-1, 2)}

        if self.transform:

            sample = self.transform(sample)

        return sample





def get_train_loader():

    scale = Rescale(110)

    crop = RandomCrop(110)

    flip = RandomFlip(0.5)

    rot = RandomRotation(15)

    norm = Normalize()

    totensor = ToTensor()

    composed = transforms.Compose([flip, scale, rot, crop, norm, totensor])



    train_data = FaceLandmarkDataset('train_data.npy', 'train_ldmk.npy', transform=composed)

    train_loader = Data.DataLoader(train_data, batch_size=32, shuffle=True, num_workers=4)

    return train_loader



def get_test_loader():

    scale = Rescale(110)

    norm = Normalize()

    totensor = ToTensor()

    composed = transforms.Compose([scale, norm, totensor])

    test_data = FaceDataset('test_data.npy', transform=composed)

    test_loader = Data.DataLoader(test_data, batch_size=32, shuffle=False, num_workers=4)

    return test_loader

# construct models

import torch.nn as nn





class Model(nn.Module):

    def __init__(self):

        super(Model, self).__init__()

        self.conv = nn.Sequential(

            nn.Conv2d(1, 16, 3),

            nn.ReLU(),

            nn.MaxPool2d(2, stride=2),



            nn.Conv2d(16, 32, 3),

            nn.ReLU(),

            nn.MaxPool2d(2, stride=2),



            nn.Conv2d(32, 64, 3),

            nn.ReLU(),

            nn.MaxPool2d(2, stride=2),



            nn.Conv2d(64, 128, 3),

            nn.ReLU(),

            nn.MaxPool2d(2, stride=2),

        )



        self.fc = nn.Sequential(

            nn.Linear(128 * 5 * 5, 512),

            nn.ReLU(),

            nn.Dropout(0.2),



            nn.Linear(512, 512),

            nn.ReLU(),

            nn.Dropout(0.2),



            nn.Linear(512, 30)

        )



    def forward(self, x):

        conv_feat = self.conv(x)

        out = self.fc(conv_feat.view(-1, 128 * 5 * 5))

        return out





train_loader = get_train_loader()

test_loader = get_test_loader()
# test loader

from torchvision.utils import make_grid

import cv2

import torch

import math



def get_grid(batch_img, batch_lm):

    processed = []

    for idx in range(batch_img.shape[0]):

        img = (batch_img[idx, 0].numpy() * 0.5 + 0.5) * 255.0

        img = cv2.cvtColor(img.astype('uint8'), cv2.COLOR_GRAY2BGR)

        lm = (batch_lm[idx].numpy().reshape((-1, 2)) * 0.5 + 0.5) * 96

        for lm_idx in range(lm.shape[0]):

            if not math.isnan(lm[lm_idx][0]):

                cv2.circle(img, (int(lm[lm_idx][0]), int(lm[lm_idx][1])), 3, (0, 255, 0), -1)

        processed.append(torch.from_numpy(img).permute(2, 0, 1))

    processed = torch.stack(processed, dim=0)

    I = make_grid(processed).permute(1, 2, 0)



    return I



sample = iter(train_loader).next()

imgs, lms = sample['image'], sample['landmarks']

I = get_grid(imgs, lms)
import matplotlib.pyplot as plt

plt.imshow(I.numpy().astype('uint8'))
# define loss function

def loss_func(pred, gt):

    batch_sz = pred.shape[0]

    diff = pred - gt.view(batch_sz, -1)

    nan_ind = (diff!=diff)

    diff[nan_ind] = 0

    loss = diff.pow(2).sum() / (diff.numel()-nan_ind.sum())

    return loss
from torch import optim



# initialize the model

model = Model()

model.cuda()

optimizer = optim.Adam(model.parameters(), lr=1e-4)



def train(epoch):

    model.train()

    train_loss = 0

    for batch_idx, samples in enumerate(train_loader):

        imgs, lms = samples['image'], samples['landmarks']

        imgs = imgs.cuda()

        lms = lms.cuda()



        optimizer.zero_grad()

        pred = model(imgs)

        loss = loss_func(pred, lms)

        loss.backward()

        train_loss += loss.item()

        optimizer.step()

        

    train_loss /= len(train_loader)

    if (epoch+1) % 10 == 0:

        print('======> Epoch: {}\t Average Train Loss: {}'.format(epoch, train_loss))

    return train_loss



total_train_loss = []

for epoch in range(150):

    loss_epoch = train(epoch)

    total_train_loss.append(loss_epoch)

    
def test():

    imgs = iter(test_loader).next()

    imgs = imgs.cuda()

    pred = model(imgs)

    I = get_grid(imgs.detach().cpu(), pred.detach().cpu())

    return I

I = test()

plt.imshow(I)
test_data = test_loader.dataset

pred = []

for idx in range(len(test_data)):

    img = test_data[idx].unsqueeze(0)

    pt = model(img.cuda())

    pred.append(pt)

pred = torch.stack(pred, dim=0).detach().squeeze().cpu().numpy()



lookid_data = pd.read_csv('../input/IdLookupTable.csv')

lookid_list = list(lookid_data['FeatureName'])

imageID = list(lookid_data['ImageId'] - 1)

rowid = lookid_data['RowId']

rowid = list(rowid)

feature = []

for f in list(lookid_list):

    feature.append(lookid_list.index(f))



pre_list = list(pred)

preded = []

for x, y in zip(imageID, feature):

    preded.append(pre_list[x][y])



rowid = pd.Series(rowid, name='RowId')

loc = pd.Series(preded, name='Location')

submission = pd.concat([rowid, loc], axis=1)

submission.to_csv('face_key_detection_submission.csv',index=False)