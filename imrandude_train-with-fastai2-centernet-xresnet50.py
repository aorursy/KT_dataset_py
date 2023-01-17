#Install FastAI2 Library

!pip install git+https://github.com/fastai/fastai2
#Usual Imports

import numpy as np

import pandas as pd

import cv2

from tqdm import tqdm

from functools import reduce, partial



import os

from sklearn.model_selection import train_test_split

from scipy.optimize import minimize



import torch

import torch.nn as nn

import torch.nn.functional as F

import torch.optim as optim

from torch.optim import lr_scheduler

from torch.utils.data import Dataset, DataLoader

from torch.utils import model_zoo

import random

import gc

import os.path



from fastai2.basics import *

from fastai2.callback.all import *

from fastai2.vision.all import *
#List of variables used in the code



#Random seed setting

torch.manual_seed(42)

random.seed(42) 



# Indicator to set all dataset is to be loaded

full=0 



#Limit Training Datasize to 400

tr_limit = 400



#To track the loss

best_loss = torch.FloatTensor()



n_epochs = 10

BATCH_SIZE = 4

SWITCH_LOSS_EPOCH = 5

history = pd.DataFrame()

DISTANCE_THRESH_CLEAR = 2



PATH = '/kaggle/input/pku-autonomous-driving/'

IMG_WIDTH = 2048 - 512

IMG_HEIGHT = IMG_WIDTH // 16 * 5

MODEL_SCALE = 8
#Code borrowed from other notebooks, credits in the references section

train = pd.read_csv(PATH + 'train.csv')

test = pd.read_csv(PATH + 'sample_submission.csv')



camera_matrix = np.array([[2304.5479, 0,  1686.2379],

                          [0, 2305.8757, 1354.9849],

                          [0, 0, 1]], dtype=np.float32)

camera_matrix_inv = np.linalg.inv(camera_matrix)





def str2coords(s, names=['id', 'yaw', 'pitch', 'roll', 'x', 'y', 'z']):

    '''

    Input:

        s: PredictionString (e.g. from train dataframe)

        names: array of what to extract from the string

    Output:

        list of dicts with keys from `names`

    '''

    coords = []

    for l in np.array(s.split()).reshape([-1, 7]):

        coords.append(dict(zip(names, l.astype('float'))))

        if 'id' in coords[-1]:

            coords[-1]['id'] = int(coords[-1]['id'])

    return coords





#############



def imread(path, fast_mode=False):

    img = cv2.imread(path)

    if not fast_mode and img is not None and len(img.shape) == 3:

        img = np.array(img[:, :, ::-1])

    return img

#############



#############





#############

def rotate(x, angle):

    x = x + angle

    x = x - (x + np.pi) // (2 * np.pi) * 2 * np.pi

    return x



#############





def get_img_coords(s):

    '''

    Input is a PredictionString (e.g. from train dataframe)

    Output is two arrays:

        xs: x coordinates in the image

        ys: y coordinates in the image

    '''

    coords = str2coords(s)

    xs = [c['x'] for c in coords]

    ys = [c['y'] for c in coords]

    zs = [c['z'] for c in coords]

    P = np.array(list(zip(xs, ys, zs))).T

    img_p = np.dot(camera_matrix, P).T

    img_p[:, 0] /= img_p[:, 2]

    img_p[:, 1] /= img_p[:, 2]

    img_xs = img_p[:, 0]

    img_ys = img_p[:, 1]

    img_zs = img_p[:, 2] # z = Distance from the camera

    return img_xs, img_ys





#############

from math import sin, cos



# convert euler angle to rotation matrix

def euler_to_Rot(yaw, pitch, roll):

    Y = np.array([[cos(yaw), 0, sin(yaw)],

                  [0, 1, 0],

                  [-sin(yaw), 0, cos(yaw)]])

    P = np.array([[1, 0, 0],

                  [0, cos(pitch), -sin(pitch)],

                  [0, sin(pitch), cos(pitch)]])

    R = np.array([[cos(roll), -sin(roll), 0],

                  [sin(roll), cos(roll), 0],

                  [0, 0, 1]])

    return np.dot(Y, np.dot(P, R))

#############

def draw_line(image, points):

    color = (255, 0, 0)

    cv2.line(image, tuple(points[0][:2]), tuple(points[3][:2]), color, 16)

    cv2.line(image, tuple(points[0][:2]), tuple(points[1][:2]), color, 16)

    cv2.line(image, tuple(points[1][:2]), tuple(points[2][:2]), color, 16)

    cv2.line(image, tuple(points[2][:2]), tuple(points[3][:2]), color, 16)

    return image





def draw_points(image, points):

    for (p_x, p_y, p_z) in points:

        cv2.circle(image, (p_x, p_y), int(1000 / p_z), (0, 255, 0), -1)

    return image

#############

def visualize(img, coords):

    # You will also need functions from the previous cells

    x_l = 1.02

    y_l = 0.80

    z_l = 2.31

    

    img = img.copy()

    for point in coords:

        # Get values

        x, y, z = point['x'], point['y'], point['z']

        yaw, pitch, roll = -point['pitch'], -point['yaw'], -point['roll']

        # Math

        Rt = np.eye(4)

        t = np.array([x, y, z])

        Rt[:3, 3] = t

        Rt[:3, :3] = euler_to_Rot(yaw, pitch, roll).T

        Rt = Rt[:3, :]

        P = np.array([[x_l, -y_l, -z_l, 1],

                      [x_l, -y_l, z_l, 1],

                      [-x_l, -y_l, z_l, 1],

                      [-x_l, -y_l, -z_l, 1],

                      [0, 0, 0, 1]]).T

        img_cor_points = np.dot(camera_matrix, np.dot(Rt, P))

        img_cor_points = img_cor_points.T

        img_cor_points[:, 0] /= img_cor_points[:, 2]

        img_cor_points[:, 1] /= img_cor_points[:, 2]

        img_cor_points = img_cor_points.astype(int)

        # Drawing

        img = draw_line(img, img_cor_points)

        img = draw_points(img, img_cor_points[-1:])

    

    return img





def _regr_preprocess(regr_dict, flip=False):

    if flip:

        for k in ['x', 'pitch', 'roll']:

            regr_dict[k] = -regr_dict[k]

    for name in ['x', 'y', 'z']:

        regr_dict[name] = regr_dict[name] / 100

    regr_dict['roll'] = rotate(regr_dict['roll'], np.pi)

    regr_dict['pitch_sin'] = sin(regr_dict['pitch'])

    regr_dict['pitch_cos'] = cos(regr_dict['pitch'])

    regr_dict.pop('pitch')

    regr_dict.pop('id')

    return regr_dict



def _regr_back(regr_dict):

    for name in ['x', 'y', 'z']:

        regr_dict[name] = regr_dict[name] * 100

    regr_dict['roll'] = rotate(regr_dict['roll'], -np.pi)

    

    pitch_sin = regr_dict['pitch_sin'] / np.sqrt(regr_dict['pitch_sin']**2 + regr_dict['pitch_cos']**2)

    pitch_cos = regr_dict['pitch_cos'] / np.sqrt(regr_dict['pitch_sin']**2 + regr_dict['pitch_cos']**2)

    regr_dict['pitch'] = np.arccos(pitch_cos) * np.sign(pitch_sin)

    return regr_dict



def preprocess_image(img, flip=False):

    img = img[img.shape[0] // 2:]

    bg = np.ones_like(img) * img.mean(1, keepdims=True).astype(img.dtype)

    bg = bg[:, :img.shape[1] // 6]

    img = np.concatenate([bg, img, bg], 1)

    img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))

    if flip:

        img = img[:,::-1]

    return (img / 255).astype('float32')



def get_mask_and_regr(img, labels, flip=False):

    mask = np.zeros([IMG_HEIGHT // MODEL_SCALE, IMG_WIDTH // MODEL_SCALE], dtype='float32')

    regr_names = ['x', 'y', 'z', 'yaw', 'pitch', 'roll']

    regr = np.zeros([IMG_HEIGHT // MODEL_SCALE, IMG_WIDTH // MODEL_SCALE, 7], dtype='float32')

    coords = str2coords(labels)

    xs, ys = get_img_coords(labels)

    for x, y, regr_dict in zip(xs, ys, coords):

        x, y = y, x

        x = (x - img.shape[0] // 2) * IMG_HEIGHT / (img.shape[0] // 2) / MODEL_SCALE

        x = np.round(x).astype('int')

        y = (y + img.shape[1] // 6) * IMG_WIDTH / (img.shape[1] * 4/3) / MODEL_SCALE

        y = np.round(y).astype('int')

        if x >= 0 and x < IMG_HEIGHT // MODEL_SCALE and y >= 0 and y < IMG_WIDTH // MODEL_SCALE:

            mask[x, y] = 1

            regr_dict = _regr_preprocess(regr_dict, flip)

            regr[x, y] = [regr_dict[n] for n in sorted(regr_dict)]

    if flip:

        mask = np.array(mask[:,::-1])

        regr = np.array(regr[:,::-1])

    return mask, regr



def imread(path, fast_mode=False):

    img = cv2.imread(path)

    if not fast_mode and img is not None and len(img.shape) == 3:

        img = np.array(img[:, :, ::-1])

    return img



img = imread(PATH + 'train_images/ID_8a6e65317' + '.jpg')

IMG_SHAPE = img.shape



def convert_3d_to_2d(x, y, z, fx = 2304.5479, fy = 2305.8757, cx = 1686.2379, cy = 1354.9849):

    # stolen from https://www.kaggle.com/theshockwaverider/eda-visualization-baseline

    return x * fx / z + cx, y * fy / z + cy



def optimize_xy(r, c, x0, y0, z0):

    def distance_fn(xyz):

        x, y, z = xyz

        x, y = convert_3d_to_2d(x, y, z0)

        y, x = x, y

        x = (x - IMG_SHAPE[0] // 2) * IMG_HEIGHT / (IMG_SHAPE[0] // 2) / MODEL_SCALE

        x = np.round(x).astype('int')

        y = (y + IMG_SHAPE[1] // 6) * IMG_WIDTH / (IMG_SHAPE[1] *4/3) / MODEL_SCALE

        y = np.round(y).astype('int')

        return (x-r)**2 + (y-c)**2

    

    res = minimize(distance_fn, [x0, y0, z0], method='Powell')

    x_new, y_new, z_new = res.x

    return x_new, y_new, z0



def clear_duplicates(coords):

    for c1 in coords:

        xyz1 = np.array([c1['x'], c1['y'], c1['z']])

        for c2 in coords:

            xyz2 = np.array([c2['x'], c2['y'], c2['z']])

            distance = np.sqrt(((xyz1 - xyz2)**2).sum())

            if distance < DISTANCE_THRESH_CLEAR:

                if c1['confidence'] < c2['confidence']:

                    c1['confidence'] = -1

    return [c for c in coords if c['confidence'] > 0]



def extract_coords(prediction):

    logits = prediction[0]

    regr_output = prediction[1:]

    points = np.argwhere(logits > 0)

    col_names = sorted(['x', 'y', 'z', 'yaw', 'pitch_sin', 'pitch_cos', 'roll'])

    coords = []

    for r, c in points:

        regr_dict = dict(zip(col_names, regr_output[:, r, c]))

        coords.append(_regr_back(regr_dict))

        coords[-1]['confidence'] = 1 / (1 + np.exp(-logits[r, c]))

        coords[-1]['x'], coords[-1]['y'], coords[-1]['z'] = optimize_xy(r, c, coords[-1]['x'], coords[-1]['y'], coords[-1]['z'])

    coords = clear_duplicates(coords)

    return coords



def coords2str(coords, names=['yaw', 'pitch', 'roll', 'x', 'y', 'z', 'confidence']):

    s = []

    for c in coords:

        for n in names:

            s.append(str(c.get(n, 0)))

    return ' '.join(s)

class CarDataset(Dataset):

    """Car dataset."""



    def __init__(self, dataframe, root_dir, training=True, transform=None):

        self.df = dataframe

        self.root_dir = root_dir

        self.transform = transform

        self.training = training



    def __len__(self):

        return len(self.df)



    def __getitem__(self, idx):

        if torch.is_tensor(idx):

            idx = idx.tolist()

        

        # Get image name

        idx, labels = self.df.values[idx]

        img_name = self.root_dir.format(idx)

        

        # Augmentation

        flip = False

        if self.training:

            flip = np.random.randint(2) == 1

        

        # Read image

        img0 = imread(img_name, True)

        img = preprocess_image(img0, flip=flip)

        img = np.rollaxis(img, 2, 0)

        

        # Get mask and regression maps

        mask, regr = get_mask_and_regr(img0, labels, flip=False)

        regr = np.rollaxis(regr, 2, 0)

        

        #Modify Mask to incorprorate regr 

        mask = np.concatenate((np.expand_dims(mask, axis=0), regr), axis=0)

        

        return [img, mask]

#Test/Train Image directories

train_images_dir = PATH + 'train_images/{}.jpg'

test_images_dir = PATH + 'test_images/{}.jpg'



#Dataframe for train, valid and test

df_train, df_dev = train_test_split(train, test_size=0.01, random_state=42)

df_test = test
# Create dataset objects

train_dataset = CarDataset(df_train, train_images_dir, training=True)

dev_dataset = CarDataset(df_dev, train_images_dir, training=False)



Stop_tr = len(train_dataset)

Random_tr_ListOfIntegers = [random.randrange(0, Stop_tr) for iter in range(tr_limit)]



#Limit Train dataset to 400 images

if not full:

    train_dataset = torch.utils.data.Subset(train_dataset, Random_tr_ListOfIntegers)



# Create data generators - they will produce batches

train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE//2, shuffle=True, num_workers=0)

dev_loader = DataLoader(dataset=dev_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

class double_conv(nn.Module):

    '''(conv => BN => ReLU) * 2'''

    def __init__(self, in_ch, out_ch):

        super(double_conv, self).__init__()

        self.conv = nn.Sequential(

            nn.Conv2d(in_ch, out_ch, 3, padding=1),

            nn.BatchNorm2d(out_ch),

            nn.ReLU(inplace=True),

            nn.Conv2d(out_ch, out_ch, 3, padding=1),

            nn.BatchNorm2d(out_ch),

            nn.ReLU(inplace=True)

        )



    def forward(self, x):

        x = self.conv(x)

        return x



class up(nn.Module):

    def __init__(self, in_ch, out_ch, bilinear=True):

        super(up, self).__init__()



        #  would be a nice idea if the upsampling could be learned too,

        #  but my machine do not have enough memory to handle all those weights

        if bilinear:

            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        else:

            self.up = nn.ConvTranspose2d(in_ch//2, in_ch//2, 2, stride=2)



        self.conv = double_conv(in_ch, out_ch)



    def forward(self, x1, x2=None):

        x1 = self.up(x1)

        

        # input is CHW

        diffY = x2.size()[2] - x1.size()[2]

        diffX = x2.size()[3] - x1.size()[3]



        x1 = F.pad(x1, (diffX // 2, diffX - diffX//2,

                        diffY // 2, diffY - diffY//2))

        

        # for padding issues, see 

        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a

        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd

        

        if x2 is not None:

            x = torch.cat([x2, x1], dim=1)

        else:

            x = x1

        x = self.conv(x)

        return x



def get_mesh(batch_size, shape_x, shape_y):

    mg_x, mg_y = np.meshgrid(np.linspace(0, 1, shape_y), np.linspace(0, 1, shape_x))

    mg_x = np.tile(mg_x[None, None, :, :], [batch_size, 1, 1, 1]).astype('float32')

    mg_y = np.tile(mg_y[None, None, :, :], [batch_size, 1, 1, 1]).astype('float32')

    mesh = torch.cat([torch.tensor(mg_x).to(device), torch.tensor(mg_y).to(device)], 1)

    return mesh

model=xresnet50()

model = nn.Sequential(*list(model.children())[:-4])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class CentResnet1(nn.Module):

    '''Mixture of previous classes'''

    def __init__(self, n_classes):

        super(CentResnet1, self).__init__()

        self.base_model = model

        

        # Lateral layers convert resnet outputs to a common feature size

        self.lat8 = nn.Conv2d(512, 256, 1)

        self.lat16 = nn.Conv2d(1024, 256, 1)

        self.lat32 = nn.Conv2d(2048, 256, 1)

        self.bn8 = nn.GroupNorm(16, 256)

        self.bn16 = nn.GroupNorm(16, 256)

        self.bn32 = nn.GroupNorm(16, 256)



       

        self.conv0 = double_conv(5, 64)

        self.conv1 = double_conv(64, 128)

        self.conv2 = double_conv(128, 512)

        self.conv3 = double_conv(512, 1024)

        

        self.mp = nn.MaxPool2d(2)

        

        self.up1 = up(1282 , 512) #+ 1024

        self.up2 = up(512 + 512, 256)

        self.outc = nn.Conv2d(256, n_classes, 1)

        

    

    def forward(self, x):

        batch_size = x.shape[0]

        mesh1 = get_mesh(batch_size, x.shape[2], x.shape[3])

        x0 = torch.cat([x, mesh1], 1)

        x1 = self.mp(self.conv0(x0))

        x2 = self.mp(self.conv1(x1))

        x3 = self.mp(self.conv2(x2))

        x4 = self.mp(self.conv3(x3))

        

                # Run frontend network

        feats32 = self.base_model(x)

        lat32 = F.relu(self.bn32(self.lat32(feats32)))

        

        # Add positional info

        mesh2 = get_mesh(batch_size, lat32.shape[2], lat32.shape[3])

        feats = torch.cat([lat32, mesh2], 1)

        x = self.up1(feats, x4)

        x = self.up2(x, x3)

        x = self.outc(x)

        return x

def mod_criterion(prediction, mask, weight=0.4, size_average=True):

    #Extract regr and mask values from the combined mask

    regr = mask[0,1:].unsqueeze(0)

    mask = mask[0,0:1]

    # Binary mask loss

    pred_mask = torch.sigmoid(prediction[:, 0])

#     mask_loss = mask * (1 - pred_mask)**2 * torch.log(pred_mask + 1e-12) + (1 - mask) * pred_mask**2 * torch.log(1 - pred_mask + 1e-12)

    mask_loss = mask * torch.log(pred_mask + 1e-12) + (1 - mask) * torch.log(1 - pred_mask + 1e-12)

    mask_loss = -mask_loss.mean(0).sum()

    

    # Regression L1 loss

    pred_regr = prediction[:, 1:]

    regr_loss = (torch.abs(pred_regr - regr).sum(1) * mask).sum(1).sum(1) / mask.sum(1).sum(1)

    regr_loss = regr_loss.mean(0)

  

    # Sum

    loss = weight*mask_loss +(1-weight)* regr_loss

    if not size_average:

        loss *= prediction.shape[0]

    return loss
from torchvision import datasets, transforms

from torch.optim.lr_scheduler import StepLR

from fastai2.optimizer import Adam



data = DataBunch(train_loader, dev_loader).cuda()

learn = Learner(data, CentResnet1(8), loss_func=mod_criterion, opt_func=Adam)
learn.lr_find()
learn.fit_one_cycle(n_epochs, 1e-2)
learn.recorder.plot_loss()
learn.save("Centernet-Xresnet50-ph1")