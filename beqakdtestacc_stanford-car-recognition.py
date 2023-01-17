!pip install console-progressbar
!pip install pretrainedmodels
# Preprocessing
from console_progressbar import ProgressBar
import matplotlib.pyplot as plt
import pretrainedmodels
import seaborn as sns
import pandas as pd
import numpy as np
import cv2 as cv2
import scipy.io
import tarfile
import shutil
import random
import os
r = range
train_path = '../input/stanford-cars-dataset/cars_train/cars_train/'
test_path  = '../input/stanford-cars-dataset/cars_test/cars_test/'
img_width, img_height = 224, 224

train_output = '/kaggle/working/data/train/' 
valid_output = '/kaggle/working/data/valid/'
test_output  = '/kaggle/working/data/test/'
print("Lets see how many cars we have!")
print('..............................')
train = scipy.io.loadmat('../input/stanford-devkit/stanford_devkit/devkit/cars_train_annos.mat')
train = np.transpose(train['annotations'])
train_unique = [img[0][4][0][0] for img in train]
print(np.unique(train_unique))
print('')
print('Number of differenct cars is:', np.unique(train_unique).shape[0])
def create_folder(path):
    if not os.path.exists(path):
        os.makedirs(path)
def validate_train(fnames, bboxes, labels, data_str):
    # We have to split train data into two dataset 0.8 for train and 0.2 for valid
    train_num = int(round(len(fnames) * 0.8))
    train_ids = random.sample(r(len(fnames)), train_num)
    pb = ProgressBar(total=100, prefix='Save train data', suffix='', decimals=3, length=50, fill='=')
    for index in r(len(fnames)):
        # First find proper path(valid/train)
        if data_str == 'train':
            label = labels[index]
        fname = fnames[index]
        if data_str == 'train':
            new_path = train_path + fname
        else:
            new_path = test_path + fname
        img = cv2.imread(new_path)
        
        # Create dist path
        if data_str == 'train':
            dst_path = valid_output
            if index in train_ids:
                dst_path = train_output

            dst_path = dst_path + label + '/'
            create_folder(dst_path)
            dst_path += fname
        else:
            dst_path = test_output + fname
        # Take bboxes and crop the image
        a1, b1, a2, b2 = bboxes[index]
        a1, b1 = max(0, a1 - 16), max(0, b1 - 16)
        a2, b2 = min(a2 + 16, img.shape[1]), min(b2 + 16, img.shape[0])
        
        # Crop image and save it to the path
        img = cv2.resize(img[b1:b2, a1:a2], (img_width, img_height))
        
        cv2.imwrite(dst_path, img)
        
        # Print progress
        pb.print_progress_bar((index + 1) * 100 / len(fnames))
def vaidate_data(data_str):
    data = None
    bbox_id, fname_id, label_id = r(4), 5, 4
    fnames, bboxes, labels = [], [], None
    if data_str == 'train':
        labels = []
        data = scipy.io.loadmat('../input/devkit-test/devkit/cars_train_annos.mat')
    else:
        # We have no labels for test data.
        data = scipy.io.loadmat('../input/stanford-devkit/stanford_devkit/devkit/cars_test_annos.mat')
    data = np.transpose(data['annotations'])
    
    for image in data:
        # bounding box coordinates!
        bbox_arr = [] 
        for _ in bbox_id:
            bbox_arr.append(image[0][_][0][0])
        bboxes.append(bbox_arr)
        
        # Fnames
        if data_str =='train':
            fnames.append(image[0][fname_id][0])
        else:
            fnames.append(image[0][4][0])
        # Labels
        if data_str == 'train':
            labels.append('%04d' % (image[0][label_id][0][0]))
    if data_str == 'train':
        validate_train(fnames, bboxes, labels, data_str)
    else:
        validate_train(fnames, bboxes, [], data_str)
        
# create_folder('/kaggle/working/data/train')
# create_folder('/kaggle/working/data/valid')
create_folder('/kaggle/working/data/test')
# vaidate_data('train')
vaidate_data('test')
from torch import nn
import torch.nn.functional as F
# Model imports
from fastai.metrics import error_rate
from fastai.vision import *
from fastai import *
import torchvision
# This loss is taken from fastai forum page
class my_loss(nn.Module):
    def __init__(self, alpha=1., gamma=2.):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets, **kwargs):
        CE_loss = nn.CrossEntropyLoss(reduction='none')(inputs, targets)
        pt = torch.exp(-CE_loss)
        F_loss = self.alpha * ((1-pt)**self.gamma) * CE_loss
        return F_loss.mean()
# This resnext preprocessing is taken from fastai forum page
def se_resnext50_32x4d(pretrained=False):
    pretrained = 'imagenet' if pretrained else None
    model = pretrainedmodels.se_resnext50_32x4d(pretrained=pretrained)
    return nn.Sequential(*list(model.children()))
test = scipy.io.loadmat('../input/stanford-devkit/stanford_devkit/devkit/cars_test_annos_withlabels.mat')
temp_arr = []
for _ in r(test['annotations'].shape[1]):
    fname = test['annotations']['fname']
    fname = np.array(fname)
    fname = np.transpose(fname)[_][0][0]
    temp_arr.append(fname)
    
test_data = df=pd.DataFrame(data=np.transpose(np.array(test['annotations']['class'],dtype=np.int)),
                  index=temp_arr)
test_data.to_csv('/kaggle/working/data/test_data.csv')
from fastai.imports import *
learn = load_learner('../input/teeeeeeeest/','fastai_model_vol2.0.pkl', test=ImageList.from_csv('/kaggle/working/data','test_data.csv',folder='test'))
preds,y = learn.TTA(ds_type=DatasetType.Test)
a = preds
b = np.array(df[0] - 1,dtype=np.int) 
b = torch.from_numpy(b)
acc=accuracy(a,b)

acc
cars_meta = scipy.io.loadmat('../input/stanford-devkit/stanford_devkit/devkit/cars_meta.mat')
class_names = cars_meta['class_names']  # shape=(1, 196)
class_names = np.transpose(class_names)
print('class_names.shape: ' + str(class_names.shape))
path = '../input/stanford-cars-dataset/cars_test/cars_test/00002.jpg'
im = open_image(path)
im
pred_class,pred_idx,outputs = learn.predict(im)
print('Predicted class is:', int(pred_class))
print('Sample class_name: [{}]'.format(class_names[int(pred_class) + 1][0]))

print('')
real_num = np.transpose(np.array(test['annotations']['class']))[int(path.split('/')[-1].split('.')[0]) - 1][0][0][0] - 1
print('Actuall class is:', real_num)
print('Sample class_name: [{}]'.format(class_names[real_num + 1][0]))
path = '../input/stanford-cars-dataset/cars_test/cars_test/00100.jpg'
im = open_image(path)
im
pred_class,pred_idx,outputs = learn.predict(im)
print('Predicted class is:', int(pred_class))
print('Sample class_name: [{}]'.format(class_names[int(pred_class) + 1][0]))

print('')
real_num = np.transpose(np.array(test['annotations']['class']))[int(path.split('/')[-1].split('.')[0]) - 1][0][0][0] - 1
print('Actuall class is:', real_num)
print('Sample class_name: [{}]'.format(class_names[real_num + 1][0]))
path = '../input/stanford-cars-dataset/cars_test/cars_test/01000.jpg'
im = open_image(path)
im
pred_class,pred_idx,outputs = learn.predict(im)
print('Predicted class is:', int(pred_class))
print('Sample class_name: [{}]'.format(class_names[int(pred_class) + 1][0]))

print('')
real_num = np.transpose(np.array(test['annotations']['class']))[int(path.split('/')[-1].split('.')[0]) - 1][0][0][0] - 1
print('Actuall class is:', real_num)
print('Sample class_name: [{}]'.format(class_names[real_num + 1][0]))
cars_meta = scipy.io.loadmat('../input/stanford-devkit/stanford_devkit/devkit/cars_meta.mat')
class_names = cars_meta['class_names']  # shape=(1, 196)
class_names = np.transpose(class_names)

temp = pd.DataFrame(class_names)

def mark(val):
    return val[0].split()[0]

temp[0] = temp[0].apply(mark)

my_map = {}
for _ in range(temp[0].shape[0]):
    my_map[_] = temp[0][_]
my_map_id = {}
count = 0
for _ in temp[0].unique():
    my_map_id[_] = count
    count += 1
my_map_id
my_map
# Now we take our predictions and labels and see how many correct model we have
correct_ones = 0
for index in r(a.shape[0]):
    pred_class = int(a[index].argmax())
    true_class = int(b[index])
    
    pred_model = my_map_id[my_map[pred_class]]
    true_model = my_map_id[my_map[true_class]]
    if true_model == pred_model:
        correct_ones += 1
print("acc on only model is:", correct_ones*100/a.shape[0])
path = '../input/stanford-cars-dataset/cars_test/cars_test/01002.jpg'
im = open_image(path)
im
pred_class,pred_idx,outputs = learn.predict(im)
print('Predicted class is:', my_map_id[my_map[int(pred_class)]])
print('Sample class_name: [{}]'.format(class_names[int(pred_class) + 1][0]))

print('')
real_num = np.transpose(np.array(test['annotations']['class']))[int(path.split('/')[-1].split('.')[0]) - 1][0][0][0] - 1
print('Actuall class is:', my_map_id[my_map[real_num]])
print('Sample class_name: [{}]'.format(class_names[real_num + 1][0]))
path = '../input/stanford-cars-dataset/cars_test/cars_test/01001.jpg'
im = open_image(path)
im
pred_class,pred_idx,outputs = learn.predict(im)
print('Predicted class is:', my_map_id[my_map[int(pred_class)]])
print('Sample class_name: [{}]'.format(class_names[int(pred_class) + 1][0]))

print('')
real_num = np.transpose(np.array(test['annotations']['class']))[int(path.split('/')[-1].split('.')[0]) - 1][0][0][0] - 1
print('Actuall class is:', my_map_id[my_map[real_num]])
print('Sample class_name: [{}]'.format(class_names[real_num + 1][0]))