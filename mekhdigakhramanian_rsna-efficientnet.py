!pip install ../input/kerasapplications/keras-team-keras-applications-3b180cb -f ./ --no-index
!pip install ../input/efficientnet/efficientnet-1.1.0/ -f ./ --no-index
import os
import cv2
import pydicom
import pandas as pd
import numpy as np 
import tensorflow as tf 
import matplotlib.pyplot as plt 
import random
from tqdm.notebook import tqdm 
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_absolute_error
from tensorflow_addons.optimizers import RectifiedAdam
from tensorflow.keras import Model
import tensorflow.keras.backend as K
import tensorflow.keras.layers as L
import tensorflow.keras.models as M
from tensorflow.keras.optimizers import Nadam
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image
root = '/kaggle/input/rsna-str-pulmonary-embolism-detection'
for item in os.listdir(root):
    path = os.path.join(root, item)
    if os.path.isfile(path):
        print(path)
print('Reading train data...')
train = pd.read_csv("../input/rsna-str-pulmonary-embolism-detection/train.csv")
print(train.shape)
train.head()
print('Reading test data...')
test = pd.read_csv("../input/rsna-str-pulmonary-embolism-detection/test.csv")
print(test.shape)
test.head()
print('Reading sample data...')
ss = pd.read_csv("../input/rsna-str-pulmonary-embolism-detection/sample_submission.csv")
print(ss.shape)
ss.head()
def convert_to_rgb(array):
    array = array.reshape((512, 512, 1))
    return np.stack([array, array, array], axis=2).reshape((512, 512, 3))
    
def custom_dcom_image_generator(batch_size, dataset, test=False, debug=False):
    
    fnames = dataset[['StudyInstanceUID', 'SeriesInstanceUID', 'SOPInstanceUID']]
    
    if not test:
        Y = dataset[['negative_exam_for_pe', 'rv_lv_ratio_gte_1', 'rv_lv_ratio_lt_1',
                     'leftsided_pe', 'chronic_pe', 'rightsided_pe',
                     'acute_and_chronic_pe', 'central_pe', 'indeterminate']]
        prefix = 'input/rsna-str-pulmonary-embolism-detection/train'
        
    else:
        prefix = 'input/rsna-str-pulmonary-embolism-detection/test'
    
    X = []
    batch = 0
    for st, sr, so in fnames.values:
        if debug:
            print(f"Current file: ../{prefix}/{st}/{sr}/{so}.dcm")

        dicom = get_img(f"../{prefix}/{st}/{sr}/{so}.dcm")
        image = convert_to_rgb(dicom)
        X.append(image)
        
        del st, sr, so
        
        if len(X) == batch_size:
            if test:
                yield np.array(X)
                del X
            else:
                yield np.array(X), Y[batch*batch_size:(batch+1)*batch_size].values
                del X
                
            gc.collect()
            X = []
            batch += 1
        
    if test:
        yield np.array(X)
    else:
        yield np.array(X), Y[batch*batch_size:(batch+1)*batch_size].values
        del Y
    del X
    gc.collect()
    return
def get_img(path):
    d = pydicom.dcmread(path)
    return cv2.resize((d.pixel_array - d.RescaleIntercept) / (d.RescaleSlope * 1000), (512, 512))
import albumentations as Alb

augs = {'Original': None,
              'Blur': Alb.Blur(p=1.0),
             #'MedianBlur': A.MedianBlur(blur_limit=5, p=1.0),
              'GaussianBlur': Alb.GaussianBlur(p=1.0),
             'MotionBlur': Alb.MotionBlur(p=1.0),
        'GridDropout': Alb.GridDropout(p=1.0),
        #'CenterCrop': A.CenterCrop(height=256, width=256, p=1.0),
#        'RandomRotate90': Alb.RandomRotate90(p=1.0),
        # 'ShiftScaleRotate': A.ShiftScaleRotate(p=1.0),
#         'Rotate': Alb.Rotate()
       }

image = get_img(f'../input/rsna-str-pulmonary-embolism-detection/train/0003b3d648eb/d2b2960c2bbf/08ad39846fed.dcm')
print("Real SHape = ",image.shape)
for ite,(key, aug) in enumerate(augs.items()):
    if aug is not None:
        image = aug(image=image)['image']
        print("New Shape = ",image.shape)
        plt.imshow(image)
def convert_to_rgb(array):
    array = array.reshape((512, 512, 1))
    return np.stack([array, array, array], axis=2).reshape((512, 512, 3))
    
def custom_dcom_image_generator(batch_size, dataset, test=False, debug=False):
    
    fnames = dataset[['StudyInstanceUID', 'SeriesInstanceUID', 'SOPInstanceUID']]
    
    if not test:
        Y = dataset[['negative_exam_for_pe', 'rv_lv_ratio_gte_1', 'rv_lv_ratio_lt_1',
                     'leftsided_pe', 'chronic_pe', 'rightsided_pe',
                     'acute_and_chronic_pe', 'central_pe', 'indeterminate']]
        prefix = 'input/rsna-str-pulmonary-embolism-detection/train'
        
    else:
        prefix = 'input/rsna-str-pulmonary-embolism-detection/test'
    
    X = []
    batch = 0
    for st, sr, so in fnames.values:
        if debug:
            print(f"Current file: ../{prefix}/{st}/{sr}/{so}.dcm")

        dicom = get_img(f"../{prefix}/{st}/{sr}/{so}.dcm")
        image = convert_to_rgb(dicom)
        X.append(image)
        
        del st, sr, so
        
        if len(X) == batch_size:
            if test:
                yield np.array(X)
                del X
            else:
                yield np.array(X), Y[batch*batch_size:(batch+1)*batch_size].values
                del X
                
            gc.collect()
            X = []
            batch += 1
        
    if test:
        yield np.array(X)
    else:
        yield np.array(X), Y[batch*batch_size:(batch+1)*batch_size].values
        del Y
    del X
    gc.collect()
    return
Dropout_model = 0.5
from tensorflow.keras.layers import (
    Dense, Dropout, Activation, Flatten, Input, BatchNormalization, GlobalAveragePooling2D, Add, Conv2D, AveragePooling2D, 
    LeakyReLU, Concatenate 
)
import efficientnet.tfkeras as efn

def get_efficientnet(model, shape):
    models_dict = {
        'b0': efn.EfficientNetB0(input_shape=shape,weights=None,include_top=False),
        'b1': efn.EfficientNetB1(input_shape=shape,weights=None,include_top=False),
        'b2': efn.EfficientNetB2(input_shape=shape,weights=None,include_top=False),
        'b3': efn.EfficientNetB3(input_shape=shape,weights=None,include_top=False),
        'b4': efn.EfficientNetB4(input_shape=shape,weights=None,include_top=False),
        'b5': efn.EfficientNetB5(input_shape=shape,weights=None,include_top=False),
        'b6': efn.EfficientNetB6(input_shape=shape,weights=None,include_top=False),
        'b7': efn.EfficientNetB7(input_shape=shape,weights=None,include_top=False)
    }
    return models_dict[model]

def build_model(shape=(512, 512, 1), model_class=None):
    inp = Input(shape=shape)
    base = get_efficientnet(model_class, shape)
    x = base(inp)
    x = GlobalAveragePooling2D()(x)
    inp2 = Input(shape=(4,))
    x2 = tf.keras.layers.GaussianNoise(0.2)(inp2)
    x = Concatenate()([x, x2]) 
    x = Dropout(Dropout_model)(x)
    x = Dense(1)(x)
    model = Model([inp, inp2] , x)
    return model
model_classes = ['b5'] #['b0','b1','b2','b3',b4','b5','b6','b7']
models = [build_model(shape=(512, 512, 1), model_class=m) for m in model_classes]
print('Number of models: ' + str(len(models)))

model = models[0]
model.summary()