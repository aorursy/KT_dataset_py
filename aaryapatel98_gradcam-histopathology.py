import keras

from keras.applications.densenet import DenseNet201, preprocess_input,decode_predictions

from keras.layers import Dense, Input, Dropout, MaxPooling2D, Concatenate, GlobalAveragePooling2D, GlobalMaxPooling2D, Flatten, Concatenate

from keras.models import Model

import pandas as pd

from random import shuffle

import numpy as np

import cv2

import glob

import gc

import os

import tensorflow as tf

from keras.regularizers import l2

from keras.preprocessing.image import ImageDataGenerator

from keras.layers import Conv2D, MaxPooling2D

from keras.layers import Dense, Dropout, Flatten, Activation, Input, BatchNormalization, Add, GlobalAveragePooling2D,AveragePooling2D,GlobalMaxPooling2D,concatenate

from keras.layers import Lambda, Reshape, DepthwiseConv2D, ZeroPadding2D, Add, MaxPooling2D,Activation, Flatten, Conv2D, Dense, Input, Dropout, Concatenate, GlobalMaxPooling2D, GlobalAveragePooling2D, BatchNormalization



from keras.models import Sequential

from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint,TensorBoard,TerminateOnNaN

from keras.optimizers import Adam,RMSprop

from keras.models import Model,load_model

from keras.applications import NASNetMobile,MobileNetV2,densenet,resnet50,xception



from keras_applications.resnext import ResNeXt50

from albumentations import Resize,Compose, RandomRotate90, Transpose, Flip, OneOf, CLAHE, IAASharpen, IAAEmboss, RandomBrightnessContrast, JpegCompression, Blur, GaussNoise, HueSaturationValue, ShiftScaleRotate, Normalize





from sklearn.utils import shuffle

from sklearn.metrics import confusion_matrix

from sklearn.model_selection import train_test_split,StratifiedKFold

from skimage import data, exposure

import itertools

import shutil

import matplotlib.pyplot as plt

os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

%matplotlib inline
df = pd.read_csv('../input/histopathologic-cancer-detection/train_labels.csv')
init_model = load_model('../input/densenet-8020/densenet169_one_cycle_model.h5')
!pip install vis
import keras

import tensorflow as tf

import vis ## keras-vis

import matplotlib.pyplot as plt

import numpy as np

print("keras      {}".format(keras.__version__))

print("tensorflow {}".format(tf.__version__))
# init_model.summary()

# for ilayer, layer in enumerate(init_model.layers):

#     print("{:3.0f} {:10}".format(ilayer, layer.name))
class_label = ['no_tumor','tumor']
!pip install keras-vis
lis0 = df.loc[df['label'] == 0]['id'].tolist()
lis1 = df.loc[df['label'] == 1]['id'].tolist()
from keras.preprocessing.image import load_img, img_to_array

#_img = load_img("duck.jpg",target_size=(224,224))

_img0 = load_img('/kaggle/input/histopathologic-cancer-detection/train/' + lis0[15] + '.tif')

plt.imshow(_img0)

plt.show()
bgr = cv2.imread('/kaggle/input/histopathologic-cancer-detection/train/' + lis0[15] + '.tif')



lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)



lab_planes = cv2.split(lab)



clahe = cv2.createCLAHE(clipLimit=40,tileGridSize=(20,20))



lab_planes[0] = clahe.apply(lab_planes[0])



lab = cv2.merge(lab_planes)



bgr0 = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
plt.imshow(bgr0)
from keras.preprocessing.image import load_img, img_to_array

#_img = load_img("duck.jpg",target_size=(224,224))

_img1 = load_img('/kaggle/input/histopathologic-cancer-detection/train/' + lis1[10001] + '.tif')

plt.imshow(_img1)

plt.show()
bgr = cv2.imread('/kaggle/input/histopathologic-cancer-detection/train/' + lis1[10001] + '.tif')



lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)



lab_planes = cv2.split(lab)



clahe = cv2.createCLAHE(clipLimit=20,tileGridSize=(9,9))



lab_planes[0] = clahe.apply(lab_planes[0])



lab = cv2.merge(lab_planes)



bgr1 = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
plt.imshow(bgr1)
plt.figure(figsize=(8,8))

plt.figure(1)

plt.subplot(121)

plt.title('Input Image')

plt.imshow(_img1)



plt.subplot(122)

plt.title('CLAHE Preprocessed Image')

plt.imshow(bgr1)

plt.show()

plt.savefig('clahe.png')
img0               = img_to_array(_img0)

img0               = preprocess_input(img0)

y_pred0            = init_model.predict(img0[np.newaxis,...])

class_index = 0
img1       = img_to_array(_img1)

img1               = preprocess_input(img1)

y_pred1            = init_model.predict(img1[np.newaxis,...])

class_index = 0
from vis.utils import utils

# Utility to search for layer index by name. 

# Alternatively we can specify this as -1 since it corresponds to the last layer.

layer_idx = utils.find_layer_idx(init_model, 'dense_3')

# Swap softmax with linear

init_model.layers[layer_idx].activation = keras.activations.linear

model = utils.apply_modifications(init_model)
img_t_0 = load_img('/kaggle/input/histopathologic-cancer-detection/train/' + lis1[10001] + '.tif')

img_t_1 = load_img('/kaggle/input/histopathologic-cancer-detection/train/' + lis1[10002] + '.tif')

img_t_2 = load_img('/kaggle/input/histopathologic-cancer-detection/train/' + lis1[10003] + '.tif')

img_t_3 = load_img('/kaggle/input/histopathologic-cancer-detection/train/' + lis1[10004] + '.tif')
img_nt_0 = load_img('/kaggle/input/histopathologic-cancer-detection/train/' + lis0[15] + '.tif')

img_nt_1 = load_img('/kaggle/input/histopathologic-cancer-detection/train/' + lis0[16] + '.tif')

img_nt_2 = load_img('/kaggle/input/histopathologic-cancer-detection/train/' + lis0[1001] + '.tif')

img_nt_3 = load_img('/kaggle/input/histopathologic-cancer-detection/train/' + lis0[101] + '.tif')
lis_img_t = [img_t_0,img_t_1,img_t_2,img_t_3]

lis_img_nt = [img_nt_0,img_nt_1,img_nt_2,img_nt_3]
from vis.visualization import visualize_cam

penultimate_layer_idx = utils.find_layer_idx(model, "relu") 



seed_input = img0

grad_top1_0_  = visualize_cam(model, layer_idx, 0, seed_input, 

                           penultimate_layer_idx = penultimate_layer_idx,#None,

                           backprop_modifier = None,

                           grad_modifier = None)
from vis.visualization import visualize_cam

penultimate_layer_idx = utils.find_layer_idx(model, "relu") 



seed_input = img1

grad_top1_1_  = visualize_cam(model, layer_idx, 0, seed_input, 

                           penultimate_layer_idx = penultimate_layer_idx,#None,

                           backprop_modifier = None,

                           grad_modifier = None)
_img_t_0               = img_to_array(img_t_0)

_img_t_0               = preprocess_input(_img_t_0)

y_pred_0_t            = init_model.predict(_img_t_0[np.newaxis,...])



seed_input = _img_t_0



grad_top1_t_0  = visualize_cam(model, layer_idx, 0, seed_input, 

                           penultimate_layer_idx = penultimate_layer_idx,#None,

                           backprop_modifier = None,

                           grad_modifier = None)



_img_t_1               = img_to_array(img_t_1)

_img_t_1               = preprocess_input(_img_t_1)

y_pred_1_t            = init_model.predict(_img_t_1[np.newaxis,...])



seed_input = _img_t_1



grad_top1_t_1  = visualize_cam(model, layer_idx, 0, seed_input, 

                           penultimate_layer_idx = penultimate_layer_idx,#None,

                           backprop_modifier = None,

                           grad_modifier = None)



_img_t_2               = img_to_array(img_t_2)

_img_t_2               = preprocess_input(_img_t_2)

y_pred_2_t            = init_model.predict(_img_t_2[np.newaxis,...])



seed_input = _img_t_2



grad_top1_t_2  = visualize_cam(model, layer_idx, 0, seed_input, 

                           penultimate_layer_idx = penultimate_layer_idx,#None,

                           backprop_modifier = None,

                           grad_modifier = None)



_img_t_3               = img_to_array(img_t_3)

_img_t_3               = preprocess_input(_img_t_3)

y_pred_3_t            = init_model.predict(_img_t_3[np.newaxis,...])



seed_input = _img_t_3

grad_top1_t_3  = visualize_cam(model, layer_idx, 0, seed_input, 

                           penultimate_layer_idx = penultimate_layer_idx,#None,

                           backprop_modifier = None,

                           grad_modifier = None)
_img_nt_0               = img_to_array(img_nt_0)

_img_nt_0               = preprocess_input(_img_nt_0)

y_pred_0_nt            = init_model.predict(_img_nt_0[np.newaxis,...])



seed_input = _img_nt_0



grad_top1_nt_0  = visualize_cam(model, layer_idx, 0, seed_input, 

                           penultimate_layer_idx = penultimate_layer_idx,#None,

                           backprop_modifier = None,

                           grad_modifier = None)



_img_nt_1               = img_to_array(img_nt_1)

_img_nt_1               = preprocess_input(_img_nt_1)

y_pred_1_nt            = init_model.predict(_img_nt_1[np.newaxis,...])



seed_input = _img_nt_1



grad_top1_nt_1  = visualize_cam(model, layer_idx, 0, seed_input, 

                           penultimate_layer_idx = penultimate_layer_idx,#None,

                           backprop_modifier = None,

                           grad_modifier = None)



_img_nt_2               = img_to_array(img_nt_2)

_img_nt_2               = preprocess_input(_img_nt_2)

y_pred_2_nt            = init_model.predict(_img_nt_2[np.newaxis,...])



seed_input = _img_nt_2



grad_top1_nt_2  = visualize_cam(model, layer_idx, 0, seed_input, 

                           penultimate_layer_idx = penultimate_layer_idx,#None,

                           backprop_modifier = None,

                           grad_modifier = None)



_img_nt_3               = img_to_array(img_nt_3)

_img_nt_3               = preprocess_input(_img_nt_3)

y_pred_3_nt            = init_model.predict(_img_nt_3[np.newaxis,...])



seed_input = _img_nt_3

grad_top1_nt_3  = visualize_cam(model, layer_idx, 0, seed_input, 

                           penultimate_layer_idx = penultimate_layer_idx,#None,

                           backprop_modifier = None,

                           grad_modifier = None)
ypred_nt = [y_pred_0_nt,y_pred_1_nt,y_pred_2_nt,y_pred_3_nt]

ypred_t = [y_pred_0_t,y_pred_1_t,y_pred_2_t,y_pred_3_t]

nt = [grad_top1_nt_0,grad_top1_nt_1,grad_top1_nt_2,grad_top1_nt_3]

t = [grad_top1_t_0,grad_top1_t_1,grad_top1_t_2,grad_top1_t_3]
def plot_map(grads):

    fig, axes = plt.subplots(1,2,figsize=(14,5))

    axes[0].imshow(_img0)

    axes[1].imshow(_img0)

    i = axes[1].imshow(grads,cmap="jet",alpha=0.8)

    fig.colorbar(i)

    plt.suptitle("Pr(class={}) = {:5.6f}".format(

                      class_label[0],

                      y_pred0[0,0]))

    plt.savefig('no_tumor_class.png')

plot_map(grad_top1_nt_0)
def plot_map(grads):

    fig, axes = plt.subplots(1,2,figsize=(14,5))

    axes[0].imshow(_img1)

    axes[1].imshow(_img1)

    i = axes[1].imshow(grads,cmap="jet",alpha=0.8)

    fig.colorbar(i)

    plt.suptitle("Pr(class={}) = {:5.6f}".format(

                      class_label[1],

                      y_pred1[0,0]))

    plt.savefig('tumor_class.png')

plot_map(grad_top1_t_0)
def plot_map():

    fig, axes = plt.subplots(2,4, figsize=(16,12))

    fig.suptitle('Grad-CAM\nPredicted / Actual / Probability',fontsize=20)

    for i in range(4):

        axes[0,i].imshow(lis_img_nt[i])

        axes[0,i].imshow(nt[i],cmap="jet",alpha=0.8)

        axes[0,i].set_xticks([])

        axes[0,i].set_yticks([])

        axes[0,i].set_title(f'{class_label[ypred_nt[i][0,0] > 0.5]} / {class_label[0]} / {ypred_nt[i][0,0]:.4f}')

    axes[0,0].set_ylabel('Non Tumor\nSamples', fontsize=16, rotation=0, labelpad=80)

    for i in range(4):

        axes[1,i].imshow(lis_img_t[i])

        axes[1,i].imshow(t[i],cmap="jet",alpha=0.8)

        axes[1,i].set_xticks([])

        axes[1,i].set_yticks([])

        axes[1,i].set_title(f'{class_label[ypred_t[i][0,0] > 0.5]} / {class_label[1]} / {ypred_t[i][0,0]:.4f}')

    axes[1,0].set_ylabel('Tumor Samples', fontsize=16, rotation=0, labelpad=80)

    plt.subplots_adjust(wspace=0, hspace=0)

    plt.savefig('tumor_class.png')

plot_map()