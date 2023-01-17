!pip freeze > '../working/requirements.txt'
import os

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import csv

from tqdm import tqdm

from glob import glob

import random

import cv2

import matplotlib.gridspec as gridspec

import seaborn as sns

import zlib

import itertools

import scipy

import skimage

from skimage.transform import resize

import sklearn

from sklearn.model_selection import train_test_split

from keras.utils.np_utils import to_categorical

from tqdm import tqdm

import warnings

warnings.filterwarnings("ignore")

%matplotlib inline
imageSize=75

path = "../input/nematode cells/nematode cells/"

def get_data(folder):

    X = []

    y = []

    for folderName in os.listdir(folder):

        if not folderName.startswith('.'):

            if folderName in ['interphase']:

                label = 0

            elif folderName in ['mitosis']:

                label = 1             

            else:

                label = 2

            for image_filename in tqdm(os.listdir(folder + folderName)):

                img_file = cv2.imread(folder + folderName + '/' + image_filename)

                if img_file is not None:

                    img_file = skimage.transform.resize(img_file, (imageSize, imageSize, 3))

                    img_arr = np.asarray(img_file)

                    X.append(img_arr)

                    y.append(label)

    X = np.asarray(X)

    y = np.asarray(y)

    return X,y

X2, y2= get_data(path)



X_train2, X_test2, y_train2, y_test2 = train_test_split(X2, y2, test_size=0.2) 

y_trainHot2 = to_categorical(y_train2, num_classes = 2)

y_testHot2 = to_categorical(y_test2, num_classes = 2)
def plotHistogram(a):

    """

    Plot histogram of RGB Pixel Intensities

    """

    plt.figure(figsize=(10,5))

    plt.subplot(1,2,1)

    plt.imshow(a)

    plt.axis('off')

    histo = plt.subplot(1,2,2)

    histo.set_ylabel('Count')

    histo.set_xlabel('Pixel Intensity')

    n_bins = 30

    plt.hist(a[:,:,0].flatten(), bins= n_bins, lw = 0, color='r', alpha=0.5);

    plt.hist(a[:,:,1].flatten(), bins= n_bins, lw = 0, color='g', alpha=0.5);

    plt.hist(a[:,:,2].flatten(), bins= n_bins, lw = 0, color='b', alpha=0.5);

plotHistogram(X_train2[1])
print("Interphase")

multipleImages = glob("../input/nematode cells/nematode cells/interphase/**")

i_ = 0

plt.rcParams['figure.figsize'] = (10.0, 10.0)

plt.subplots_adjust(wspace=0, hspace=0)

for l in multipleImages[:25]:

    im = cv2.imread(l)

    im = cv2.resize(im, (128, 128)) 

    plt.subplot(5, 5, i_+1) #.set_title(l)

    plt.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB)); plt.axis('off')

    i_ += 1
print("Mitosis")

multipleImages = glob("../input/nematode cells/nematode cells/mitosis/**")

i_ = 0

plt.rcParams['figure.figsize'] = (10.0, 10.0)

plt.subplots_adjust(wspace=0, hspace=0)

for l in multipleImages[:25]:

    im = cv2.imread(l)

    im = cv2.resize(im, (128, 128)) 

    plt.subplot(5, 5, i_+1) #.set_title(l)

    plt.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB)); plt.axis('off')

    i_ += 1
map_characters = {0: 'Interphase', 1: 'Mitosis'}

dict_characters=map_characters

df = pd.DataFrame()

df["labels"]=y_train2

lab = df['labels']

dist = lab.value_counts()

sns.countplot(lab)

print(dict_characters)
imageSize=75

path = "../input/jurkat cells (merged only)/jurkat cells (merged only)/"

def get_data(folder):

    X = []

    y = []

    for folderName in os.listdir(folder):

        if not folderName.startswith('.'):

            if folderName in ['G1-G2-S']:

                label = 0

            elif folderName in ['Anaphase']:

                label = 1

            elif folderName in ['Metaphase']:

                label = 2

            elif folderName in ['Prophase']:

                label = 3                                

            elif folderName in ['Telophase']:

                label = 4                

            else:

                label = 5

            for image_filename in tqdm(os.listdir(folder + folderName)):

                img_file = cv2.imread(folder + folderName + '/' + image_filename)

                if img_file is not None:

                    img_file = skimage.transform.resize(img_file, (imageSize, imageSize, 3))

                    img_arr = np.asarray(img_file)

                    X.append(img_arr)

                    y.append(label)

    X = np.asarray(X)

    y = np.asarray(y)

    return X,y

X, y= get_data(path)



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

y_trainHot = to_categorical(y_train, num_classes = 5)

y_testHot = to_categorical(y_test, num_classes = 5)
plotHistogram(X_train[1])
print("Interphase")

multipleImages = glob("../input/jurkat cells (merged only)/jurkat cells (merged only)/G1-G2-S/**")

i_ = 0

plt.rcParams['figure.figsize'] = (10.0, 10.0)

plt.subplots_adjust(wspace=0, hspace=0)

for l in multipleImages[:25]:

    im = cv2.imread(l)

    im = cv2.resize(im, (128, 128)) 

    plt.subplot(5, 5, i_+1) #.set_title(l)

    plt.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB)); plt.axis('off')

    i_ += 1
print("Mitosis")

multipleImages = glob('../input/jurkat cells (merged only)/jurkat cells (merged only)/Metaphase/**')

i_ = 0

plt.rcParams['figure.figsize'] = (10.0, 10.0)

plt.subplots_adjust(wspace=0, hspace=0)

for l in multipleImages[:25]:

    im = cv2.imread(l)

    im = cv2.resize(im, (128, 128)) 

    plt.subplot(5, 5, i_+1) #.set_title(l)

    plt.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB)); plt.axis('off')

    i_ += 1
map_characters = {0: 'G1/G2/S', 1: 'Anaphase', 2: 'Metaphase',3:'Prophase',4:'Telophase'}

dict_characters=map_characters

df = pd.DataFrame()

df["labels"]=y_train

lab = df['labels']

dist = lab.value_counts()

sns.countplot(lab)

print(dict_characters)
# dependencies for the fastai part of this kernel

# note that this half of the kernel is independent of the analysis performed above

# !pip install numpy==1.16.0

# !pip install pandas==0.23.4

# !pip install matplotlib==2.2.3

# !pip install torch==1.0.0

# !pip install fastai==1.0.39

X=[];y=[];X_train=[];y_train=[];X_test=[];y_test=[]
# adapted from lesson 1 of the fastai v1 course (see forums.fast.ai for more detail)

# turn on GPU and enable internet



import numpy as np

import pandas as pd

import matplotlib

import matplotlib.pyplot as plt

import torch

import fastai

from fastai import *

from fastai.vision import *

from fastai.callbacks.hooks import *



print('numpy version: ',np.__version__)

print('pandas version: ',pd.__version__)

print('matplotlib version: ',matplotlib.__version__)

print('pytorch version: ',torch.__version__)

print('fastai version: ',fastai.__version__)



%reload_ext autoreload

%autoreload 2

%matplotlib inline
img_dir='../input/jurkat cells (merged only)/jurkat cells (merged only)/'

path=Path(img_dir)

data = ImageDataBunch.from_folder(path, train=".", 

                                  valid_pct=0.3,

                                  ds_tfms=get_transforms(do_flip=True,flip_vert=True, max_rotate=90,max_lighting=0.3),

                                  size=224,bs=64, 

                                  num_workers=0).normalize(imagenet_stats)

print(f'Classes: \n {data.classes}')

data.show_batch(rows=4, figsize=(10,10))
learn = create_cnn(data, models.resnet34, metrics=accuracy, model_dir="/tmp/model/")

learn.fit_one_cycle(10)
interp = ClassificationInterpretation.from_learner(learn)

interp.plot_confusion_matrix(figsize=(4,4), dpi=60)