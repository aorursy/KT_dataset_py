import os

import cv2

import matplotlib.pyplot as plt

import pandas as pd

import numpy as np

import json

import math

import PIL

from PIL import ImageOps

from keras.models import Sequential, Model

from keras.layers import Dense, Flatten, Activation, Dropout, GlobalAveragePooling2D

from keras.preprocessing.image import ImageDataGenerator

from keras import optimizers, applications

from keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard, EarlyStopping

from keras import backend as K 

from keras.utils.np_utils import to_categorical

from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import train_test_split

import keras



from tqdm.auto import tqdm

tqdm.pandas()
!ls -l ../input/
!ls -l ../input/aptos2019-blindness-detection/train_images | wc -l
!ls -l ../input/aptos2019-blindness-detection/test_images | wc -l
!ls -l ../input/retinopathy-train-2015/rescaled_train_896/rescaled_train_896 | wc -l
train_path_2015 = "../input/retinopathy-train-2015/rescaled_train_896/rescaled_train_896/"

train_path = "../input/aptos2019-blindness-detection/train_images/"

test_path = "../input/aptos2019-blindness-detection/test_imges/"

df_train = pd.read_csv("../input/aptos2019-blindness-detection/train.csv")

df_train.head()
df_test = pd.read_csv("../input/aptos2019-blindness-detection/test.csv")

df_test.head()
df_train_2015 = pd.read_csv("../input/retinopathy-train-2015/rescaled_train_896/trainLabels.csv")

df_train_2015.head()
!ls -lU ../input/aptos2019-blindness-detection/train_images/ | head -10
!ls -lU ../input/retinopathy-train-2015/rescaled_train_896/rescaled_train_896 | head -10
n_rows = df_train.shape[0]

n_rows
df_train["filename"] = df_train["id_code"]+".png"

df_train["path"] = [train_path]*n_rows

#the year is just to be able to easily separate the past and present datasets later

df_train["year"] = [2019]*n_rows

df_train.head()
n_rows_2015 = df_train_2015.shape[0]

n_rows_2015
df_train_2015["filename"] = df_train_2015["image"]+".png"

df_train_2015["path"] = [train_path_2015]*n_rows_2015

df_train_2015["year"] = [2015]*n_rows_2015

df_train_2015.head()
df_train_2015.columns = ["id_code", "diagnosis", "filename", "path", "year"]

df_train_2015.head()
df_train_all = pd.concat([df_train,df_train_2015], axis=0, sort=False).reset_index()

df_train_all.head()
df_train_all.tail()
#replacing df_train with the full set to calculate features and do visualizations all at once, keeping the original (present) just in case

df_train_orig = df_train

df_train = df_train_all
%%time

img_sizes = []

widths = []

heights = []

aspect_ratios = []



for index, row in tqdm(df_train.iterrows(), total=df_train.shape[0]):

    filename = row["filename"]

    path = row["path"]

    img_path = os.path.join(path, filename)

    with open(img_path, 'rb') as f:

        img = PIL.Image.open(f)

        img_size = img.size

        img_sizes.append(img_size)

        widths.append(img_size[0])

        heights.append(img_size[1])

        aspect_ratios.append(img_size[0]/img_size[1])



df_train["width"] = widths

df_train["height"] = heights

df_train["aspect_ratio"] = aspect_ratios

df_train["size"] = img_sizes
df_train.head()
df_sorted = df_train.sort_values(by="aspect_ratio")
df_sorted.head()
df_sorted[df_sorted["year"] == 2015].head()
df_sorted[df_sorted["year"] == 2019].head()
df_sorted.tail()
df_sorted[df_sorted["year"] == 2015].tail()
df_sorted[df_sorted["year"] == 2019].tail()
#This just shows a single image in the notebook

def show_img(filename, path):

        img = PIL.Image.open(f"{path}/{filename}")

        npa = np.array(img)

        print(npa.shape)

        #https://stackoverflow.com/questions/35902302/discarding-alpha-channel-from-images-stored-as-numpy-arrays

#        npa3 = npa[ :, :, :3]

        print(filename)

        plt.imshow(npa)

import matplotlib



font = {'family' : 'normal',

        'weight' : 'normal',

        'size'   : 22}



matplotlib.rc('font', **font)
row = df_sorted[df_sorted["year"] == 2019].iloc[0]

show_img(row.filename, row.path)
row = df_sorted[df_sorted["year"] == 2015].iloc[0]

show_img(row.filename, row.path)
def plot_first_9(df_to_plot):

    plt.figure(figsize=[30,30])

    for x in range(9):

        path = df_to_plot.iloc[x].path

        filename = df_to_plot.iloc[x].filename

        img = PIL.Image.open(f"{path}/{filename}")

        print(filename)

        plt.subplot(3, 3, x+1)

        plt.imshow(img)

        title_str = filename+", diagnosis: "+str(df_to_plot.iloc[x].diagnosis)

        plt.title(title_str)
del df_sorted

df_sorted = df_train.sort_values(by="aspect_ratio", ascending=True)
plot_first_9(df_sorted[df_sorted["year"] == 2019])
plot_first_9(df_sorted[df_sorted["year"] == 2015])
del df_sorted

df_sorted = df_train.sort_values(by="aspect_ratio", ascending=False)
plot_first_9(df_sorted[df_sorted["year"] == 2019])
plot_first_9(df_sorted[df_sorted["year"] == 2015])
del df_sorted

df_sorted = df_train.sort_values(by="diagnosis", ascending=False)

df_sorted.head()
plot_first_9(df_sorted[df_sorted["year"] == 2019])
plot_first_9(df_sorted[df_sorted["year"] == 2015])
del df_sorted

df_sorted = df_train.sort_values(by="diagnosis", ascending=True)

df_sorted.head()
plot_first_9(df_sorted[df_sorted["year"] == 2019])
plot_first_9(df_sorted[df_sorted["year"] == 2015])
df_train.describe()
df_sorted = df_train.sort_values(by="width", ascending=True)



plot_first_9(df_sorted[df_sorted["year"] == 2019])


plot_first_9(df_sorted[df_sorted["year"] == 2015])