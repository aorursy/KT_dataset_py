import numpy as np

import pandas as pd 

import seaborn as sns

import matplotlib.pyplot as plt

import plotly.express as px

import plotly.figure_factory as ff

import plotly.graph_objects as go

from scipy import stats

import cv2

import glob

import keras

from keras.preprocessing.image import ImageDataGenerator

from keras.applications import MobileNetV2

from keras.utils import to_categorical

from keras.layers import Dense

from keras import Model

from keras.callbacks import ModelCheckpoint

from keras.models import load_model

from tensorflow.keras.applications.xception import Xception

import tensorflow as tf

import sys



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
model = keras.models.load_model("/kaggle/input/models/model_efnB3_final.h5")
sub = pd.read_csv("/kaggle/input/landmark-recognition-2020/sample_submission.csv")

sub["filename"] = sub.id.str[0]+"/"+sub.id.str[1]+"/"+sub.id.str[2]+"/"+sub.id+".jpg"

sub
test_gen = ImageDataGenerator().flow_from_dataframe(

    sub,

    directory="/kaggle/input/landmark-recognition-2020/test/",

    x_col="filename",

    y_col=None,

    weight_col=None,

    target_size=(256, 256),

    color_mode="rgb",

    classes=None,

    class_mode=None,

    batch_size=1,

    shuffle=True,

    subset=None,

    interpolation="nearest",

    validate_filenames=False)
y_pred_one_hot = model.predict_generator(test_gen, verbose=1, steps=len(sub))
y_pred = np.argmax(y_pred_one_hot, axis=-1)

y_prob = np.max(y_pred_one_hot, axis=-1)

print(y_pred.shape, y_prob.shape)
train_df=pd.read_csv('../input/landmark-recognition-2020/train.csv')

train_list = glob.glob('../input/landmark-recognition-2020/train/*/*/*/*')

train_df["filename"] = train_df.id.str[0]+"/"+train_df.id.str[1]+"/"+train_df.id.str[2]+"/"+train_df.id+".jpg"

train_df["label"] = train_df.landmark_id.astype(str)



from collections import Counter



c = train_df.landmark_id.values

count = Counter(c).most_common(1000)

print(len(count), count[-1])



# only keep 1000 classes

keep_labels = [i[0] for i in count]

train_keep = train_df[train_df.landmark_id.isin(keep_labels)]
y_uniq = np.unique(train_keep.landmark_id.values)



y_pred = [y_uniq[Y] for Y in y_pred]
for i in range(len(sub)):

    sub.loc[i, "landmarks"] = str(y_pred[i])+" "+str(y_prob[i])

sub = sub.drop(columns="filename")

sub.to_csv("submission.csv", index=False)

sub