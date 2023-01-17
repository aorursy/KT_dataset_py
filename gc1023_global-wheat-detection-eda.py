!pip install -q efficientnet
import os

import gc

import re



import cv2

import math

import numpy as np

import scipy as sp

import pandas as pd



import tensorflow as tf

from IPython.display import SVG

import efficientnet.tfkeras as efn

from keras.utils import plot_model

import tensorflow.keras.layers as L

from keras.utils import model_to_dot

import tensorflow.keras.backend as K

from tensorflow.keras.models import Model

from kaggle_datasets import KaggleDatasets

from tensorflow.keras.applications import DenseNet121



import seaborn as sns

from tqdm import tqdm

import matplotlib.cm as cm

from sklearn import metrics

import matplotlib.pyplot as plt

from sklearn.utils import shuffle

from sklearn.model_selection import train_test_split



tqdm.pandas()

import plotly.express as px

import plotly.graph_objects as go

import plotly.figure_factory as ff

from plotly.subplots import make_subplots



np.random.seed(0)

tf.random.set_seed(0)



import warnings

warnings.filterwarnings("ignore")
EPOCHS = 20

SAMPLE_LEN = 100

TRAIN_IMAGE_PATH = "../input/global-wheat-detection/train/"

TEST_IMAGE_PATH = "../input/global-wheat-detection/test/"

TRAIN_PATH = "../input/global-wheat-detection/train.csv"

SUB_PATH = "../input/global-wheat-detection/sample_submission.csv"



sub = pd.read_csv(SUB_PATH)

train_data = pd.read_csv(TRAIN_PATH)
train_data.head()
sub.head()
import pandas_profiling

train_data.profile_report(title='Global Wheat Detection Train')
def load_image(image_id):

    file_path = image_id + ".jpg"

    image = cv2.imread(TRAIN_IMAGE_PATH + file_path)

    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    #return cv2.cvtColor(image,cv2.COLOR_RGB2RGE)

train_images = train_data["image_id"][:SAMPLE_LEN].progress_apply(load_image)
fig = px.imshow(cv2.resize(train_images[0], (205, 136)))

fig.show()
red_values = [np.mean(train_images[idx][:, :, 0]) for idx in range(len(train_images))]

green_values = [np.mean(train_images[idx][:, :, 1]) for idx in range(len(train_images))]

blue_values = [np.mean(train_images[idx][:, :, 2]) for idx in range(len(train_images))]

values = [np.mean(train_images[idx]) for idx in range(len(train_images))]
fig = ff.create_distplot([values], group_labels=["Channels"], colors=["purple"])

fig.update_layout(showlegend=False, template="simple_white")

fig.update_layout(title_text="Distribution of channel values")

fig.data[0].marker.line.color = 'rgb(0, 0, 0)'

fig.data[0].marker.line.width = 0.5

fig
fig = ff.create_distplot([red_values], group_labels=["R"], colors=["red"])

fig.update_layout(showlegend=False, template="simple_white")

fig.update_layout(title_text="Distribution of red channel values")

fig.data[0].marker.line.color = 'rgb(0, 0, 0)'

fig.data[0].marker.line.width = 0.5

fig
fig = ff.create_distplot([green_values], group_labels=["G"], colors=["green"])

fig.update_layout(showlegend=False, template="simple_white")

fig.update_layout(title_text="Distribution of green channel values")

fig.data[0].marker.line.color = 'rgb(0, 0, 0)'

fig.data[0].marker.line.width = 0.5

fig
fig = ff.create_distplot([blue_values], group_labels=["B"], colors=["blue"])

fig.update_layout(showlegend=False, template="simple_white")

fig.update_layout(title_text="Distribution of blue channel values")

fig.data[0].marker.line.color = 'rgb(0, 0, 0)'

fig.data[0].marker.line.width = 0.5

fig
fig = go.Figure()



for idx, values in enumerate([red_values, green_values, blue_values]):

    if idx == 0:

        color = "Red"

    if idx == 1:

        color = "Green"

    if idx == 2:

        color = "Blue"

    fig.add_trace(go.Box(x=[color]*len(values), y=values, name=color, marker=dict(color=color.lower())))

    

fig.update_layout(yaxis_title="Mean value", 

                  xaxis_title="Color channel",

                  title="Mean value vs. Color channel", 

                  template="plotly_white"

                 )
fig = ff.create_distplot([red_values, green_values, blue_values],

                         group_labels=["R", "G", "B"],

                         colors=["red", "green", "blue"])

fig.update_layout(title_text="Distribution of red channel values", template="simple_white")

fig.data[0].marker.line.color = 'rgb(0, 0, 0)'

fig.data[0].marker.line.width = 0.5

fig.data[1].marker.line.color = 'rgb(0, 0, 0)'

fig.data[1].marker.line.width = 0.5

fig.data[2].marker.line.color = 'rgb(0, 0, 0)'

fig.data[2].marker.line.width = 0.5

fig