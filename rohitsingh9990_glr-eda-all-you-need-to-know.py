import os





import random

import seaborn as sns

import cv2



# General packages

import pandas as pd

import numpy as np

import matplotlib

import matplotlib.pyplot as plt

import PIL

import IPython.display as ipd

import glob

import h5py

import plotly.graph_objs as go

import plotly.express as px

from PIL import Image

from tempfile import mktemp



from bokeh.layouts import column, row

from bokeh.models import ColumnDataSource, LinearAxis, Range1d

from bokeh.models.tools import HoverTool

from bokeh.palettes import BuGn4

from bokeh.plotting import figure, output_notebook, show

from bokeh.transform import cumsum

from math import pi



output_notebook()





from IPython.display import Image, display

import warnings

warnings.filterwarnings("ignore")
os.listdir('../input/landmark-recognition-2020/')

BASE_PATH = '../input/landmark-recognition-2020'



TRAIN_DIR = f'{BASE_PATH}/train'

TEST_DIR = f'{BASE_PATH}/test'



print('Reading data...')

train = pd.read_csv(f'{BASE_PATH}/train.csv')

submission = pd.read_csv(f'{BASE_PATH}/sample_submission.csv')

print('Reading data completed')
display(train.head())

print("Shape of train_data :", train.shape)
display(submission.head())

print("Shape of submission :", submission.shape)
# displaying only top 30 landmark

landmark = train.landmark_id.value_counts()

landmark_df = pd.DataFrame({'landmark_id':landmark.index, 'frequency':landmark.values}).head(30)



landmark_df['landmark_id'] =   landmark_df.landmark_id.apply(lambda x: f'landmark_id_{x}')



fig = px.bar(landmark_df, x="frequency", y="landmark_id",color='landmark_id', orientation='h',

             hover_data=["landmark_id", "frequency"],

             height=1000,

             title='Number of images per landmark_id (Top 30 landmark_ids)')

fig.show()
import PIL

from PIL import Image, ImageDraw





def display_images(images, title=None): 

    f, ax = plt.subplots(5,5, figsize=(18,22))

    if title:

        f.suptitle(title, fontsize = 30)



    for i, image_id in enumerate(images):

        image_path = os.path.join(TRAIN_DIR, f'{image_id[0]}/{image_id[1]}/{image_id[2]}/{image_id}.jpg')

        image = Image.open(image_path)

        

        ax[i//5, i%5].imshow(image) 

        image.close()       

        ax[i//5, i%5].axis('off')



        landmark_id = train[train.id==image_id.split('.')[0]].landmark_id.values[0]

        ax[i//5, i%5].set_title(f"ID: {image_id.split('.')[0]}\nLandmark_id: {landmark_id}", fontsize="12")



    plt.show() 
samples = train.sample(25).id.values

display_images(samples)
samples = train[train.landmark_id == 138982].sample(25).id.values



display_images(samples)
samples = train[train.landmark_id == 126637].sample(25).id.values



display_images(samples)
samples = train[train.landmark_id == 20409].sample(25).id.values



display_images(samples)
samples = train[train.landmark_id == 83144].sample(25).id.values



display_images(samples)