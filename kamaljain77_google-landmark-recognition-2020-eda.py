import os



import random

import seaborn as sns

import cv2





import pandas as pd

pd.set_option('display.max_colwidth', 1000)

import numpy as np

import matplotlib

import matplotlib.pyplot as plt

import PIL

import IPython.display as ipd

import glob

import h5py

import plotly.graph_objs as go

import plotly.express as px

from PIL import Image, ImageDraw

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
image_samples = os.listdir('../input/landmark-recognition-2020/')



BASE_PATH = '../input/landmark-recognition-2020'



TRAIN_DIR = f'{BASE_PATH}/train'

TRST_DIR = f'{BASE_PATH}/test'



print('Reading Data ...')

train = pd.read_csv(f'{BASE_PATH}/train.csv')

submission = pd.read_csv(f'{BASE_PATH}/sample_submission.csv')

print('Reading data is completed')
display(train.head(10))

print("Shape of train_data : ", train.shape)
display(submission.head())

print("Shape of Sample Submission", submission.shape)
# display top 10 landmarks



landmark = train.landmark_id.value_counts()

landmark_df = pd.DataFrame({'landmark_id': landmark.index, 'frequency': landmark.values}).head(10)



landmark_df['landmark_id'] = landmark_df.landmark_id.apply(lambda x: f'landmark_id_{x}')

print(landmark_df.head())



fig = px.bar(landmark_df, x="frequency", y = "landmark_id", color='landmark_id', hover_data = ["landmark_id", "frequency"],

            height = 500, title = 'Number of Images per landmark_id (Top 10 landmark_ids)'

            )



fig.show()
# display bottom 10 landmarks



landmark = train.landmark_id.value_counts()

landmark_df = pd.DataFrame({'landmark_id': landmark.index, 'frequency': landmark.values}).tail(10)



landmark_df['landmark_id'] = landmark_df.landmark_id.apply(lambda x: f'landmark_id_{x}')





fig = px.bar(landmark_df, x="frequency", y = "landmark_id", color='landmark_id', hover_data = ["landmark_id", "frequency"],

            height = 500, title = 'Number of Images per landmark_id (Top 10 landmark_ids)'

            )



fig.show()
# Missing Data in the training set

total = train.isnull().sum().sort_values(ascending= False)

percent = (train.isnull().sum()/train.isnull().count()).sort_values(ascending = False)

missing_train_data = pd.concat([total, percent], axis = 1, keys = ['Total', 'Percent'])

missing_train_data.head()
#Class distribution



plt.figure(figsize = (10, 8))

plt.title('Category Distribuition')

sns.distplot(train['landmark_id'])



plt.show()
print("Number of classes under 20 occurences",

      (train['landmark_id'].value_counts() <= 20).sum(),

      'out of total number of categories',len(train['landmark_id'].unique()))
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

lands = pd.DataFrame(train.landmark_id.value_counts())

lands.reset_index(inplace=True)

lands.columns = ['landmark_id','count']
print("Number of classes {}".format(lands.shape[0]))
print("Total of examples in train set = ",lands['count'].sum())
NUM_THRESHOLD = 50

top_lands = set(lands[lands['count'] >= NUM_THRESHOLD]['landmark_id'])

print("Number of TOP classes {}".format(len(top_lands)))
new_train = train[train['landmark_id'].isin(top_lands)]

print("Total of examples in subset of train: {}".format(new_train.shape[0]))
ax = lands['count'].plot(loglog=True, grid=True)

ax.set(xlabel="Landmarks", ylabel="Count")