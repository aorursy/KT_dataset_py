!pip install itk --quiet

!pip install itkwidgets --quiet
import os

import numpy as np 

import pandas as pd 



import openslide

import gc

import matplotlib.pyplot as plt

from collections import defaultdict



from PIL import Image

from tqdm import tqdm



import plotly

import plotly.express as px

import plotly.graph_objects as go

from plotly.subplots import make_subplots





from IPython.display import FileLinks



import itk

import itkwidgets

from ipywidgets import interact , interactive , IntSlider , ToggleButtons

from ipywidgets import interact

plotly.offline.init_notebook_mode (True)
train_images_dir = '../input/prostate-cancer-grade-assessment/train_images/'

train_images     = os.listdir (train_images_dir)
train_csv = pd.read_csv ('../input/prostate-cancer-grade-assessment/train.csv')
train_images  = []



for i , image in enumerate(tqdm(os.listdir (train_images_dir))):

    img             =  image 

    img_size_MB     = f"{os.stat(train_images_dir+image).st_size / 1024 **2 : 1.2f} " 

    wsi             = openslide.OpenSlide (train_images_dir + image)

    train_images.append((img ,  

                         img_size_MB ,

                         wsi.level_dimensions[0] , 

                         wsi.level_dimensions[1] , 

                         wsi.level_dimensions[2] , 

                         np.product(wsi.level_dimensions[0]) * 3, 

                         np.product(wsi.level_dimensions[1]) * 3, 

                         np.product(wsi.level_dimensions[2]) * 3

                        ))

    gc.collect()

    

train_images = pd.DataFrame ( train_images , 

                              columns = ['img' , 

                                  'Image Size (MB)' ,

                                  'Image Shape Level0' ,

                                  'Image Shape Level1' , 

                                  'Image Shape Level2' , 

                                  'Total Pixels Level0' , 

                                  'Total Pixels Level1' , 

                                  'Total Pixels Level2']

                            )
train_images.head()
train_images['Image Size (MB)'] = train_images['Image Size (MB)'].astype('float32')
def image_name(img) :

    return img.split('.')[0]



train_images['img'] = train_images['img'].map(image_name)
train_images.shape  , train_csv.shape , train_images['img'].nunique()  , train_csv['image_id'].nunique()
train_df = train_csv.merge (train_images , 

                            left_on = "image_id" , 

                            right_on = "img")
train_df.shape
#Drop duplicate column

train_df.drop('img', axis = 1, inplace= True)
train_df.head()
del train_images , train_csv
def get_width (image) :

    return image[0]

def get_height (image) :

    return image[1]
width  = train_df['Image Shape Level0'].map(get_width)

height = train_df['Image Shape Level0'].map(get_height) 



fig    = px.scatter (train_df , 

            x =  width , 

            y = height , 

            color = width,

           title = 'Image Size in Pixels - Level 0 (highest) Resolution)')



fig.update_layout ( yaxis=dict(title_text="Height") , 

                    xaxis=dict(title_text="Width") , 

                    title_font_family="Open Sans"

                  )
width  = train_df['Image Shape Level1'].map(get_width)

height = train_df['Image Shape Level1'].map(get_height) 

fig    =  px.scatter (train_df , 

            x =  width , 

            y = height , 

            color = width , 

            title = 'Image Size in Pixels - Level 1 Resolution)')



fig.update_layout (yaxis=dict(title_text="Height") , 

                    xaxis=dict(title_text="Width") , 

                    title_font_family="Open Sans")

width  = train_df['Image Shape Level2'].map(get_width)

height = train_df['Image Shape Level2'].map(get_height) 

fig    = px.scatter (train_df , 

            x =  width , 

            y = height , 

            color = width , 

            title = 'Image Size in Pixels - Level 2 (lowest) Resolution)')

fig.update_layout (yaxis=dict(title_text="Height") , 

                    xaxis=dict(title_text="Width") , 

                    title_font_family="Open Sans")

def change_level (level):

    width  = train_df[f'Image Shape Level{level}'].map(get_width)

    height = train_df[f'Image Shape Level{level}'].map(get_height) 

    fig = px.scatter (train_df , 

            x =  width , 

            y = height , 

            color = width , 

            title = f'Image Size in Pixels - Level {level} (Lowest Resolution)')

    

    fig.update_layout ( yaxis=dict(title_text="Height") , 

                    xaxis=dict(title_text="Width") , 

                    title_font_family="Open Sans")

    fig.show()

    return level
interact (change_level , level = (0 , 2))
fig = px.histogram (train_df ,

              x = ['Total Pixels Level0'] ,

              hover_data = ['gleason_score' , 'isup_grade'],

              color = 'data_provider' ,

              marginal = 'rug',

              title = 'Pixel Distribution at Level 0')



fig.update_layout ( yaxis=dict(title_text="Images Count") , 

                    xaxis=dict(title_text="Total Number of Pixels") , 

                    title_font_family="Open Sans")

fig.show()
fig = px.histogram (train_df ,

              x = ['Total Pixels Level1'] ,

              hover_data = ['gleason_score' , 'isup_grade'],

              color = 'data_provider' ,

              marginal = 'rug',

              title = 'Pixel Distribution at Level 1')

fig.update_layout ( yaxis=dict(title_text="Images Count") , 

                    xaxis=dict(title_text="Pixel Size") , 

                    title_font_family="Open Sans")

fig.show()
fig = px.histogram (train_df ,

              x = ['Total Pixels Level2'] ,

              hover_data = ['gleason_score' , 'isup_grade'],

              color = 'data_provider' ,

              marginal = 'rug',

              title = 'Pixel Distribution at Level 2'

                   )



fig.update_layout ( yaxis=dict(title_text="Images Count") , 

                    xaxis=dict(title_text="Pixel Size") , 

                    title_font_family="Open Sans")

fig.show()
def change_level (level):

    width  = train_df[f'Image Shape Level{level}'].map(get_width)

    height = train_df[f'Image Shape Level{level}'].map(get_height) 

    fig = px.scatter (train_df , 

            x =  width , 

            y = height , 

            color = train_df['data_provider'] , 

            title = f'Image Size Distribution across Data providers')

    fig.update_traces(marker=dict(size=12,

                      line=dict(width=2,color='DarkSlateGrey')),

                      selector=dict(mode='markers')

                  )



    fig.show()

    return level
interact (change_level , level = (0 , 2))
fig = make_subplots(rows=1, 

                    cols=3 , 

                    subplot_titles=("Level 0", 

                                    "Level 1" , 

                                    "Level 2")

                   )

fig.add_trace(go.Violin(

                        x = train_df['isup_grade'] ,

                        y = train_df['Total Pixels Level0'], 

                        points = 'all', name = "Level 0"

                ), row = 1, col = 1 )



fig.add_trace(go.Violin(

                        x = train_df['isup_grade'] ,

                        y = train_df['Total Pixels Level1'], 

                        points = 'all' , name = "Level 1"

                ) , row = 1, col = 2 )





fig.add_trace(go.Violin(

                        x = train_df['isup_grade'] ,

                        y = train_df['Total Pixels Level2'], 

                        points = 'all', name = "Level 2" 

                ), row =1  , col = 3 )



fig.update_layout(

    autosize=True , 

    width=2000,

    height=500 , 

    title = 'Total Pixels distribution by Grading '

)
