import os

import json



import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns





#To visualise the trend and analyse.

import plotly.express as px

import plotly.io as pio

pio.templates.default = "plotly_dark"



import plotly.offline as py

from plotly.offline import init_notebook_mode 





py.init_notebook_mode(connected=True)

%matplotlib inline

Train_data = "../input/herbarium-2020-fgvc7/nybg2020/train/"

Test_data = "../input/herbarium-2020-fgvc7/nybg2020/test/"

Meta_info  = "metadata.json"
import codecs

def meta_ifo():

    with codecs.open(Train_data+Meta_info,"r",encoding="utf-8",errors="ignore") as f:

        training_meta_info = json.load(f)



    with codecs.open(Test_data+Meta_info,"r",encoding="utf-8",errors="ignore") as f:

        testing_meta_info = json.load(f)

        

    return training_meta_info,testing_meta_info
train_meta_info ,test_meta_info = meta_ifo()

train_meta_info.keys()
annotations = pd.DataFrame(train_meta_info['annotations'])

annotations.columns = ['category_id', 'id', 'image_id', 'region_id']



categories = pd.DataFrame(train_meta_info['categories'])

categories.columns = ['family', 'genus', 'category_id', 'category_name']



images = pd.DataFrame(train_meta_info['images'])

images.columns = ['image_file_name', 'height', 'image_id', 'license', 'width']



licenses = pd.DataFrame(train_meta_info['licenses'])

licenses.columns = ['licenses_id', 'license_name', 'url']



regions = pd.DataFrame(train_meta_info['regions'])

regions.columns = ['region_id', 'region_name']
column_info = {

                "categories":categories.columns,

                "annotations":annotations.columns,

                "images":images.columns,

                "licenses":licenses.columns,

                "regions":regions.columns    

                }
dataframe = annotations.copy(deep=True)

dataframe = dataframe.merge(categories,on="category_id",how="outer")

dataframe = dataframe.merge(images,on="image_id",how="outer")

dataframe = dataframe.merge(regions,on="region_id",how="outer")
dataframe.sample(n=10)
imageFiles = dataframe.dropna(subset=['image_file_name'])

images  = imageFiles['image_file_name'].tolist()

train_images = ['../input/herbarium-2020-fgvc7/nybg2020/train/'+i for i in images]
imageFiles.tail()
import matplotlib.image as mpimg

max_rows = 5

max_cols = 5

pic_index = 0

pic_index += 250

fig = plt.gcf()

fig.set_size_inches(max_cols * 5 , max_rows * 5)



for i, img_path in enumerate(train_images[pic_index - 25:pic_index]):

    # Set up subplot; subplot indices start at 1

    sp = plt.subplot(max_rows, max_cols, (i+1))

    sp.axis('Off')  # Don't show axes (or gridlines)

    img = mpimg.imread(img_path)

    plt.imshow(img)



plt.show()
sortedData = dataframe.groupby(by=['category_id'],as_index=False,sort=True)['family'].count().sort_values(['family'], ascending=False)

sortedData = sortedData.head(n=10000)

sortedData.columns = ["Category","Total Specimen"]

sortedData.head()
df = px.data.gapminder()



fig = px.scatter(sortedData,

                 x="Category",

                 y="Total Specimen",

                 size="Total Specimen",

                 color="Total Specimen",

                 hover_name="Total Specimen",

                 log_x=True,

                 height=1000,

                 size_max=60)

fig.show()
imageFilesCopyDf = imageFiles.copy(deep=True)

imageFilesCopyDf = imageFiles.groupby(["height","width"]).size().reset_index(name='Total')

imageFilesCopyDf.sort_values("Total",axis=0,ascending=False)

image_training_dataset = imageFiles[["category_id","family","genus","image_file_name"]]
image_training_dataset.sample(n=10)
from sklearn.model_selection import train_test_split as TTS

train_set , validation_set= TTS(image_training_dataset,test_size=0.2,shuffle=True,random_state=42)