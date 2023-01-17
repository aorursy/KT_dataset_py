import numpy as np

import pandas as pd

import re

import xgboost as xgb

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline



import plotly.offline as py

py.init_notebook_mode(connected=True)

import plotly.graph_objs as go

import plotly.tools as tls



from sklearn.metrics import accuracy_score

from sklearn.model_selection import KFold

from sklearn.model_selection import cross_val_score

from IPython.display import Image as PImage

from subprocess import check_call

from PIL import Image, ImageDraw, ImageFont

dataset = pd.read_csv('../input/new-york-city-airbnb-open-data/AB_NYC_2019.csv')

dataset.head(5)

# Using 'copy()' allows to clone the dataset, creating a different object with the same values

original_train = dataset.copy() 

dataset['last_review'].head(5)
dataset['last_review'] = dataset["last_review"].apply(lambda x : 0 if type(x)== float and np.isnan(x) else 1 )

dataset['last_review'].head(5)
# Remove all NULLS in the reviews_per_month column

dataset['reviews/month'] = dataset["reviews_per_month"].apply(lambda x : 0 if type(x)== float and np.isnan(x) else float(x) )

dataset['reviews/month'].head(5)
# Feature selection: remove variables no longer containing relevant information

drop_elements = ['id', 'name', 'host_id', 'host_name', 'neighbourhood','reviews_per_month']

dataset = dataset.drop(drop_elements, axis = 1)
dataset.head(5)
NB_data = dataset[{'price',

                   'minimum_nights',

                   'number_of_reviews',

                   'last_review',

                   'calculated_host_listings_count',

                   'availability_365',

                   'reviews/month'}].groupby(dataset['neighbourhood_group']).mean()

  

# printing the means value of neighbourhood_group that have all mean values 

NB_data
RM_data = dataset[{'price',

                   'minimum_nights',

                   'number_of_reviews',

                   'last_review',

                   'calculated_host_listings_count',

                   'availability_365',

                   'reviews/month'}].groupby(dataset['room_type']).mean()

  

# printing the means value of neighbourhood_group that have all mean values 

RM_data
# Count the total number of room type

CntTypRoom = dataset.groupby('room_type').size().reset_index(name='No: of room')

CntTypRoom
# Count the total number of neighbourhood group

CntNB = dataset.groupby('neighbourhood_group').size().reset_index(name='No: of NB gruop')

CntNB
rename = {'Bronx':'B', 'Brooklyn':'Bk','Manhattan':'M','Queens':'Q','Staten Island':'S'}

dataset['neighbourhood_group'] = dataset['neighbourhood_group'].replace(rename)

dataset.head(5)
reroom = {'Entire home/apt':'E', 'Private room':'P','Shared room':'S'}

dataset['room_type'] = dataset['room_type'].replace(reroom)

dataset.head(5)
reroom2 = {'E':0, 'P':1,'S':2}

dataset['room_type'] = dataset['room_type'].replace(reroom2)

dataset.head(5)
rename = {'B':1091, 'Bk':20104,'M':21661,'Q':5666,'S':373}

dataset['neighbourhood_group'] = dataset['neighbourhood_group'].replace(rename)

dataset.head(5)
import matplotlib.pyplot as plt



BBox = (dataset.longitude.min(), dataset.longitude.max(),      

         dataset.latitude.min(),dataset.latitude.max())

BBox
Map = plt.imread('../input/new-york-city-airbnb-open-data/New_York_City_.png',"rb")
fig, ax = plt.subplots(figsize = (15,15))

ax.scatter(dataset.longitude, dataset.latitude, zorder=1, alpha= 0.2, c='b', s=10)

ax.set_title('Plotting Airbnb Room Data on NewYork Map')

ax.set_xlim(BBox[0],BBox[1])

ax.set_ylim(BBox[2],BBox[3])

ax.imshow(Map, zorder=0, extent = BBox, aspect= 'equal')
SMdata =dataset[0:6000]

SMdata.head(5)
BBox2 = (SMdata.longitude.min(),SMdata.longitude.max(),      

         SMdata.latitude.min(),SMdata.latitude.max())

BBox2
fig, ax = plt.subplots(figsize = (15,15))

ax.scatter(SMdata.longitude, SMdata.latitude, zorder=1, alpha= 0.2, c='b', s=10)

ax.set_title('Plotting Airbnb Room Data on NewYork Map 1000 data')

ax.set_xlim(BBox[0],BBox[1])

ax.set_ylim(BBox[2],BBox[3])

ax.imshow(Map, zorder=0, extent = BBox, aspect= 'equal')
# Feature selection: remove variables no longer containing relevant information

drop_elements = ['neighbourhood_group',  

                 'last_review',

                 'latitude', 

                 'longitude',

                 'calculated_host_listings_count',

                 'availability_365',

                 'reviews/month']

SMdata = SMdata.drop(drop_elements, axis = 1)

SMdata.head(5)
import scipy.cluster.hierarchy as shc

#Number of customer form row index 0 to 1000

data = SMdata.iloc[0:6000,:].values 

plt.figure(figsize=(50,50))

plt.title(" Dendograms of First 6000 Customer of Airbnb ")

dend = shc.dendrogram(shc.linkage(data, method='ward'))

from sklearn.cluster import AgglomerativeClustering



cluster = AgglomerativeClustering(n_clusters=5, affinity='euclidean', linkage='ward')

cluster.fit_predict(data)
plt.figure(figsize=(15, 15))

plt.scatter(data[:,0], data[:,1], c=cluster.labels_, cmap='rainbow')