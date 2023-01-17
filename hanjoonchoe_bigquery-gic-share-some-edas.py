# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

        

# hide warnings

import warnings

warnings.simplefilter('ignore')
import pandas as pd

import numpy as np

import geopandas as gpd

from shapely.geometry import Point, Polygon
train = pd.read_csv('/kaggle/input/bigquery-geotab-intersection-congestion/train.csv')

test = pd.read_csv('/kaggle/input/bigquery-geotab-intersection-congestion/test.csv')
train_df = train.copy()

test_df = test.copy()
train_df.isnull().sum()
train_df = train.drop(['EntryStreetName','ExitStreetName'],axis=1)
train_df.isnull().sum()
def QreIndex(dataset,name):

  

  index = {'Percentile':[],'Distance':[]}

  

  list = [name+'_p{}'.format(s) for s in [20,40,50,60,80]]

  

  data_df = pd.DataFrame(data=index)

  

  for i in list:

    

    data = pd.DataFrame(data=index)

    data['Distance'] = dataset[i].T

    data['Percentile'] = i

    data_df = pd.concat([data_df,data],ignore_index=True)

    data_df0 = data_df[data_df['Distance']>0]

    

  return data_df ,data_df0





percentile_df,percentile_df0 = QreIndex(train_df,'DistanceToFirstStop')

#quantile_df0 = quantile_df[quantile_df['Distance']>0]
import matplotlib.pyplot as plt

import seaborn as sns



Percentile_list = ['DistanceToFirstStop','TotalTimeStopped','TimeFromFirstStop']



def Qdistplot(dataset,title):

  

  index = ['Percentile','Distance']

  

  list = dataset[index[0]].unique()

  fig = plt.figure()

  

  for i in list:

    

    x = dataset[index[1]].loc[dataset[index[0]]==i]

    x = np.log(x+1)

    ax = sns.distplot(x,kde=False)

    ax.set(xlim=(min(x), max(x)))

    ax.set_ylabel('Counts')

    ax.set_xlabel('Natural Log')

    list = [j[-3:] for j in list]

    ax.legend(labels=list,loc='upper right',fontsize='small')

  ax.set_title(title)

  plt.show()

  

for i in Percentile_list:

  

  percentile_df,percentile_df0=QreIndex(train_df,i)

  Qdistplot(percentile_df,i)
percentile_df0.head()
train_df['City'].unique()
cities  = {'Atlanta':1, 'Boston':2, 'Chicago':3, 'Philadelphia':4}



  

def encode(x):

  

  if pd.isna(x):

    return 0

  

  for city in cities.keys():

    

    return cities[city]

  

  return 0



train_df['City_num'] = train_df['City'].apply(encode)

def Hours(dataset,hours):

  

  k=0

  

  for i in hours:

    

    idx = dataset['Hour']==i

    if k==0:

      idx2 = idx.copy()

      k = k+1

    else:

      idx2 = (idx2|idx)

  data = dataset.loc[idx2]

  

  return data

      
Timeline = ['Hour','Weekend','Month']

percentile = [20,40,50,60,80]

percentile2 = [20,50,80]

def TimeStopPlot(dataset,Percentile,types):

  

  Percentile_list = ['DistanceToFirstStop','TotalTimeStopped','TimeFromFirstStop']

  width=3

  fig,ax =plt.subplots(1,3)

  fig.set_size_inches(15, 3)

  

  for i in range(width):

    if types=='Weekend':

      dataset.groupby([types,'City'])[Percentile_list[i]+'_p{}'.format(Percentile)].mean().unstack().plot(kind='bar',ax=ax[i])

      ax[i].set_title(Percentile_list[i]+'_p{}'.format(Percentile))

    else:

      dataset.groupby([types,'City'])[Percentile_list[i]+'_p{}'.format(Percentile)].mean().unstack().plot(ax=ax[i])

      ax[i].set_title(Percentile_list[i]+'_p{}'.format(Percentile))

  plt.show()



def SelectiveTimeplot(dataset,Percentile,Timeline):

  

  for j in Percentile:

    

    [TimeStopPlot(train_df,j,i) for i in Timeline]

      

SelectiveTimeplot(train_df,percentile2,Timeline)
def Category(datasets,name):

  

  length = datasets[name].nunique()

  dataset = [0]*length

  list = datasets[name].unique()

  

  for i in range(length):

    

    dataset[i] = datasets[datasets[name]==list[i]]

    

  return dataset





train_city_df = Category(train_df,'City')

def trafficMap(dataset):

  

  fig,ax = plt.subplots(figsize = (15,15))

  crs = {'init' :'epsg:4326'}

  city = dataset['City'].unique()[0]

  geometry = [Point(xy) for xy in zip(dataset['Longitude'],dataset['Latitude'])]

  geo_df = gpd.GeoDataFrame(dataset, crs = crs

                          , geometry = geometry)

  minx, miny, maxx, maxy = geo_df.total_bounds

  ax.set_xlim(minx-0.01, maxx+0.01)

  ax.set_ylim(miny-0.01, maxy+0.01)

  #geo_df[geo_df['Hour']==4].plot(ax = ax, markersize = 5,color='r', marker='o', label='1')

  geo_df[geo_df['Hour']==8].plot(ax = ax, markersize = 5, color='r', marker='v', label='2')

  #geo_df[geo_df['Hour']==12].plot(ax = ax, markersize = 0.5, color='b', marker='*', label='3')

  #geo_df[geo_df['Hour']==16].plot(ax = ax, markersize = 0.3, color='g', marker='o', label='4')

  geo_df[geo_df['Hour']==24].plot(ax = ax, markersize = 1, color='b', marker='*', label='5')

  ax.set_title(city)

  plt.show()



trafficMap(train_city_df[0])

trafficMap(train_city_df[1])

trafficMap(train_city_df[2])

trafficMap(train_city_df[3])
#From : https://www.machinelearningplus.com/statistics/mahalanobis-distance/

import scipy as sp

def mahalanobis(x=None, data=None, cov=None):

    """Compute the Mahalanobis Distance between each row of x and the data  

    x    : vector or matrix of data with, say, p columns.

    data : ndarray of the distribution from which Mahalanobis distance of each observation of x is to be computed.

    cov  : covariance matrix (p x p) of the distribution. If None, will be computed from data.

    """

    x_minus_mu = x - np.mean(data)

    if not cov:

        cov = np.cov(data.values.T)

    inv_covmat = sp.linalg.inv(cov)

    left_term = np.dot(x_minus_mu, inv_covmat)

    mahal = np.dot(left_term, x_minus_mu.T)

    return mahal.diagonal()
def Compute_Mahala(dataset):

  distance = dataset[['Longitude','Latitude']]

  distance = distance.round(5)

  distance = distance.drop_duplicates()

  distance['mahala'] = mahalanobis(x=distance, data=distance[['Latitude','Longitude']])

    

  return distance

  

def Mahalanobis_dist(dataset):

  

  fig = plt.figure(figsize=(20,3))

  

  for i in range(len(dataset)):

    plt.subplot(1, 5, i+1)

    distance = Compute_Mahala(dataset[i])

    city = dataset[i]['City'].unique()[0]

    ax = sns.distplot(distance['mahala'],kde=False)

    ax.set_title(city)

    ax.set_xlabel('Mahalanobis_distance')

    dataset[i]['mahala'] = distance['mahala']

  return dataset



train_city_df = Mahalanobis_dist(train_city_df)
distance = train_city_df[0][['Longitude','Latitude']]

distance = distance.round(5)

distance = distance.drop_duplicates()

distance['mahala'] = mahalanobis(x=distance, data=distance[['Latitude','Longitude']])