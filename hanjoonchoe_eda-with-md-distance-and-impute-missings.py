# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import warnings

warnings.filterwarnings("ignore")



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.



import pandas as pd

import numpy as np

import geopandas as gpd

from shapely.geometry import Point, Polygon

import matplotlib.pyplot as plt

import seaborn as sns
train = pd.read_csv('/kaggle/input/bigquery-geotab-intersection-congestion/train.csv')

test = pd.read_csv('/kaggle/input/bigquery-geotab-intersection-congestion/test.csv')

train_df = train.copy()

test_df = test.copy()
#From : https://www.machinelearningplus.com/statistics/mahalanobis-distance/

import scipy as sp

from scipy import linalg

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
def AddMahala(datasets):

    list = datasets['City'].unique()

    count = datasets['City'].nunique()

    datasets['Mahalanobis'] = np.nan

  

    for i in range(count):

    

        dataset = datasets.loc[datasets['City']==list[i]]

        distance = dataset[['Longitude', 'Latitude']]

        distance = distance.round(5)

        distance = distance.drop_duplicates()

        mahala = mahalanobis(x=distance, data=distance[['Latitude','Longitude']])

    

        count2 = len(mahala)

        for j in range(count2):

            idx1 = datasets[['Longitude','Latitude']].round(5)==distance.iloc[j]

            idx1 = idx1['Longitude']&idx1['Latitude']

            datasets['Mahalanobis'].loc[idx1] = mahala[j]



    return datasets

  



train_df2 = AddMahala(train_df)
def plotMdist(dataset):

    fig = plt.figure(figsize=(20,3))

    count = dataset['City'].nunique()

    list = dataset['City'].unique()

    for i in range(count):

        plt.subplot(1, count, i+1)

        distance = dataset.loc[dataset['City']==list[i]]

        ax = sns.distplot(distance['Mahalanobis'],kde=False)

        ax.set_title(list[i])

        ax.set_xlabel('Mahalanobis_distance')

        #dataset[i]['mahala'] = distance['mahala']



plotMdist(train_df2)
def splitwrtMahal(datasets,num):

  

    list = datasets['City'].unique()

    count = len(list)

    datasets['Mahalcat'] = np.nan

  

    for i in range(count):

    

        idx = datasets['City']==list[i]

        dataset = datasets.loc[idx]

        values = dataset['Mahalanobis']

        split = np.linspace(0,1,num+1)

        values = values.quantile(split[1:-1]).values

        len_val = len(values)

        for j in (range(len_val+1)):

            if j ==0:

                idx1 = dataset['Mahalanobis']<values[j]

            elif 0<j<len_val :

                idx1 = (dataset['Mahalanobis']>=values[j-1])&(dataset['Mahalanobis']<values[j])

            else:

                idx1 = dataset['Mahalanobis']>=values[j-1]



        

            dataset['Mahalcat'].loc[idx1] = j

        datasets['Mahalcat'].loc[idx] = dataset['Mahalcat']

    return datasets

train_df3 = splitwrtMahal(train_df2,4)
train_df4 = train_df3.loc[train_df3['TotalTimeStopped_p80']>0]
train_df3.isnull().sum()
train_df3.groupby('Mahalcat').TotalTimeStopped_p80.count().plot(kind='bar')
check1 = ['DistanceToFirstStop_p80','TotalTimeStopped_p80']

check2 = ['Hour','Month','Weekend']

def MahalPlot(datasets,check1,check2):

    

    count = datasets['City'].nunique()

    list = datasets['City'].unique()

    fig,ax =plt.subplots(1,4)

    fig.set_size_inches(15, 3)

    for i in range(count):

        dataset = datasets.loc[datasets['City']==list[i]]

        

        if check2 =='Weekend':

            dataset.groupby([check2,'Mahalcat'])[check1].mean().unstack().plot(ax=ax[i],kind='bar')

        else:

            dataset.groupby([check2,'Mahalcat'])[check1].mean().unstack().plot(ax=ax[i])

        if i==0:

            ax[i].set_ylabel(check1)

        if check2 == 'Month':

            ax[i].set(xlim=(6, 12))

        ax[i].set_title(list[i])

        

    plt.show()

MahalPlot(train_df3,check1[0],check2[0])

MahalPlot(train_df3,check1[0],check2[1])

MahalPlot(train_df3,check1[0],check2[2])

MahalPlot(train_df3,check1[1],check2[0])

MahalPlot(train_df3,check1[1],check2[1])

MahalPlot(train_df3,check1[1],check2[2])
MahalPlot(train_df4,check1[0],check2[0])

MahalPlot(train_df4,check1[0],check2[1])

MahalPlot(train_df4,check1[0],check2[2])

MahalPlot(train_df4,check1[1],check2[0])

MahalPlot(train_df4,check1[1],check2[1])

MahalPlot(train_df4,check1[1],check2[2])
def trafficMap(dataset):

  

    fig,ax = plt.subplots(figsize = (10,10))

    crs = {'init' :'epsg:4326'}

    city = dataset['City'].unique()[0]

    geometry = [Point(xy) for xy in zip(dataset['Longitude'],dataset['Latitude'])]

    geo_df = gpd.GeoDataFrame(dataset, crs = crs

                          , geometry = geometry)

    minx, miny, maxx, maxy = geo_df.total_bounds

    ax.set_xlim(minx-0.01, maxx+0.01)

    ax.set_ylim(miny-0.01, maxy+0.01)

    abc = dataset.groupby(['IntersectionId']).TotalTimeStopped_p80.mean().quantile(.90)

    abc1 = dataset.groupby(['IntersectionId']).TotalTimeStopped_p80.mean().quantile(.10)

    geo_df[geo_df['TotalTimeStopped_p80']>=abc].plot(ax = ax, markersize = 0.8, color='b', marker='*', label='5')

    geo_df[(geo_df['TotalTimeStopped_p80']<=abc1)].plot(ax = ax, markersize = 0.2, color='r', marker='*', label='5')

  

    ax.set_title(city)

    plt.show()
a2 = train_df3.loc[(train_df3['Mahalcat']<=1)&(train_df3['City']=='Chicago')]

trafficMap(a2)

a2 = train_df3.loc[(train_df3['Mahalcat']>1)&(train_df3['City']=='Chicago')]

trafficMap(a2)

a2 = train_df3.loc[(train_df3['Mahalcat']<=1)&(train_df3['City']=='Philadelphia')]

trafficMap(a2)

a2 = train_df3.loc[(train_df3['Mahalcat']>1)&(train_df3['City']=='Philadelphia')]

trafficMap(a2)
train_df3.isnull().sum()
# This...

train_df3['Entry']=train_df['Path'].str.split("_").str.get(0)

train_df3['Exit']=train_df['Path'].str.split("_").str.get(2)
train_df3.groupby(['Entry','Exit']).DistanceToFirstStop_p80.mean()