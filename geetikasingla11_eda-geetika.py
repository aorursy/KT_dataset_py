# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
import numpy as np                                                    #importing all the necessary libraraies

import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

import warnings

warnings.filterwarnings('ignore')

from sklearn.preprocessing import StandardScaler, PowerTransformer
data     = pd.read_csv('../input/spotifydata-19212020/data.csv')                    #reading data file

print('Rows in Data       :',data.shape[0])                                         #display no of rows

print('Columns in Data    :',data.shape[1],'\n',data.columns)                       #display no of columns
data.info()                                                                                      #data.info()
data.describe(include='all')                                                                 #data.describe()
cat_cols = [cols for cols in data if data[cols].dtypes=='O']                    #checking category columns

num_cols = [cols for cols in data if data[cols].dtypes=='float64']              #checking continuous columns

int_cols = [cols for cols in data if data[cols].dtypes=='int64']                #checking discrete columns

print('Category Columns    :',cat_cols)

print('Continuous Columns  :',num_cols)

print('Discrete Columns    :',int_cols)
for cols in cat_cols:                                                            #checking same songs/artists

    print(cols ,len(data[cols].value_counts()))

print(data.duplicated().value_counts())
plt.figure(figsize=(25,6))

plt.subplot(131)

sns.countplot(data['key'])

plt.subplot(132)

sns.countplot(data['explicit'])

plt.subplot(133)

sns.countplot(data['mode'])

plt.show()
for cols in num_cols:

    plt.figure(figsize=(20,5))

    

    plt.subplot(131)

    sns.kdeplot(data[cols],color='g',shade=True)

    plt.title(cols+' Distribution')



    plt.subplot(132)

    sns.boxplot(y = data[cols],color='pink')

    plt.title(cols+' Boxplot')

    

    plt.subplot(133)

    sns.scatterplot(x=cols,y='popularity',data=data)

    plt.title(cols+' versus popularity')

    

    plt.show()
rows=5

columns=3

fig=plt.figure(figsize=(25,40))

for i,cols in enumerate (data.select_dtypes(include=['int64','float64']).columns):

    ax=plt.subplot(rows,columns,i+1)

    sns.lineplot(x='year',y=cols,data=data,ax=ax,color='r')

    ax.set_title(cols+'yearwise')

fig.tight_layout()

plt.show()
plt.figure(figsize=(12,10))

sns.heatmap(data.corr(),annot=True)

b,t=plt.ylim()

b+=0.5

t-=0.5

plt.ylim(b,t)

plt.show()
print(data.skew())
data.min()
#As we can see our data is skewed and also has some negative values, we will transform this to make our data more normal and positive.

pt = PowerTransformer(method='yeo-johnson', standardize=True) 

data_scaled = pd.DataFrame(pt.fit_transform(data[['duration_ms','instrumentalness','liveness','speechiness','loudness']]), 

                           columns=['duration_ms','instrumentalness','liveness','speechiness','loudness'])
data_scaled.min()
data_scaled.skew()
num_cols = ['duration_ms','instrumentalness','liveness','speechiness','loudness']

for cols in num_cols:

    plt.figure(figsize=(20,5))

    

    plt.subplot(131)

    sns.kdeplot(data[cols],color='g',shade=True)

    plt.title(cols+' Distribution')



    plt.subplot(132)

    sns.kdeplot(data_scaled[cols],color='b',shade=True)

    plt.title(cols+' Distribution after scaling')

    

    plt.subplot(133)

    sns.boxplot(y = data_scaled[cols],color='pink')

    plt.title(cols+' Boxplot')

    

    plt.show()
data['duration_ms']      = data_scaled['duration_ms']+9

data['instrumentalness'] = data_scaled['instrumentalness']+1

data['liveness']         = data_scaled['liveness']+3

data['speechiness']      = data_scaled['speechiness']+3

data['loudness']         = data_scaled['loudness']+5
plt.figure(figsize=(12,6))

x = data.groupby("name")["popularity"].mean().sort_values(ascending=False).head(10)

ax = sns.barplot(x.index, x)

ax.set_title('Top Song with Popularity')

ax.set_ylabel('Popularity')

ax.set_xlabel('Songs')

plt.xticks(rotation = 90)
plt.figure(figsize=(12,6))

x = data.groupby("artists")["popularity"].sum().sort_values(ascending=False).head(10)

ax = sns.barplot(x.index, x)

ax.set_title('Top Artists with Popularity')

ax.set_ylabel('Popularity')

ax.set_xlabel('Artists')

plt.xticks(rotation = 90)