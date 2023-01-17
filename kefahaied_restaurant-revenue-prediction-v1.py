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



# Any results you write to the current directory are saved as output.
data =  pd.read_csv('../input/restaurant-revenue-prediction/train.csv')

test_data = pd.read_csv('../input/restaurant-revenue-prediction/test.csv')
data.describe()
data.head()
data.columns
data.dtypes
data.info()
data['Type'].unique()
data['City Group'].unique()
data['City'].unique()
import matplotlib.pyplot as plt

import seaborn as sns



plt.figure(figsize=(14,6))

plt.subplot(1,2,1)

sns.boxplot(data.revenue)

plt.subplot(1,2,2)

sns.distplot(data.revenue, bins=20, kde=False)

plt.show()
#City distribution

data["City"].value_counts().plot(kind='bar')
data["Type"].value_counts().plot(kind='bar')
data["City Group"].value_counts().plot(kind='bar')
# Crrelation between revenue and feature (p)s

def numFeaturePlot():

    features=(data.loc[:,'P1':'P37']).columns.tolist()

    plt.figure(figsize=(35,18))

    j=1

    while j<len(features):

        col=features[j-1]

        plt.subplot(6,6,j)

        sorted_grp = data.groupby(col)["revenue"].sum().sort_values(ascending=False).reset_index()

        x_val = sorted_grp.index

        y_val = sorted_grp['revenue'].values

        plt.scatter(x_val, y_val)

        plt.xticks(rotation=60)

        plt.xlabel(col, fontsize=20)

        plt.ylabel('Revenue', fontsize=20)

        j+=1    

    plt.tight_layout()

    plt.show()

numFeaturePlot()
# This method helps in understanding the correlation between the different features and the Revenue.

def featureCatPlot(col):

    

    plt.figure(figsize=(15,6))

    i=1

    if not data[col].dtype.name=='int64' and not data[col].dtype.name=='float64':

        plt.subplot(1,2,i)

        sns.boxplot(x=col,y='revenue',data=data)

        plt.xticks(rotation=60)

        plt.ylabel('Revenue')

        i+=1 

        plt.subplot(1,2,i)

        mean=data.groupby(col)['revenue'].mean()

        level=mean.sort_values().index.tolist()

        data[col]=data[col].astype('category')

        data[col].cat.reorder_categories(level,inplace=True)

        data[col].value_counts().plot()

        plt.xticks(rotation=60)

        plt.xlabel(col)

        plt.ylabel('Counts')       

        plt.show()
featureCatPlot('City Group')

# Splitting 01/31/2018 as 01, 31, 2018

train_date=data['Open Date'].str.split('/', n = 2, expand = True)

data['month']=train_date[0]

data['days']=train_date[1]

data['year']=train_date[2]



test_date=test_data['Open Date'].str.split('/', n = 2, expand = True)

test_data['month']=test_date[0]

test_data['days']=test_date[1]

test_data['year']=test_date[2]

data['month']

featureCatPlot('month')
data.sort_values('revenue', ascending=False)[:20]
top_6= data.sort_values('revenue', ascending=False)[:20]

plt.figure(figsize=(13,12))

plt.title("The top 6 resturants")

sns.barplot(x=top_6['City'], y=top_6['revenue'])



best_month= data.sort_values('revenue', ascending=False)[:20]

plt.figure(figsize=(13,12))



sns.barplot(x=best_month['month'], y=best_month['revenue'])

plt.xticks(rotation=60)
best_type= data.sort_values('revenue', ascending=False)



plt.figure(figsize=(13,12))



sns.barplot(x=best_type['Type'], y=best_type['revenue'])
data.isnull().sum()
# Get list of categorical variables

s = (data.dtypes == 'object')

object_cols = list(s[s].index)

print("Categorical variables:")

print(object_cols)
from sklearn import preprocessing

from sklearn.model_selection import train_test_split
data = data.drop('Id', axis=1)

test_data = test_data.drop('Id', axis=1)

y= data.revenue

X = data.drop(['revenue'], axis=1)
