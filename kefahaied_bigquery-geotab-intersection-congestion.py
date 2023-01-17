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


import seaborn as sns

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.preprocessing import LabelEncoder

from sklearn import preprocessing

from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import train_test_split

from sklearn.model_selection import KFold

from sklearn.preprocessing import StandardScaler

# import keras

from bayes_opt import BayesianOptimization

import lightgbm as lgb

import os, sys

!pip install tensorflow --upgrade
import tensorflow as tf
print(tf.__version__)
from tensorflow import feature_column

from tensorflow.keras import layers
# Load data 

train = pd.read_csv('../input/bigquery-geotab-intersection-congestion/train.csv')

test = pd.read_csv('../input/bigquery-geotab-intersection-congestion/test.csv')

submission = pd.read_csv('../input/bigquery-geotab-intersection-congestion/sample_submission.csv')
train
train.columns
test.columns
test
train.describe()
train.columns
train.dtypes
train.info()
train.isnull().sum()

train.dropna(axis=0, inplace=True)
train.isnull().sum()
def missing_values(train):

    df = pd.DataFrame(train.isnull().sum()).reset_index()

    df.columns = ['Feature', 'Frequency']

    df['Percentage'] = (df['Frequency']/train.shape[0])*100

    df['Percentage'] = df['Percentage'].astype(str) + '%'

    df.sort_values('Percentage', inplace = True, ascending = False)

    return df



missing_values(train).head()
#Finding the numerical columns 

num_cols = train._get_numeric_data().columns

print("Numerical Columns")

print(num_cols)



# Get list of categorical variables

s = (train.dtypes == 'object')

object_cols = list(s[s].index)



print("Categorical variables:")

print(object_cols)

for object_col in object_cols:

    print("---------------------------")

    print(train[object_col].unique())    
#Submission data

#the first number being the RowId and the second being the metric id (of the TargetId)

submission

for i in ['TotalTimeStopped_p20', 'TotalTimeStopped_p50', 'TotalTimeStopped_p80', 'DistanceToFirstStop_p20', 

          'DistanceToFirstStop_p50', 'DistanceToFirstStop_p80']:

    plt.figure(figsize = (12, 8))

    plt.scatter(train.index, train[i])

    plt.title('{} distribution'.format(i))
def tv_ratio(train, column):

    df = train[train[column]==0]

    ratio = df.shape[0] / train.shape[0]

    return ratio



target_variables = ['TotalTimeStopped_p20', 'TotalTimeStopped_p50', 'TotalTimeStopped_p80', 

                    'DistanceToFirstStop_p20', 'DistanceToFirstStop_p50', 'DistanceToFirstStop_p80']



for i in target_variables:

    print('{} have a 0 ratio of: '.format(i), tv_ratio(train, i))
fig, ax = plt.subplots(nrows=2, ncols=2)

sns.set_style("whitegrid")



train[train['City']=='Atlanta'].groupby('Hour')['TotalTimeStopped_p80'].mean().plot(

    ax=ax[0,0],title="Atlanda's Total Stoppage Time in Hours", color='r', figsize=(18,15))



train[train['City']=='Boston'].groupby('Hour')['TotalTimeStopped_p80'].mean().plot(

    ax=ax[0,1],title="Boston's Total Stoppage Time in Hours", color='r', figsize=(18,15))





train[train['City']=='Chicago'].groupby('Hour')['TotalTimeStopped_p80'].mean().plot(

    ax=ax[1,0],title="Chicago's Total Stoppage Time in Hours", color='r', figsize=(18,15))





train[train['City']=='Philadelphia'].groupby('Hour')['TotalTimeStopped_p80'].mean().plot(

    ax=ax[1,1],title="Philadelphia's Total Stoppage Time in Hours", color='r', figsize=(18,15))



plt.show()
def plot_dist(train, test, column, type = 'kde', together = True):

    if type == 'kde':

        if together == False:

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (12,8))

            sns.kdeplot(train[column], ax = ax1, color = 'blue', shade=True)

            ax1.set_title('{} distribution of the train set'.format(column))

            sns.kdeplot(test[column], ax = ax2, color = 'red', shade=True)

            ax2.set_title('{} distribution of the test set'.format(column))

            plt.show()

        else:

            fig , ax = plt.subplots(1, 1, figsize = (12,8))

            sns.kdeplot(train[column], ax = ax, color = 'blue', shade=True, label = 'Train {}'.format(column))

            sns.kdeplot(test[column], ax = ax, color = 'red', shade=True, label = 'Test {}'.format(column))

            ax.set_title('{} Distribution'.format(column))

            plt.show()

    else:

        if together == False:

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (12,8))

            sns.distplot(train[column], ax = ax1, color = 'blue', kde = False)

            ax1.set_title('{} distribution of the train set'.format(column))

            sns.distplot(test[column], ax = ax2, color = 'red', kde = False)

            ax2.set_title('{} distribution of the test set'.format(column))

            plt.show()

        else:

            fig , ax = plt.subplots(1, 1, figsize = (12,8))

            sns.distplot(train[column], ax = ax, color = 'blue', kde = False)

            sns.distplot(test[column], ax = ax, color = 'red', kde = False)

            plt.show()

    

plot_dist(train, test, 'Latitude', type = 'kde', together = True)

plot_dist(train, test, 'Latitude', type = 'other', together = False)

def get_frec(df, column):

    df1 = pd.DataFrame(df[column].value_counts(normalize = True)).reset_index()

    df1.columns = [column, 'Percentage']

    df1.sort_values(column, inplace = True, ascending = True)

    return df1





def plot_frec(train, test, column):

    df = get_frec(train, column)

    df1 = get_frec(test, column)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (12,8))

    sns.barplot(df[column], df['Percentage'], ax = ax1, color = 'blue')

    ax1.set_title('{} percentages for the train set'.format(column))

    sns.barplot(df1[column], df1['Percentage'], ax = ax2, color = 'red')

    ax2.set_title('{} percentages for the test set'.format(column))

    

plot_frec(train, test, 'Month')
train["same_street_exact"] = (train["EntryStreetName"] ==  train["ExitStreetName"]).astype(int)

test["same_street_exact"] = (test["EntryStreetName"] ==  test["ExitStreetName"]).astype(int)
le = preprocessing.LabelEncoder()

train["Intersection"] = train["IntersectionId"].astype(str) + train["City"]

test["Intersection"] = test["IntersectionId"].astype(str) + test["City"]



print(train["Intersection"].sample(6).values)
pd.concat([train["Intersection"],test["Intersection"]],axis=0).drop_duplicates().values
le.fit(pd.concat([train["Intersection"],test["Intersection"]]).drop_duplicates().values)

train["Intersection"] = le.transform(train["Intersection"])

test["Intersection"] = le.transform(test["Intersection"])
pd.get_dummies(train["City"],dummy_na=False, drop_first=False).head()

train = pd.concat([train,pd.get_dummies(train["City"],dummy_na=False, drop_first=False)],axis=1).drop(["City"],axis=1)

test = pd.concat([test,pd.get_dummies(test["City"],dummy_na=False, drop_first=False)],axis=1).drop(["City"],axis=1)
train.shape,test.shape

test.head()

train.columns

FEAT_COLS = ["IntersectionId",

             'Intersection',

            'same_street_exact',

           "Hour","Weekend","Month",

          'Latitude', 'Longitude',

          'Atlanta', 'Boston', 'Chicago',

       'Philadelphia']
train.head()

train.columns

X = train[FEAT_COLS]

y1 = train["TotalTimeStopped_p20"]

y2 = train["TotalTimeStopped_p50"]

y3 = train["TotalTimeStopped_p80"]

y4 = train["DistanceToFirstStop_p20"]

y5 = train["DistanceToFirstStop_p50"]

y6 = train["DistanceToFirstStop_p80"]
y = train[['TotalTimeStopped_p20', 'TotalTimeStopped_p50', 'TotalTimeStopped_p80',

        'DistanceToFirstStop_p20', 'DistanceToFirstStop_p50', 'DistanceToFirstStop_p80']]
testX = test[FEAT_COLS]

lr = RandomForestRegressor(n_estimators=100,min_samples_split=3)
lr.fit(X,y1)

pred1 = lr.predict(testX)

lr.fit(X,y2)

pred2 = lr.predict(testX)

lr.fit(X,y3)

pred3 = lr.predict(testX)

lr.fit(X,y4)

pred4 = lr.predict(testX)

lr.fit(X,y5)

pred5 = lr.predict(testX)

lr.fit(X,y6)

pred6 = lr.predict(testX)





# Appending all predictions

all_preds = []

for i in range(len(pred1)):

    for j in [pred1,pred2,pred3,pred4,pred5,pred6]:

        all_preds.append(j[i])   

        

sub  = pd.read_csv("../input/bigquery-geotab-intersection-congestion/sample_submission.csv")

sub["Target"] = all_preds

sub.to_csv("benchmark_beat_rfr_multimodels.csv",index = False)



print(len(all_preds))
lr.fit(X,y)

print("fitted")



all_preds = lr.predict(testX)
## convert list of lists to format required for submissions

print(all_preds[0])



s = pd.Series(list(all_preds) )

all_preds = pd.Series.explode(s)



print(len(all_preds))

print(all_preds[0])
sub  = pd.read_csv("../input/bigquery-geotab-intersection-congestion/sample_submission.csv")

print(sub.shape)

sub.head()
sub["Target"] = all_preds.values

sub.sample(5)