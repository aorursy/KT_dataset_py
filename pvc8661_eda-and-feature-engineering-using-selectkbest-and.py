# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))


# Any results you write to the current directory are saved as output.
# reading train data

#import first 10,000,000 rows of train and all test data
df_train = pd.read_csv('../input/train.csv', nrows=10000000) #Reading the dataset in a dataframe using Pandas
df_test = pd.read_csv('../input/test.csv')
display(df_train.head(5))
display('==========================================================================')
display(df_test.head(5))
display(df_train.shape)
display(df_train.describe())
print("Finding the count of null values in the police dataset")
pd.isnull(df_train).sum()
# Exploring APP ID 
app=df_train['app']
print("Most App : ", len(app.unique()))
print(app.value_counts().head(4))
app_name = app.value_counts().head(4)
# OS  Distribution 
os=df_train['os']
print("Most os : ", len(os.unique()))
print(os.value_counts().head(4))
os = os.value_counts().head(4)
plt.figure(figsize=(15,8))
sns.barplot(x=app_name.index,y=app_name.values,alpha=0.9)
plt.xticks(rotation='vertical')
plt.xlabel('APP ID ', fontsize=12)
plt.ylabel('Most APP ID USED FOR MARKETING', fontsize=12)
plt.title("Distribution of APP ID ", fontsize=16)
plt.show()
plt.figure(figsize=(8,10))
df_train.groupby([df_train['os'].head(10)]).size().sort_values(ascending=True).plot(kind='barh')
plt.title('Most OS ')
plt.ylabel('OS used ')
plt.xlabel('Distribution of OS Version used by various user')
plt.show()
channel=df_train['channel']
print("Most channel : ", len(app.unique()))
print(channel.value_counts().head(4))
channel_id = channel.value_counts().head(4)
plt.figure(figsize=(15,8))
sns.barplot(x=channel_id.index,y=channel_id.values,alpha=0.9)
plt.xticks(rotation='vertical')
plt.xlabel('Channel Id ', fontsize=12)
plt.ylabel('Most Channel Id USED', fontsize=12)
plt.title("Distribution of Channel Id ", fontsize=16)
plt.show()
df_num = df_train.select_dtypes(include = ['float64', 'int64'])
df_num.head(4)
print('Numerical distribution of the dataset')
plt.figure(figsize=(9, 8))
df_num.hist(figsize=(16, 20), bins=50, xlabelsize=8, ylabelsize=8); # ; avoid having the matplotlib verbose informations
plt.show()
df_train['year']    = pd.to_datetime(df_train.click_time).dt.year.astype('uint8')
df_train['hour']    = pd.to_datetime(df_train.click_time).dt.hour.astype('uint8')
df_train['day']     = pd.to_datetime(df_train.click_time).dt.day.astype('uint8')
df_train['wday']    = pd.to_datetime(df_train.click_time).dt.dayofweek.astype('uint8')
print(df_train.dtypes) 
plt.figure(figsize=(8,10))
df_train.groupby([df_train['day']]).size().sort_values(ascending=True).plot(kind='barh')
plt.title('Most day ')
plt.ylabel('day used ')
plt.xlabel('Distribution of days used by various user')
plt.show()
# Understanding the correlation between the features and the labels
plt.figure(figsize=(15, 5))
plt.hist(app[df_train['is_attributed'] == 0], bins=20, normed=True, label='App Downloaded')
plt.hist(app[df_train['is_attributed'] == 1], bins=20, normed=True, alpha=0.7, label='App Not downloaded')
plt.legend()
plt.title('Label distribution', fontsize=15)
plt.xlabel('App ', fontsize=15)
plt.show()

hour=df_train['hour']
plt.figure(figsize=(15, 5))
plt.hist(hour[df_train['is_attributed'] == 0], bins=20, normed=True, label='App Downloaded')
plt.hist(hour[df_train['is_attributed'] == 1], bins=20, normed=True, alpha=0.7, label='App Not downloaded')
plt.legend()
plt.title('Label distribution', fontsize=15)
plt.xlabel('Hour ', fontsize=15)
plt.show()
wday=df_train['wday']
plt.figure(figsize=(15, 5))
plt.hist(wday[df_train['is_attributed'] == 0], bins=20, normed=True, label='App Downloaded')
plt.hist(wday[df_train['is_attributed'] == 1], bins=20, normed=True, alpha=0.7, label='App Not downloaded')
plt.legend()
plt.title('Label distribution', fontsize=15)
plt.xlabel('wday ', fontsize=15)
plt.show()
device=df_train['device']
plt.figure(figsize=(15, 5))
plt.hist(device[df_train['is_attributed'] == 0], bins=20, normed=True, label='App Downloaded')
plt.hist(device[df_train['is_attributed'] == 1], bins=20, normed=True, alpha=0.7, label='App Not downloaded')
plt.legend()
plt.title('Label distribution', fontsize=15)
plt.xlabel('Device', fontsize=15)
plt.show()
channel=df_train['channel']
plt.figure(figsize=(15, 5))
plt.hist(channel[df_train['is_attributed'] == 0], bins=20, normed=True, label='App Downloaded')
plt.hist(channel[df_train['is_attributed'] == 1], bins=20, normed=True, alpha=0.7, label='App Not downloaded')
plt.legend()
plt.title('Label distribution', fontsize=15)
plt.xlabel('channel', fontsize=15)
plt.show()
os=df_train['os']
plt.figure(figsize=(15, 5))
plt.hist(os[df_train['is_attributed'] == 0], bins=20, normed=True, label='App Downloaded')
plt.hist(os[df_train['is_attributed'] == 1], bins=20, normed=True, alpha=0.7, label='App Not downloaded')
plt.legend()
plt.title('Label distribution', fontsize=15)
plt.xlabel('os', fontsize=15)
plt.show()
display(df_train.shape)
display(df_train.columns)
cols = list(df_train.columns.values) #Make a list of all of the columns in the df
cols.pop(cols.index('is_attributed')) #Remove b from list
df_train = df_train[cols+['is_attributed']]
display(df_train.columns)
display(df_train.shape)
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

X=df_train.loc[:, : 'wday']
X.drop('click_time',axis=1,inplace=True)
X.drop('attributed_time',axis=1,inplace=True)
print(X.columns)
Y =df_train['is_attributed']
# feature extraction
test = SelectKBest(score_func=chi2, k=4)
fit = test.fit(X, Y)
# summarize scores
np.set_printoptions(precision=3)
scores=fit.scores_
print(scores)
print('====================================================================')
print(np.sort(scores))
# The best features which are describing the data are :
# IP , APP , Channel , OS , Device , hour , wday , day , year 
features = fit.transform(X)
print(features.shape)
print(features[0:5,:])
from sklearn.ensemble import ExtraTreesClassifier
model = ExtraTreesClassifier()
model.fit(X, Y)
print(model.feature_importances_)