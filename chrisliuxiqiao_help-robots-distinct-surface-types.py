import numpy as np

import pandas as pd

import os 

import time 

from sklearn import preprocessing 

from sklearn.metrics import accuracy_score

import matplotlib.pyplot as plt

import seaborn as sns

from scipy import stats

from scipy.stats import norm 

from sklearn.preprocessing import StandardScaler

from matplotlib import rcParams 

%matplotlib inline

le=preprocessing.LabelEncoder()

from numba import jit 

import itertools 

from seaborn import countplot, lineplot, barplot

from sklearn.feature_selection import SelectKBest

from sklearn.feature_selection import chi2

from scipy.stats import randint as sp_randint

from sklearn.model_selection import KFold, StratifiedKFold

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score, confusion_matrix

from sklearn.model_selection import LeaveOneGroupOut, GroupKFold, GridSearchCV, RandomizedSearchCV

import matplotlib.style as style

style.use('ggplot')
x_train=pd.read_csv('../input/X_train.csv')
x_test=pd.read_csv('../input/X_test.csv')
target=pd.read_csv('../input/y_train.csv')
x_train.head()
x_test.head()
target.head()
len(x_train.measurement_number.value_counts())
x_train.describe()
x_test.describe()
target.describe()
#Check missing data at training data

missing_num=x_train.isnull().sum()

missing_data=pd.concat([missing_num],axis=1,keys=['Missing_Num'])

missing_data
#Check missing data in testing data

missing_num=x_test.isnull().sum()

missing_data=pd.concat([missing_num],axis=1,keys=['Missing_Data'])

missing_data
target['group_id'].nunique()
sns.set(style='dark')

sns.countplot(y='surface',data=target,order=target['surface'].value_counts().index)

plt.show()
fig, ax=plt.subplots(1,1,figsize=(26,8))

tmp=pd.DataFrame(target.groupby(['group_id','surface'])['series_id'].count().reset_index())

m=tmp.pivot(index='surface',columns='group_id',values='series_id')

s=sns.heatmap(m,linewidths=.1,linecolor='black',annot=True,cmap='YlGnBu')

s.set_title('Number of surface category per group_id',size=25)

plt.show()
plt.figure(figsize=(50,15))

sns.set(style='darkgrid')

sns.countplot(x='group_id',data=target,order=target['group_id'].value_counts().index)

plt.show()
serie1=x_train.head(128)

plt.figure(figsize=(30,20))

for i, col in enumerate(serie1.columns[3:]):

    plt.subplot(3,4,i+1)

    plt.plot(serie1[col])

    plt.title(col)
series_dict={}

for series_id in (x_train['series_id'].unique()):

    series_dict[series_id]=x_train[x_train['series_id']==series_id]

def seriesplot(series_id):

    style.use('ggplot')

    plt.figure(figsize=(30,20))

    for i, col in enumerate(series_dict[series_id].columns[3:]):

        plt.subplot(3,4,i+1)

        plt.plot(series_dict[series_id][col])

        plt.title(col)
seriesplot(2)
fig,ax=plt.subplots(figsize=(8,8))

sns.heatmap(x_train.iloc[:,3:].corr(),annot=True,linewidths=0.5,cmap='YlGnBu',fmt='.2f',ax=ax)
#Strong Correlation between:

#    angular_velocity_Z and angular_velocity_Y

#    orientation_X and orientation_Y

#    orientation_Y and orientation_Z
fig,ax=plt.subplots(figsize=(8,8))

sns.heatmap(x_test.iloc[:,3:].corr(),annot=True,linewidths=.5,cmap='YlGnBu',fmt='.2f',ax=ax)
#Beacuse sizes of the training data and testing data are different, 

#so it won't be a problem when the correlation coefficients are different.
def feature_distribution(df1,df2,label1,label2,features,a=2,b=5):

    i=0

    sns.set_style('whitegrid')

    plt.figure()

    fig,ax=plt.subplots(a,b,figsize=(17,9))

    

    for feature in features:

        i+=1

        plt.subplot(a,b,i)

        sns.kdeplot(df1[feature],bw=.5,label=label1)

        sns.kdeplot(df2[feature],bw=.5,label=label2)

        plt.xlabel(feature,fontsize=9)

        locs,labels=plt.xticks()

        plt.tick_params(axis='x',which='major',labelsize=8)

        plt.tick_params(axis='y',which='major',labelsize=8)

    plt.show()



features=x_train.columns[3:]

feature_distribution(x_train,x_test,'train','test',features)
def class_distribution(classes,tt,features,a=5,b=2):

    i=0

    sns.set_style('whitegrid')

    plt.figure()

    fig,ax=plt.subplots(a,b,figsize=(16,24))

    

    for feature in features:

        i+=1

        plt.subplot(a,b,i)

        for each_class in classes:

            ttc=tt[tt['surface']==each_class]

            sns.kdeplot(ttc[feature],bw=.5,label=each_class)

        plt.xlabel(feature,fontsize=9)

        locs,labels=plt.xticks()

        plt.tick_params(axis='x',which='major',labelsize=8)

        plt.tick_params(axis='y',which='major',labelsize=8)

    plt.show()

    
classes=(target['surface'].value_counts()).index

feature=x_train.columns.values[3:]

concatenation=x_train.merge(target,on='series_id',how='left')

class_distribution(classes,concatenation,features)
plt.figure(figsize=(26,16))

for i,col in enumerate(concatenation.columns[3:13]):

    ax=plt.subplot(3,4,i+1)

    ax=plt.title(col)

    for surface in classes:

        surface_feature=concatenation[concatenation['surface']==surface]

        sns.kdeplot(surface_feature[col],label=surface)
#Histogram 

plt.figure(figsize=(26,16))

for i,col in enumerate(x_train.columns[3:]):

    ax=plt.subplot(3,4,i+1)

    sns.distplot(x_train[col],bins=100,label='train')

    sns.distplot(x_test[col],bins=100,label='test')

    ax.legend()
#Orientation x,y,z are not normally distributed or bell shaped distributed

#Angular velocity and linear accelaration are normally distributed



    