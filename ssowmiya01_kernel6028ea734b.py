# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from math import pi

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import LabelEncoder

from sklearn.preprocessing import MinMaxScaler

from numpy.linalg import eig

import random

import matplotlib.pyplot as plt

import seaborn as sns





# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
data=pd.read_csv("/kaggle/input/adult-census-income/adult.csv")
data.shape
cat=[]

for col in list(data.columns.values):

    if data[col].dtypes=='object':

        cat.append(col)

remain=[]

for col in data.columns.values: 

    if col not in cat:

        remain.append(col)

encod=LabelEncoder()

en_df=data[cat].apply(encod.fit_transform)

df=pd.DataFrame(en_df.join(data[remain]))
m=np.mean(df.T,axis=1)

center=df-m

covar=np.cov(center.T)

evalue,evector=eig(covar)

pca=evector.T.dot(center.T)

pca=pca.T

print(pca)
scalar=MinMaxScaler()

nordata=pd.DataFrame(scalar.fit_transform(df),columns=df.columns)

y=nordata['income']

df=nordata.copy()

df.drop('income',axis=1,inplace=True)
x_train,x_test,y_train,y_test=train_test_split(df,y,test_size=0.3,random_state=2,shuffle=True)

print(x_train.shape,'  ',y_train.shape)

print(x_test.shape,'  ',y_test.shape)

train=x_train.join(y_train)

x_test=x_test.to_numpy()

train=train.to_numpy()

y_test=y_test.to_numpy()
samp1=nordata.sample(n=100)

sam_x=samp1.copy()

sam_y=samp1['income']

sam_x.drop('income',axis=1,inplace=True)
sns.scatterplot(x=df['workclass'],y=y)
sns.set_style("darkgrid")

sns.lineplot(x=sam_x['education'],y=y)
fig, ax = plt.subplots(figsize=(12,10)) 

samp=df.sample(n=10)

samp=samp.corr()

sns.heatmap(data=samp,annot=True,linecolor='black',linewidths=.5, ax=ax)
#usinfg predefined library

x_train=train[:,0:14]

y_train=train[:,14]

from sklearn.naive_bayes import GaussianNB

model = GaussianNB()

model.fit(x_train,y_train)

predicted= model.predict(x_test)
def split_class(data):

    d = dict()

    for i in range(len(data)):

        vector = data[i]

        class_value = vector[-1]

        if (class_value not in d):

            d[class_value] = list()

        d[class_value].append(vector)

    return d
def mean(numbers):

    return sum(numbers)/float(len(numbers)) 

def std(numbers):

    avg = mean(numbers)

    variance = sum([(x-avg)**2 for x in numbers]) / float(len(numbers)-1)

    return sqrt(variance)
def gaussian(x,mean,std):

     return np.exp((-(x-mean)**2)/2*std**2)/np.sqrt(2*pi*std)

def summarize_data(data):

    info=[(mean(col),std(col),len(col))for col in zip(*data)]

    del info[-1]

    return info

def summarize_class(data):

    d=split_class(data)

    summ=dict()

    for val,rows in d.items():

        summ[val]=summarize_data(rows)

    return summ
def probability(summ,data):

    prob={}

    tot_rows = sum([summ[label][0][2] for label in summ])

    for classval,classsum in summ.items():

        prob[classval]=summ[classval][0][2]/float(tot_rows)

        for i in range(len(classsum)):

            mean,std,_=classsum[i]

            prob[classval]*=gaussian(data[i],mean,std)

    return prob
def predict(data,row):

    prob=probability(data,row)

    y,pre_prob=None,-1

    for val,pro in prob.items():

        if y is None or pre_prob<pro:

            y=val

            pre_prob=pro

    return y
def naive_bayes(train,test):

    model=summarize_class(train)

    pred=[]

    for i in test:

        val=predict(model,i)

        pred.append(val)

    return pred
prediction=naive_bayes(train,x_test)#final result 