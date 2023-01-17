import numpy as np

import pandas as pd

import scipy as sp

import matplotlib.pyplot as plt

import re

import datetime

import seaborn as sns

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier

from sklearn import svm

from sklearn import preprocessing



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))
train=pd.read_csv("../input/train.csv")

test=pd.read_csv("../input/test.csv")
pd.set_option('display.max_colwidth',100)
train.head()
train.isnull().sum()

test.isnull().sum()
train['name'].fillna(" ")

train['desc'].fillna(" ")

test['desc'].fillna(" ")
col_date=['state_changed_at','created_at','launched_at','deadline']



for i in col_date:

    train[i]=train[i].apply(lambda x: datetime.datetime.fromtimestamp(int(x)).strftime("%Y-%m-%d %H:%M:%S"))

    

for i in col_date:

    test[i]=test[i].apply(lambda x: datetime.datetime.fromtimestamp(int(x)).strftime("%Y-%m-%d %H:%M:%S"))
sns.countplot(x='final_status',data=train)
sns.countplot(x='disable_communication',data=train, hue='final_status')
train['disable_communication'].value_counts()
goal_1=train['goal'][train['final_status']==1]

goal_0=train['goal'][train['final_status']==0]



print("Average Goal value when Funding is approved: ",goal_1.mean())

print("Average Goal value when Funding is not approved: ",goal_0.mean())
sns.factorplot(x='final_status', y='goal',data=train, hue='disable_communication',kind='bar')
sns.countplot(x='country',data=train,hue='final_status')
sns.countplot(x='currency',data=train,hue='final_status')
sns.factorplot(x='final_status',data=train,y='backers_count',kind='bar')
col_date=['state_changed_at','created_at','launched_at','deadline']



for i in col_date:

    train[i]=train[i].apply(lambda x: datetime.datetime.strptime(x,"%Y-%m-%d %H:%M:%S"))

    

for i in col_date:

    test[i]=test[i].apply(lambda x: datetime.datetime.strptime(x,"%Y-%m-%d %H:%M:%S"))
d1=[]

d2=[]

for i in range(len(train['deadline'])):

        d1.append((train['deadline'].iloc[i]-train['launched_at'].iloc[i]).total_seconds())

        

for i in range(len(test['deadline'])):

        d2.append((test['deadline'].iloc[i]-test['launched_at'].iloc[i]).total_seconds())

        

        

train['lunched_before_deadline']=d1

test['launched_before_deadline']=d2
d3=[]

d4=[]

for i in range(len(train['deadline'])):

        d3.append((train['deadline'].iloc[i]-train['created_at'].iloc[i]).total_seconds())

        

for i in range(len(test['deadline'])):

        d4.append((test['deadline'].iloc[i]-test['created_at'].iloc[i]).total_seconds())

        

        

train['created_before_deadline']=d3

test['created_before_deadline']=d4
d5=[]

d6=[]

for i in range(len(train['deadline'])):

        d5.append((train['state_changed_at'].iloc[i]-train['deadline'].iloc[i]).total_seconds())

        

for i in range(len(test['deadline'])):

        d6.append((test['state_changed_at'].iloc[i]-test['deadline'].iloc[i]).total_seconds())

        

        

train['state_changed_after_deadline']=d5

test['state_changed_after_deadline']=d6
train.head()