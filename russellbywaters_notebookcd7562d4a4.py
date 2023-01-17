#load packages

import pandas as pd

import numpy as np

import csv as csv

from sklearn import ensemble

from sklearn import tree
#find working directory

import os

os.getcwd()
#change to files directory

os.chdir('/kaggle/input')

os.getcwd()
train_df = pd.read_csv('train.csv', header=0)
whos
#get rows and shapes

train_df.shape
train_df.info()
#stat summary

train_df.describe().transpose()
#inspect first rows

train_df.head(5)
train_df['Gender']=train_df['Sex'].map({'female':0,'male':1}).astype(int)
median_age=train_df['Age'].dropna().median()

if len(train_df.Age[train_df.Age.isnull()])>0:

    train_df.loc[(train_df.Age.isnull()),'Age']=median_age