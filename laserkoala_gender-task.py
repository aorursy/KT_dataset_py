# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

color = sns.color_palette()



%matplotlib inline



pd.options.mode.chained_assignment = None



import os

print(os.listdir("../input"))
train = pd.read_csv("../input/train.csv")

test = pd.read_csv("../input/test.csv")

users = pd.read_csv("../input/users.csv")
train['OPERATION_TIME'] = pd.to_datetime(train['OPERATION_TIME'])
user_item_gender = pd.merge(train[['USER_ID','ITEM_CODE']], users[['USER_ID','GENDER']], how = "inner", on='USER_ID')
item_gender =  user_item_gender.drop(columns=['USER_ID'])

item_gender = item_gender.fillna(2)

item_gender['PERCENT'] = 0

item_gender