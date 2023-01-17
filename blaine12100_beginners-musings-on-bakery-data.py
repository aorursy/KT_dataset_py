# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
%matplotlib inline

# from fastai.imports import *
# from fastai.structured import *
import matplotlib.pyplot as plt
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.

bakery_data=pd.read_csv("../input/BreadBasket_DMS.csv")

bakery_data['Date Time']=bakery_data['Date']+" "+bakery_data['Time']

bakery_data=bakery_data.drop(['Date','Time'],axis=1)

print("Top 5 rows of data")

print(bakery_data.head(25))

print("Bottom 5 rows of data")

print(bakery_data.tail(5))

print(len(bakery_data))
bakery_data.dtypes
#Can Probably ease the memory requirement and speed up processing by formatting datatypes properly.

# bakery_data['Date']=pd.to_datetime(bakery_data['Date'])
# bakery_data['Time']=pd.to_datetime(bakery_data['Time'])
bakery_data['Date Time']=pd.to_datetime(bakery_data['Date Time'])
bakery_data.dtypes

#Types are Proper.Can move Forward.Object dtype is string.
#Frequency Distribution of Items.

bakery_data['Item'].unique()

len(bakery_data['Item'].unique())

#95 Unique Items.Must me a big bakery.

bakery_data['Item'].value_counts()

# Most Popular Ordered item is Coffee which is no surprise.I am surprised though that hot chocolate is 
# ordered very less though

# Mandatory Nan Check

bakery_data.isnull().sum()

# No empty values found so dataset is clean and processed properly.
# Finding Baskets based on transaction Numbers

baskets=bakery_data.groupby(["Transaction","Date Time","Item"]).groups

# So we have around 19640 Baskets.Let's See if we can find out exactly what these baskets comprise of.

baskets

