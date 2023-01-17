# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns





# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
data=pd.read_csv("/kaggle/input/google-play-store-apps/googleplaystore.csv")#importing dataset.
data
data.head()#we get first five row of dataset.
data.tail()#we get last five row.
data.columns#we learn columns in this dataset.
data.shape#we learn that count of columns and rows in this dataset.
data.info()# we can get information about dataset and its fields through this.
print(data['Category'].value_counts(dropna=False))#we leaarn that frequency of categories.
data.describe()#we reach some statistical information.
data.boxplot(column="Rating", by="Category", figsize=(60,20))

plt.show()#we can see outliers about rating and category fields.
data_new=data.head()

data_new
melted=pd.melt(frame=data_new, id_vars="App", value_vars=["Category", "Rating"])

melted#melting process with category and rating fields.
melted.pivot(index="App", columns="variable", values="value")#reverse of previous process, as you see index does not change.
data1=data.head()

data2=data.tail()#creating two dataframes

conc_data_row=pd.concat([data1,data2], axis=0, ignore_index=True)

conc_data_row#and process of concat. it is vertical.
data1=data["Category"].head()

data2=data["Rating"].head()

conc_data_col=pd.concat([data1,data2],axis=1)

conc_data_col#this time we selected two fields before concat process. it is horizontal.
data.dtypes
data["Category"]=data["Category"].astype("category")#lets convert the Category field from object to category.
data.dtypes#lets check
data["Type"].value_counts(dropna=False)#we can see nan values. There is only one nan value.
data1=data

data["Type"].dropna(inplace=True)#remove nan values.
assert 1==1#if this code doesn't return something, it works(the code that is above)
assert data['Type'].notnull().all()#this returns nothing because there is no nan value.
data['Type'].fillna("empty", inplace=True)#to fill empty
assert data['Type'].notnull().all()#again it returns nothing because there is no nan value.