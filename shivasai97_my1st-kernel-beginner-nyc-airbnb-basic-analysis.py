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
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns
#now lets read/load the dataset into an variable using pandas

data = pd.read_csv("/kaggle/input/new-york-city-airbnb-open-data/AB_NYC_2019.csv")
 #this function is used to display the first 5 rows from the data

data.head()
#used to display the length of the whole dataset

len(data)
#displays the shape of our data set

data.shape
#this function is used to get the basic information about the dataset  

data.info()
#this gives the overall discription about the dataset such as mean,std,count....

data.describe()
#this function is used to display the correlation between the variables in the dataset

data.corr()
#lets check the datatypes present in the dataset

data.dtypes
#it helps us to know if we have any missing values in our dataset

data.isnull().sum()
#lets use drop function to drop the columns from the dataset

data.drop(["id","host_name","last_review","reviews_per_month"],axis = 1, inplace=True)
#now let's again check the shape of our data to ensure about the number of columns present after performing the drop function

data.shape
#shows the name of the columns present in the dataset

data.columns
#this unique() function helps us to find out the unique values present in the dataset

data.neighbourhood_group.unique()
data.neighbourhood.unique()
data.room_type.unique()
#we do this with the help of value_counts() function

data.neighbourhood_group.value_counts()
data.neighbourhood_group.value_counts()
data.neighbourhood.value_counts().head()
data.room_type.value_counts()
sns.countplot(x= 'neighbourhood_group',data = data)

plt.title("Popular neighbourhood_group ")

plt.show()
plt.figure(figsize=(8,5))

sns.countplot(x='room_type',data = data)

plt.title("Most occupied room_type")

plt.show()
plt.figure(figsize=(10,6))

sns.countplot(x = 'room_type',hue = "neighbourhood_group",data = data)

plt.title("Room types occupied by the neighbourhood_group")

plt.show()
ng = data[data.price <500]

plt.figure(figsize=(12,7))

sns.boxplot(y="price",x ='neighbourhood_group' ,data = ng)

plt.title("neighbourhood_group price distribution < 500")

plt.show()
plt.figure(figsize=(10,10))

sns.scatterplot(x='longitude',y='latitude',hue = "neighbourhood_group",data = data,palette = 'hls')

plt.show()
plt.figure(figsize=(9,7))

sns.scatterplot(x='longitude',y='latitude',hue = 'availability_365',data = data, palette='coolwarm')

plt.show()