# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns #a virtualization tool



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
data = pd.read_csv("../input/BlackFriday.csv")
#top 5 rows from the dataset

data.head()
#after loading our data let's see some information about our data

data.info()
#check if dataset has any NaN value

data.isnull().sum()
#we can fill our NaN values.

data.fillna(data['Product_Category_1'].dropna().median(), inplace = True)

data.fillna(data['Product_Category_2'].dropna().median(), inplace = True)
data.isnull().sum()

#now our dataframe has 0 null values. 
#we can drop User_ID, Product_ID from the dataset. 

data = data.drop(["User_ID", "Product_ID"], axis = 1)
f,ax = plt.subplots(figsize = (18, 18))

sns.heatmap(data.corr(), annot = True, linewidths = 1, fmt = ".1f", ax = ax)

plt.show()
#we can see our data's count, max, min values as well as lower, 50 and upper percentiles with the help of describe function

data.describe()

def bargraph(xvalue, yvalue, data):

    sns.barplot(x = xvalue, y = yvalue, data = data)
bargraph("Gender", "Purchase", data)

#males have bought more than female in black friday
bargraph("Marital_Status", "Purchase", data)

#married and single people have bought same quantity of items.
bargraph("City_Category", "Purchase", data)
fig1, ax1 = plt.subplots(figsize=(12,7))

sns.set(style="darkgrid")

sns.countplot(data['Age'],hue=data['Gender'])

plt.show()
#remove + sign from Stay_In_Current_City colums

data.Stay_In_Current_City_Years = data.Stay_In_Current_City_Years.str.replace('+', '')

#remove + sign from Age

data.Age = data.Age.str.replace('+', '')

#we can use & symbol for logical and operation

data[(data['Purchase'] > 20000) & (data['Gender'] == 'F')]
#or we can use logical_and from numpy library

data[np.logical_and(data['Purchase'] > 20000, data['Gender'] == 'M')]
data.loc[:50, ['Product_Category_1', 'Product_Category_2', 'Product_Category_3', 'Gender']]
gender_mapping = {'M' : 0, 'F' : 1}

data['Gender'] = data['Gender'].map(gender_mapping)

data.head(10)

#frequency of Product 1

print(data['Product_Category_1'].value_counts(dropna = False)) #if there are NaN values that also be counted
#frequency of Product 2

print(data['Product_Category_2'].value_counts(dropna = False))

#frequency of Product 3

print(data['Product_Category_3'].value_counts(dropna = False))
data.boxplot(column = 'Purchase', by='Product_Category_1')
data.boxplot(column = 'Purchase', by = 'Product_Category_2')
data.boxplot(column = 'Purchase', by = 'Product_Category_3')
#categorize city 

city_mapping = {'A': 0, 'B': 1, 'C': 2}

data['City_Category'] = data['City_Category'].map(city_mapping)

data.head(10)
#list comprehension

data['Age'] = [0 if i == "0-17" else 1 if i == "18-25" else 2 if i == "26-35" else 3 if i == "36-45" else 4 if i == "46-50" else 5 if i == "51-55" else 6 for i in data['Age']]

data.head(10)
#we can remove purchase from dataset because we have purchase_level

data = data.drop('Purchase', axis = 1)
data.head()
#we can tidy our dataframe with melt function.

melted = pd.melt(frame = data, id_vars = 'Purchase_level', value_vars = ["Product_Category_1", "Product_Category_2", "Product_Category_3"])

melted
#concatenate data

data_head = data.head() 

data_tail = data.tail()

data_concatenate = pd.concat([data_head, data_tail], axis = 1, ignore_index = False)

data_concatenate