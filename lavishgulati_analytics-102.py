# importing pandas and numpy libraries

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

from pylab import rcParams
# reading train dataset

train = pd.read_csv("../input/Train_A102.csv")
# shape/dimensions of the dataset train

train.shape

#Thus 'train' dataset has 8523 rows and 12 columns..
# train dataset head

train.head()
# finding data types of each column

train.dtypes
# finding maximum and minimum values of item weight

train1 = pd.read_csv("../input/Train_A102.csv", usecols=["Item_Identifier", "Item_Weight"], index_col=["Item_Identifier"], header=0)

train1.head()

x = train1.max()

y = train1.min()

print(x)

print

print(y)

'''     Inference: 1. Minimum weight is 4.555     2. Maximum weight is 21.35   '''
# finding maximum and minimum values of item visibility

train1 = pd.read_csv("../input/Train_A102.csv", usecols=["Item_Identifier", "Item_Visibility"], index_col=["Item_Identifier"], header=0)

train1.head()

x = train1.max()

y = train1.min()

print(x)

print

print(y)

'''     Inference: 1. Minimum visibility is 0.0    2. Maximum visibility is 0.328391   '''
# finding maximum and minimum values of item MRP

train1 = pd.read_csv("../input/Train_A102.csv", usecols=["Item_Identifier", "Item_MRP"], index_col=["Item_Identifier"], header=0)

train1.head()

x = train1.max()

y = train1.min()

print(x)

print

print(y)

'''     Inference: 1. Minimum MRP is 31.29   2. Maximum MRP is 266.8884   '''
# finding maximum and minimum values of item outlet sales

train1 = pd.read_csv("../input/Train_A102.csv", usecols=["Item_Identifier", "Item_Outlet_Sales"], index_col=["Item_Identifier"], header=0)

train1.head()

x = train1.max()

y = train1.min()

print(x)

print

print(y)

'''     Inference: 1. Minimum outlet sales is 33.29   2. Maximum outlet sales is 13086.9648   '''
# Statistics of Item Weight

train.Item_Weight.describe()
train.boxplot(column=["Item_Weight"])
train1 = pd.read_csv("../input/Train_A102.csv", usecols=["Item_Weight"])

train1.head()
train1.notnull()
# Analysing data by grouping it according to Item Fat Content

train1 = pd.read_csv("../input/Train_A102.csv", usecols=["Item_Weight", "Item_Fat_Content"])

train1.head()
# Statistics of Item Weight for different groups of Item Fat Content

train1.groupby("Item_Fat_Content").describe()

# Inference: The different groups of the Item Fat Content have same meaning and thus, are redundant. 

# Example: (LF, Low Fat, lowfat) and (Regualar, regular)

# Thus, these must be grouped together for proper analysis of data..
# Renamed dataframe 

renamed_train1 = train1.set_index("Item_Fat_Content").rename({"LF": "Low Fat", "low fat": "Low Fat", "reg": "Regular"})
# Actual statistics of Item Weight for different groups of Item Fat Content

renamed_train1.groupby("Item_Fat_Content").describe()
renamed_train1.groupby("Item_Fat_Content").mean().plot()

#Inference: A strange observation is that Low Fat items have more fat than Regular items..
# Analysing data by grouping it according to Item Type

train1 = pd.read_csv("../input/Train_A102.csv", usecols=["Item_Weight", "Item_Type"])

train1.head()
# Statistics of Item Weight for different groups of Item Type

train1.groupby("Item_Type").describe()
train1.groupby("Item_Type").mean().plot(kind="line", figsize=(14,5))

#Inference: Others group of foods have highest item weight followed by Starchy Foods.

#           Breads group of foods have lowest item weight followed by Hard Drinks.
# Analysing data by grouping it according to Outlet Size

train1 = pd.read_csv("../input/Train_A102.csv", usecols=["Item_Weight", "Outlet_Size"])

train1.head()
train1.groupby("Outlet_Size").describe()
train1.groupby("Outlet_Size").mean().plot(kind="line")

#Inference: Item with higher outlet size has higher weight..
# Statistics of Item Visibility

train.Item_Visibility.describe()
train.boxplot(column=["Item_Visibility"])
''' A significant number of outliers are detected in Item Visibility feature'''
# Detection of Outliers in Item Visibility feature using Tukey Outlier Labelling

iqr =  0.094585 - 0.026989

a = 0.026989 - 1.5*iqr

b = 0.094585 + 1.5*iqr

print("Lower boundary for inliers: ", a)

print("Upper boundary for inliers: ", b)
train1 = pd.read_csv("../input/Train_A102.csv", usecols=["Item_Visibility"])

train1.head()

X = train1.iloc[:, 0:1].values

Item_Visibility = X[:, 0]

ItemVis_Outliers = (Item_Visibility > 0.19597900000000001)

train1[ItemVis_Outliers].head()

# Inference: A dataframe with outlier elements are detected and shown below.
ItemVis_new = train1.drop(train1[ItemVis_Outliers].index)

ItemVis_new.head()

#The outlier elements are removed and stored in the dataframe ItemVis_new
# New statistics of Item Visibility after Outliers Deletion

ItemVis_new.describe()
ItemVis_new.boxplot()
# Statistics of Item MRP

train.Item_MRP.describe()
train.boxplot(column=["Item_MRP"])
# Statistics of Item Outlet Sales

train.Item_Outlet_Sales.describe()
train.boxplot(column=["Item_Outlet_Sales"])
''' A significant number of outliers are detected in Item Outlet Sales feature'''
# Detection of Outliers in Item Outlet Sales feature using Tukey Outlier Labelling

iqr =  3101.296400 - 834.247400

a = 834.247400 - 1.5*iqr

b = 3101.296400 + 1.5*iqr

print("Lower boundary for inliers: ", a)

print("Upper boundary for inliers: ", b)
train1 = pd.read_csv("../input/Train_A102.csv", usecols=["Item_Outlet_Sales"])

train1.head()

X = train1.iloc[:, 0:1].values

Item_Outlet_Sales = X[:, 0]

IOS_Outliers = (Item_Outlet_Sales > 6501.8699)

train1[IOS_Outliers].head()

# Inference: A dataframe with outlier elements are detected and shown below.
IOS_new = train1.drop(train1[IOS_Outliers].index)

IOS_new.head()

#The outlier elements are removed and stored in the dataframe IOS_new 
# New statistics of Item Outlet Sales after Outlier Deletion

IOS_new.describe()
IOS_new.boxplot()