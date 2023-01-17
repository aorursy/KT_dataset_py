# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import seaborn as sns

import matplotlib.pyplot as plt

import os

print(os.listdir("../input"))

sns.set(style='whitegrid')

# Any results you write to the current directory are saved as output.
dataFrame = pd.read_csv('../input/BlackFriday.csv')
#Let see the first five rows of the dataset

dataFrame.head()
#Lets see some statistics about the data

dataFrame.describe()
#Lets see present total Nans in each column

dataFrame.isnull().sum()
dataFrame.shape
sns.countplot(x='Occupation', data=dataFrame)
#Lets see what are the city category we have here

dataFrame.City_Category.value_counts()
fig, ax = plt.subplots(figsize=(20,10))

hue_order = ['A', 'B', 'C']

sns.countplot(x='Occupation', hue='City_Category', hue_order=hue_order, data=dataFrame, palette='BuGn')
#Let see what are the age groups we have here

dataFrame.Age.value_counts()
#Lets visualize age-group distribution 

age_order = ['0-17', '18-25', '26-35', '36-45', '46-50', '51-55', '55+']

sns.countplot(x='Age', data=dataFrame, order=age_order)
sns.countplot(x='Age', hue='Gender', data=dataFrame, order=age_order, palette=sns.cubehelix_palette(8))
#creating a separate dataframe with two columns namely age and product_category_1 from original dataframe

data_age_prod1 = pd.concat([dataFrame['Age'], dataFrame['Product_Category_1']], axis=1)

#mapping each age group to integer value

data_age_prod1['Age'] = data_age_prod1['Age'].map({'0-17':0, '18-25':1, '26-35':2, '36-45':3, '46-50':4, '51-55':5, '55+':6})
data_age_prod1.head()
#age vs product_category_1

plt.subplots(figsize=(10,7))

sns.boxplot(x='Age', y='Product_Category_1', data=data_age_prod1)
#Filling the NaNs, here NaNs means users haven't purchased that product 

dataFrame.fillna(value=0, inplace=True)
dataFrame['Product_Category_2'].unique()
dataFrame['Product_Category_2'] = dataFrame['Product_Category_2'].astype(int)

dataFrame['Product_Category_3'] = dataFrame['Product_Category_3'].astype(int)
plt.subplots(figsize=(20,10))

sns.boxplot(x='Age', y='Product_Category_2', hue='Gender', order=age_order, data=dataFrame)
plt.subplots(figsize=(20,10))

sns.boxplot(x='Age', y='Product_Category_1', hue='Gender', order=age_order, data=dataFrame)
plt.subplots(figsize=(20,10))

sns.boxplot(x='Age', y='Product_Category_3', hue='Gender', order=age_order, data=dataFrame)
city_order = ['A', 'B', 'C']

sns.barplot(x='City_Category', y='Purchase', order=city_order, data=dataFrame)
sns.countplot(dataFrame['Gender'])
sns.barplot(x='Gender', y='Purchase', data=dataFrame)
sns.boxplot(x='Gender', y='Purchase', data=dataFrame)
sns.barplot(x='Marital_Status', y='Purchase', data=dataFrame)
sns.boxplot(x='Marital_Status', y='Purchase', data=dataFrame)
#Creating a new dataframe by concating 'Occupation' and 'Purchase' from original dataframe

df_occu_purchase = pd.concat([dataFrame['Occupation'], dataFrame['Purchase']], axis=1)
df_occu_purchase.head()
#Here we are creating another dataframe from df_occu_purchase using groupby occupation and then taking the sum

df2 = pd.DataFrame(df_occu_purchase.groupby('Occupation').sum().reset_index())
df2.head()
sns.set(style = 'white')

red = sns.color_palette('Reds')[-2]

sns.jointplot(x='Occupation', y='Purchase', data=df2, kind='kde', space=0, height=7, cmap='Reds')
#we can drop User_ID and Product_ID as these are not needed further

dataFrame.drop(columns=['User_ID', 'Product_ID'], inplace=True)
#Now, we intend to see correlation matrix, for this mapping object type value to integers 

dataFrame['Age'] = dataFrame['Age'].map({'0-17':0, '18-25':1, '26-35':2, '36-45':3, '46-50':4, '51-55':5, '55+':6})

dataFrame['City_Category'] = dataFrame['City_Category'].map({'A':0, 'B':1, 'C':2})

dataFrame['Gender'] = dataFrame["Gender"].map({'F':0, 'M':1})
dataFrame.head()
corr_mat = dataFrame.corr()

f, ax = plt.subplots(figsize=(9,5))

sns.heatmap(corr_mat, annot=True, ax=ax)