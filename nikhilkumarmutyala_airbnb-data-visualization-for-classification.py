#import required

import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt
#read the file 'NYC_2019.csv' from the file

df = pd.read_csv('/kaggle/input/new-york-city-airbnb-open-data/AB_NYC_2019.csv')
#obtain information about the dataframe

df.info()
#view dataframe df

df
#find no of columns and no of rows

df.shape
#obtaining the description of the dataframe

df.describe()
#finding out if there are any null or empty values

df.isnull().sum()
#delete the row 'last_review'

del df['last_review']
#delete the row 'last_review'

del df['host_name']
df
#find if there are any null values in the dataset

df.isnull().sum()
#fill NaN data with 0 in the dataframe and display the data

df.fillna('0',inplace=True)

df
#remove the null values from the dataset

df=df[~(df['name']=='0')]

df
#categorize the neighbourhood group into categories

df.neighbourhood_group = df.neighbourhood_group.astype('category')
#print the categories in neighbourhood group

df.neighbourhood_group.cat.categories
#crosstab the columns neighbourhood group and room type

pd.crosstab(df.neighbourhood_group, df.room_type)
#catplot room type and price

sns.catplot(x="room_type", y="price", data=df);
#catplot neighbourhood_group and price

sns.catplot(x="neighbourhood_group", y="price", kind="boxen",

            data=df);
# create countplot roomtype and neighbourhood type

plt.figure(figsize=(10,10))

df1 = sns.countplot(df['room_type'],hue=df['neighbourhood_group'], palette='plasma')
#boxplot neighbourhood_group and room availability

plt.figure(figsize=(10,10))

df1 = sns.boxplot(data=df, x='neighbourhood_group',y='availability_365',palette='plasma')