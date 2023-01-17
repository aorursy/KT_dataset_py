#Importing all the required libraries for the analysis of the automobile dataset.
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
#Imporing the dataset using pandas, naming it as df_automobile.
#Definig the null value or NaN values as '?' using na_values keyword.
df_automobile = pd.read_csv('../input/automobile-dataset/Automobile_data.csv', na_values=['?'])
#Below statement returns the first  rows ofthe dataset with their column names.
df_automobile.head(10)
#Returns the shape (Rows & Columns) of the dataset
df_automobile.shape
#Returns the Column names available in the dataset
df_automobile.columns
#isnull() function checks the presence of any nul value 
#.any() function will return boolean values of the cloumns containig null values.
df_automobile.isnull().any()
#.sum() calculates the null values availble in the particular columns
df_automobile.isnull().sum()
#plotting the histogram of colums with integer datatype.
df_automobile.hist(grid=True,figsize=(13,13), )
#The below code is used to fill null values or NaN values with either of their mean values or mode values.
#If a column with continous values is having any null values, fill it with their mean values
#and if any column contains categorical value fill it with mode
df_automobile[ 'normalized-losses' ].fillna(value= df_automobile['normalized-losses'].mean(), inplace=True)
df_automobile['num-of-doors'].fillna(value = df_automobile['num-of-doors'].mode().values[0], inplace=True)
df_automobile['bore'].fillna(value = df_automobile['bore'].mean(), inplace=True)
df_automobile['stroke'].fillna(value = df_automobile['stroke'].mean(), inplace=True)
df_automobile['horsepower'].fillna(value = df_automobile['horsepower'].mean(), inplace=True)
df_automobile['peak-rpm'].fillna(value = df_automobile['peak-rpm'].mean(), inplace=True)
df_automobile['price'].fillna(value=df_automobile['price'].mean(),inplace=True)
#Now we can see we dont have any null values.
df_automobile.isnull().any()
#Again priting the top 10 values to check if the null values are replaced with their new values or not.
df_automobile.head(10)
df_automobile.shape
df_automobile.describe()
#Heatmap technique is used to visualize the relation of the features.
plt.figure(figsize=(20,20))
sns.heatmap(df_automobile.corr(), annot = True, cmap = 'YlGnBu',fmt = '.2f')
# From the above figure we can see that highway-mpg and city-mpg are highly correlated 
#so we are deleting highway-mpg  column.
df_new = df_automobile.drop(['highway-mpg'], axis=1)  
df_new
#again checking the shape of the dataset.
df_new.shape
#Ince the make column is having same elements i.e. the automobile companies so we are aggregating
#all those companies names and their other attributes by using either their mean value or adding them.
data_agg = df_new.groupby('make')
wheel_base = data_agg['wheel-base'].agg(np.mean)
length = data_agg['length'].agg(np.mean)
width = data_agg['width'].agg(np.mean)
height = data_agg['height'].agg(np.mean)
curb_weight = data_agg['curb-weight'].agg(np.mean)
engine_size = data_agg['engine-size'].agg(np.mean)
bore = data_agg['bore'].agg(np.mean)
stroke = data_agg['stroke'].agg(np.mean)
compression_ration = data_agg['compression-ratio'].agg(np.mean)
horsepower = data_agg['horsepower'].agg(np.mean)
peak_rpm = data_agg['peak-rpm'].agg(np.mean)
city_mpg = data_agg['city-mpg'].agg(np.mean)
price = data_agg['price'].agg(np.mean)
#print(length)
#print(height)
#Here we are plotting all the features with other attributes to get the insights from the dataset.
plt.figure(figsize=(13,10))
plt.subplot(221)
plt.plot(wheel_base, 'r--')
plt.xticks(rotation = 90)
plt.xlabel('make')
plt.ylabel('peak-rpm')
plt.show()

plt.figure(figsize=(13,10))
plt.subplot(222)
plt.plot(length, 'ro')
plt.xticks(rotation = 90)
plt.xlabel('make')
plt.ylabel('length')
plt.show()

plt.figure(figsize=(13,10))
plt.subplot(223)
plt.plot(width, 'g^')
plt.xticks(rotation = 90)
plt.xlabel('make')
plt.ylabel('width')
plt.show()

plt.figure(figsize=(13,10))
plt.subplot(224)
plt.plot(height, 'rs')
plt.xticks(rotation = 90)
plt.xlabel('make')
plt.ylabel('height')
plt.show()

plt.figure(figsize=(13,10))
plt.subplot(225)
plt.plot(curb_weight, 'ro')
plt.xticks(rotation = 90)
plt.xlabel('make')
plt.ylabel('curb-weight')
plt.tight_layout()
plt.show()
plt.figure(figsize=(13, 8))
plt.xticks(rotation=90)
sns.boxplot(x = 'make', y = 'price', data = df_new)
plt.figure(figsize=(13, 8))
sns.boxplot(x = 'body-style', y = 'price', data=df_new)
plt.figure(figsize=(13, 8))
sns.violinplot(x = 'engine-type', y = 'price', data=df_new)
plt.figure(figsize = (15, 8)), 
sns.countplot(df_automobile['make'], data=df_automobile)
plt.xticks(rotation = 45)

