#Import all the libraries to prefom data anlysis.
import pandas as pd #Pandas is used to to apply operations on datafram.
import numpy as np #Numpy is numerical python use to perform mathematical operations.
import matplotlib.pyplot as plt #Used for visualization
import seaborn as sns #used for visulaization
#Importing the airbnb_new_york dataset to the variable name df using pandas.
df=pd.read_csv('../input/air-bnb-ny-2019/AB_NYC_2019.csv', encoding='latin1')
#Printing the first 5 rows of the dataframe.
df.head()
#Below code returns the number of rows and columns availbale in the dataframe.
df.shape
#.info() return the datatype and availability of any null values in any column of the dataframe.
df.info()
#Below code is provides no. of elements, min, max, standard devition and other quantile values.
df.describe()
#isnull() is used to know if there is any null values.
#if one any() is used the it will return null value availibility of each column and twice is used to know for the wholw dataframe.
df.isnull().any().any()
#.sum() return the number of null values present in the columns.
df.isnull().sum()
#Here we are deleting latitude. longitude, last_review & reviews_per_month columns to get the optimal analysis of the dataframe.
df.drop(['latitude', 'longitude','last_review','reviews_per_month'], inplace=True, axis=1)
#again printing first 10 rows to check if those columns are deleted or not.
df.head(10)
#Deleting remaining null values form name & host_name columns.
df.dropna(inplace=True)
#again checking the number of rows and columns of the data after completing the cleaning part.
df.shape
sns.set_style('darkgrid')
plt.figure(figsize=(14, 6))
plt.xticks(rotation = 90)
sns.countplot(df['neighbourhood_group'])
sns.countplot(df['room_type'])
sns.countplot(x='neighbourhood_group', hue='room_type', data=df)
plt.scatter(x='neighbourhood_group', y='price', data=df )
#In the below code we are using groupby() function so that every category present inside the neighborhood_group cloumn
#can be made into individual unit with all its other values to be transformed into their sum or with their mean values.
df_agg = df.groupby(['neighbourhood_group'])
price=df_agg['price'].agg(np.sum)
number_of_reviews=df_agg['number_of_reviews'].agg(np.mean)
minimum_nights=df_agg['minimum_nights'].agg(np.mean)
plt.figure(figsize=(12,8))

plt.subplot(221)
plt.plot(price, 'ro')
plt.ylabel('Price')
plt.xlabel('Cities')

plt.subplot(222)
plt.plot(number_of_reviews, 'go')
plt.xticks('Cities')
plt.ylabel('Mean Number of Reviews')
plt.plot(minimum_nights, 'bo')
plt.figure(figsize=(12, 8))
sns.barplot(x='neighbourhood_group', y='price',hue='room_type', data=df)
sns.catplot(x='neighbourhood_group', y='number_of_reviews', data=df)



