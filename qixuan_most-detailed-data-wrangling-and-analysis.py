# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 

import os

import pandas as pd

import matplotlib as mpl

import matplotlib.pyplot as plt

import seaborn as sns

import numpy as np  

# Configure visualisations

%matplotlib inline

mpl.style.use('ggplot') #with this, your figures would be more beautiful
#loading the dataset 

df_house=pd.read_csv("../input/housesalesprediction/kc_house_data.csv")

df_house.head()
#remove some columns that we do not need for the following steps

df_house.drop(['id', 'sqft_living15','sqft_lot15'], axis = 1, inplace = True)

df_house.head()
df_house.info()
df_house['zipcode'] = df_house['zipcode'].astype('category')
df_house.describe()
df_house.describe(include=['O'])
df_house.describe(include=['category'])
df_house.zipcode.unique()
df_house.bathrooms.unique()
df_house.bedrooms.unique()
df_house.view.unique()
df_house.condition.unique()
df_house.yr_renovated.unique()
df_house.waterfront.unique()
df_house.grade.unique()
df_house.floors.unique()
df_house.date.value_counts()
#remove 'T000000'

df_house['date'] = df_house.date.str.replace('T000000' , '')
regex = r'''(?x)

    # Year

    (?:(?:(?:\d{2})?\d{2})

    #30-day months

    (?:(?:(?:0[469]|11)(?:30|[12][0-9]|0[1-9]))|

    #31-day months

    (?:(?:0[13578]|1[02])(?:3[01]|[12][0-9]|0[1-9]))|

    #February (29 days every year)

    (?:(?:0?2)(?:[12][0-9]|0?[1-9]))))

'''



df_house[~df_house["date"].str.match(regex)]
#change the datetime format

df_house['date']= pd.to_datetime(df_house['date'])

df_house.head()
df_house.zipcode.value_counts()
df_house.loc[(df_house.sqft_living!=df_house.sqft_above+df_house.sqft_basement)]
df_house.loc[(df_house.yr_renovated<=df_house.yr_built) & (df_house.yr_renovated !=0) ]
df_house[(df_house.bathrooms==0) & (df_house.bedrooms==0)]
df_house[(df_house.floors==1)]
df_house[(df_house.floors==1) & (df_house.sqft_basement==0) & (df_house.sqft_living > df_house.sqft_lot)]
df_house.drop(df_house.index[13278],inplace=True)
df_house[(df_house.floors==1) & (df_house.sqft_basement==0) & (df_house.sqft_living > df_house.sqft_lot)]
df_house[df_house.duplicated(["lat", "long", "date", "yr_built",

                               "sqft_living", "sqft_lot"], keep=False)]
df_house.drop_duplicates(["lat", "long", "date", "yr_built",

                          "sqft_living", "sqft_lot"], keep='last', inplace=True)
df_house['price'].describe()
#calculate the distance of 3*standard deviation

three_sigma=3*(df_house.price.std())

#finding the rows that has 3sd far away from the mean

price_Editrule=[]

for i in range(len(df_house)):

    #calculate the mean of price

    price_mean=df_house.price.mean()

    #absolute value of price

    absolute_value=abs(df_house.price.iloc[i])

    #distance of 3sd away from mean

    a=price_mean+three_sigma

    #justify the price of property larger than 3sd from the mean

    if absolute_value>a:

        price_Editrule.append(i)



df_house.iloc[price_Editrule]
df_house.boxplot(column='price',sym='k.')

plt.show()
#log of each price

df_house['price']=np.log(df_house.price)

#Now plot a boxplot again

df_house.boxplot(column='price',sym='k.')

plt.show()
df_house[df_house.price>15.5]
plt.figure(figsize=(15,10))

sns.heatmap(df_house.corr(), annot=True, fmt=".2f")

plt.show()
#plot a price histgram

df_house['price'].hist(bins=100)

plt.show()
#plot living space histogram

sns.distplot(df_house['sqft_living'])

plt.show()
#2-dimention mahalanobis distance detect outliers

#The greater the value of mahalanobis distance, the higher probability of outlier it is.



from pandas import Series

from scipy.spatial import distance 

#build a new dataframe which contains juat column for price and sqft_living 

hw=df_house[['price','sqft_living']]

#define number of outliers

n_outliers =6

#use mahalanobis distance to detect each point

#series used to generate distance for each property

#hw.iloc stands for the outside index of each row; hw.mean stands for value of mean for 2 columns; np.mat create correlation matrix and reverse the matrix

m_dist_order = Series([float(distance.mahalanobis(hw.iloc[i], hw.mean(), np.mat(hw.cov().as_matrix()).I) ** 2) for i in range(len(hw))]).sort_values(ascending=False).index.tolist()  

#If the property is outlier return True, otherwise return False

is_outlier = [False, ] * (len(hw)) 

for i in range(n_outliers):  

    is_outlier[m_dist_order[i]] = True 

#outliers are displayed in red, others are displayed in blue

color = ['b', 'r']  

#turn True to 1, False to 0

pch = [1 if is_outlier[i] == True else 0 for i in range(len(is_outlier))]  

#turn 1 to 'r', turn 0 to 'b'

cValue = [color[is_outlier[i]] for i in range(len(is_outlier))]  



#plotting

fig = plt.figure()  

#set title

plt.title('Scatter Plot')  

#set x label

plt.xlabel('sqft_living')  

#set y label

plt.ylabel('price')  

#draw scatter

plt.scatter(hw['sqft_living'],  hw['price'], s=40, c=cValue) 

plt.show()  
index_list1=[]

for i in range(len(pch)):

    #if value in pch is 1, it is an outlier

    if pch[i]==1:

        index_list1.append(i)

#show outliers from 2-dimention mahalanobis distance

df_house.iloc[index_list1]
#3-dimention mahalanobis distance detect outliers

#The greater the value of mahalanobis distance, the higher probability of outlier it is.  

from mpl_toolkits.mplot3d import Axes3D



#build a new dataframe including three columns

hw=df_house[['price','sqft_living','grade']]    

  

n_outliers = 6 #select 6 outliers 

#iloc[]take 3 columns and 1 row   hw.mean()here is an array of three variables    np.mat(hw.cov().as_matrix()).I is the inverse matrix of covariance   **为乘方  

#Series's output is: the index is on the left, the value is on the right

#m_dist_order is a one-dimensional array that holds the index in descending order of Series

m_dist_order =  Series([float(distance.mahalanobis(hw.iloc[i], hw.mean(), np.mat(hw.cov().as_matrix()).I) ** 2)  

       for i in range(len(hw))]).sort_values(ascending=False).index.tolist()  

is_outlier = [False, ] * len(hw)

for i in range(n_outliers):#mahalanobis distance value is marked as True

    is_outlier[m_dist_order[i]] = True  



#outliers are displayed in red, others are displayed in blue

color = ['b', 'r']  

#turn True to 1, False to 0

pch = [1 if is_outlier[i] == True else 0 for i in range(len(is_outlier))]  

#turn 1 to 'r', turn 0 to 'b'

cValue = [color[is_outlier[i]] for i in range(len(is_outlier))]  



#plotting

fig = plt.figure()  

#using 3 dimention 

ax1 = fig.gca(projection='3d')  

#set title and labels

ax1.set_title('Scatter Plot')  

ax1.set_xlabel('price')  

ax1.set_ylabel('sqft_living')  

ax1.set_zlabel('grade')  

#plot scatter plot

ax1.scatter(hw['sqft_living'], hw['price'], hw['grade'], s=30, c=cValue)  

plt.show()  
index_list2=[]

#pch return True to 1. Here we can find the index of the outliers 

for i in range(len(pch)):

    if pch[i]==1:

        index_list2.append(i)

#show outliers from 3-dimention mahalanobis distance

df_house.iloc[index_list2]
#Set values for particular cell in index_list

df_house.iloc[index_list1,1]=0

#replace 0 with NaN

df_house['price'].replace(0,np.NaN,inplace=True)
df_house

#Writing a CSV file with the pandas library if you want

#df_house.to_csv('house.csv', encoding='utf-8',index=False)