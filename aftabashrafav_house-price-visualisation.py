import os

os.getcwd()

os.chdir('/kaggle/input/houseprice')
!pip install missingno #INSTALLING MISSINGNO LIBRARY
import numpy as np  # linear algebra

import pandas as pd # data processing

import matplotlib.pyplot as plt #for plotting

import seaborn as sns  #Data visualisation

import missingno as msno 
#reading the dataframe

df=pd.read_csv("train.csv",index_col='Id')
df.head()
df.tail()
df.info()

#it will give the overall idea about the dataframe 
df.describe()

# getting a basic description of your data
df.shape #it will give the count of number of rows and coloumns in the dataframe
#Let's look at the skewness of our dataset



df.skew()
df.dtypes.value_counts() #it will the data types and its count
#Let us examine numerical features in the train dataset

#numeric_features = df.select_dtypes(include=[np.number])

#numeric_features.columns



num_col=df.select_dtypes(exclude='object')

cat_col=df.select_dtypes(exclude=['int64','float64'])
#Let us examine categorical features in the train dataset





categorical_features = df.select_dtypes(include=[np.object])

categorical_features.columns

#categorical_features.shape
msno.heatmap(df)
# HEATMAP TO SEE MISSING VALUES

plt.figure(figsize=(15,5))

sns.heatmap(num_col.isnull(),yticklabels=0,cbar=False,cmap='viridis')
#heatmap shows LotFrontage,MasVnrArea,GarageYrBlt have the missing values


#missing value

num_col.isnull().sum()
#Datacleaning

#so here we want deal with missing values and replace all nun values of column

#mainly LotFrontage,MasVnrArea,GarageYrBlt
#correlation map

f,ax = plt.subplots(figsize=(20,2))

sns.heatmap(num_col.corr().iloc[7:8,:], annot=True, linewidths=.8, fmt= '.1f',ax=ax)

# this shows that MasVnrArea is not highly corelated to any other feature




sns.kdeplot(num_col.MasVnrArea,Label='MasVnrArea',color='b');



#so most of the values is near by 0 so we can replace the nan value with 0Analysis of features against sale price
num_col.MasVnrArea.replace({np.nan:0},inplace=True)

num_col
f,ax = plt.subplots(figsize=(20,2))

sns.heatmap(num_col.corr().iloc[1:2,:], annot=True, linewidths=.8, fmt= '.1f',ax=ax)



#lotFrontage has enough corelation with sales price so we cant drop it
num_col.LotFrontage.describe()



num_col['LotFrontage'].replace({np.nan:num_col.LotFrontage.mean()},inplace=True)
sns.scatterplot(x=num_col['LotFrontage'],y=num_col['LotArea'],hue=num_col['SalePrice']);
sns.scatterplot(x=num_col['LotFrontage'],y=num_col['1stFlrSF'],hue=num_col['SalePrice']);
sns.scatterplot(x=num_col['LotFrontage'],y=num_col['TotalBsmtSF'],hue=num_col['SalePrice']);
sns.scatterplot(x=num_col['LotFrontage'],y=num_col['GarageArea'],hue=num_col['SalePrice']);
sns.scatterplot(x=num_col['LotFrontage'],y=num_col['BedroomAbvGr'],hue=num_col['SalePrice']);
#Creating a heat map of all the numerical features.

plt.figure(figsize=(20,20))

mat = np.round(num_col.corr(), decimals=2)

sns.heatmap(data=mat, linewidths=1, linecolor='black');
#Getting features that have a correlation value greater than 0.5 against sale price.

for val in range(len(mat['SalePrice'])):

    if abs(mat['SalePrice'].iloc[val]) > 0.5:

        print(mat['SalePrice'].iloc[val:val+1]) 

sns.barplot(df.OverallQual,df.SalePrice);
sns.boxplot(y=df['SalePrice'],x=df['Street'],palette='rainbow');
sns.boxplot(y=df['SalePrice'],x=df['LotShape'],palette='rainbow');
df.Electrical.replace({np.nan:df['Electrical'].mode()},inplace=True)
df.Electrical.unique()
sns.countplot(df['Electrical']);
sns.lmplot(data= df,x='SalePrice',y='YearBuilt',scatter=False);
sns.lmplot(data= df,x='SalePrice',y='YearRemodAdd',scatter=False);
sns.lmplot(data= df,x='SalePrice',y='FullBath',scatter=False);