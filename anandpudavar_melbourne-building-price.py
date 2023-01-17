#Importing required libraries.
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import datetime

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
#Reading the file.
df=pd.read_csv('/kaggle/input/melb_data.csv')
df
#Getting the dimesnions of the dataset.
df.shape
#Getting the column names.
df.columns
#Getting the data types of all the series.
df.info()
#Looking for missing values.
df.isnull().sum()
#Looking for the frequency of occurances in this column
df.Car.value_counts()
#Looking at the missing values.
df.Car.isnull().sum()
#Looking at the statistical parameters.
df.Car.describe()
#Replacing the NaN values.
df.Car.replace({np.nan:2},inplace=True)
#Checking the distribution.
sns.countplot(df.Car);
#Looking at the statistical parameters.
df.BuildingArea.describe()
#The number of missing values.
df['BuildingArea'].isnull().sum()
#Looking at the corresponding entries in other relevant columns to see any relation.
df[df['BuildingArea'].isnull()][['BuildingArea','YearBuilt','Price','CouncilArea','Suburb']].head()
#Comapring the number of council areas in two cases.
print('The number of council areas represented :',df[df['BuildingArea'].isnull()].CouncilArea.nunique())
print('The original number of council areas in the dataset:',df.CouncilArea.nunique())
#Comparing the number o fsuburbs in both cases.
print('The number of suburbs represented :',df[df['BuildingArea'].isnull()].Suburb.nunique())
print('The original number of suburbs in the dataset:',df.Suburb.nunique())
#Replacing NaN with 'NA'
df.BuildingArea.replace({np.nan:'NA'},inplace=True)
#Getting the number of missing values.
df['YearBuilt'].isnull().sum()
#Getting the values in this series.
df.YearBuilt.value_counts()
#Statistical summary of this series.
df.YearBuilt.describe()
#Replacing NaN with 'NA'
df.YearBuilt.replace({np.nan:'NA'},inplace=True)
#The statistical summary of the column
df.CouncilArea.describe()
#Looking at the number of missing values in thus column
df['CouncilArea'].isnull().sum()
#Replacing the null values with 'Unavailable'
df.CouncilArea.replace({np.nan:'Unavailable'},inplace=True)
#Confirmation of missing value substitution.
df.isnull().sum()
#Looking the frequency of various suburb types.
df.Suburb.value_counts()
#Looking the frequency of the no. of rooms
sns.countplot(df.Rooms);
#Looking the frequency of different types.
#h - house,cottage,villa, semi,terrace; u - unit, duplex; t - townhouse; 
df.Type.value_counts()
#Distribution of price values.
sns.distplot(df['Price']);
#Statistical summary of the prices.
df.Price.describe()
#Looking the frequency of various methods.
df.Method.value_counts()
#Looking the frequency of different real estate agents.
df.SellerG.value_counts()
#Looking the frequency of different selling date.
df['Date']= pd.to_datetime(df['Date']) 
df1=pd.DataFrame()
df1['year'] = df['Date'].dt.year
df1['month']=df['Date'].dt.month
#Getting the count of values in this series.
df1.year.value_counts()
#Getting the frequency of houses sold in each month.
df1.month.value_counts()
#Distribution of Distance from CBD(central city area of the city of Melbourne)
sns.distplot(df['Distance'],color='red');
#Looking the frequency of number of bedrooms.
df.Bedroom2.value_counts()
#Looking the frequency of number of bathrooms.
df.Bathroom.value_counts()
#Looking the frequency of landsize.
df.Landsize.value_counts()
#Getting the frequency of various region names in the data.
df.Regionname.value_counts()
#Getting the frequency of Governing council for the different areas.
df.CouncilArea.value_counts()
#Checking the distribution
df.Propertycount.value_counts()
#Analysis of room features.
room_features=['Rooms','Bedroom2','Bathroom']
plt.figure(figsize=(16,50))

i = 1
for feature in room_features:
    plt.subplot(10,1,i)
    sns.boxplot(x=df[feature],y=df['Price'])
    plt.xlabel(feature,fontsize=18)
    plt.ylabel('Price',fontsize=18)
    i+=1
#Analysing sale price against number of car parks.
sns.boxplot(x=df['Car'],y=df['Price']);
#Analysing the number of rooms to number of car parks.
sns.boxplot(x=df['Car'],y=df['Rooms']);
#Analysing type of buildings to their price.
sns.boxplot(x=df['Type'],y=df['Price']);
#Analysing the room number in various building types, which could better explain the previous result.
sns.boxplot(x=df['Type'],y=df['Rooms']);
#Analysing the variation of sales price according to sales method.
sns.boxplot(x=df['Method'],y=df['Price']);
#Sellers and price range.
df2=df.groupby(['SellerG'],as_index=False)['Price'].mean()
df2=df2.sort_values(by='Price',ascending=False)
top_sell=df2.SellerG[:5].to_list()
print("The sellers that sold buildings with highest prices are:",top_sell)
#Getting the features related to the selected sellers.
df.query('SellerG in ["Weast", "Darras", "VICProp", "Sotheby\'s", "Lucas"]').reset_index(drop=True)
#Looking at the variation of sale prices across regions.
plt.figure(figsize=(15,4))
f2=sns.boxplot(x=df.Regionname,y=df.Price);
f2.set_xticklabels(f2.get_xticklabels(),rotation=90);
#Variation of sales price across council areas.
plt.figure(figsize=(20,4));
fi=sns.boxplot(x=df['CouncilArea'],y=df['Price']);
plt.xlabel('CouncilArea',fontsize=18);
plt.ylabel('Price',fontsize=18);
fi.set_xticklabels(fi.get_xticklabels(),rotation=90,fontsize=14);
#Mean price values across various region names and council areas.
df.groupby(['Regionname','CouncilArea'],as_index=False)['Price'].mean()
#Relationship between distance from CBD and sales price.
X=df.Price
Y=df.Distance
sns.set(style="darkgrid")
sns.jointplot(X,Y, kind='reg',color='r',scatter=False);
#Analysis of relationship between region name and distance from CBD
f3=sns.boxplot(x=df['Regionname'],y=df['Distance']);
f3.set_xticklabels(f3.get_xticklabels(),rotation=90);
#Variation of sales price across time.
sns.lineplot(x="Date", y="Price", data=df)
plt.xticks(rotation=15)
plt.title('Sales price with time')
plt.show()
#Representation of months in the data
sns.countplot(df1.month);
#Variation of price across months.
sns.boxenplot(x=df1.month,y=df.Price);
#Variation of sales price across years.
sns.violinplot(x=df1.year,y=df.Price);