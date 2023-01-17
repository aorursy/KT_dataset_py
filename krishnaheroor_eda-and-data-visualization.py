import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
corona=pd.read_csv("../input/covid19-italy-province/covid19_italy_province.csv")
#after the loading the data. Next step is to view/see the top 10 rows of the loaded data set



corona.head()
#last 10 rows of loaded data set



corona.tail(10)
corona.describe()
#information about each var



corona.info()
#we will be listing the columns of all the data.

#we will check all columns



corona.columns
corona.sample(frac=0.01)
#sample: random rows in the dataset

#useful for future analysis

corona.sample(5)
#next, how many rows an columns are there in the loaded data set



corona.shape
# and, will check null on all the data and if there is any null, getting the sum of all the null data's



corona.isna().sum()
#Removing duplicates if any



corona.duplicated().sum()

corona.drop_duplicates(inplace =True)
#count all the region name



corona['RegionName'].value_counts()
#total positive cases happened daily

df=corona.groupby('Date')['TotalPositiveCases'].sum()

df=df.reset_index()

df=df.sort_values('Date', ascending= True)

df.head(60)
#total positive cases in the region



df=corona.groupby('RegionName')['TotalPositiveCases'].sum()

df=df.reset_index()

df=df.sort_values('RegionName', ascending= True)

df.head(60)
#total positive cases in Italy

corona['TotalPositiveCases'].sum()
#checking the null values via graph , where you can find yellow color lines means that column contains null values.



sns.heatmap(corona.isnull(), yticklabels= False)
#plotting graph which region has maximum



sns.countplot(y=corona['RegionName'],).set_title('Regions affected overall')
#which country has most affected with corona

sns.countplot(x='Country',data=corona,hue='Country')
#which region code has highest affected in the country

sns.countplot(y='RegionCode', data=corona, hue='Country')
#which regioncode has highest affected

sns.countplot(y='RegionCode', data=corona, hue='RegionCode')
plt.figure(figsize=(10,8))

sns.countplot(y='RegionCode', data=corona, hue='Date')
plt.figure(figsize=(10,8))

sns.countplot(y='Country', data=corona, hue='Date')
plt.figure(figsize=(8,8))

sns.countplot(y='Country', data=corona, hue='TotalPositiveCases')
plt.figure(figsize=(10,8))

sns.countplot(y='Date', data=corona, hue='TotalPositiveCases')
plt.figure(figsize=(10,5))

Confirmed_positive_cases=corona['Date'].value_counts().sort_index()

Confirmed_positive_cases.cumsum().plot(legend='accumulated')

Confirmed_positive_cases.plot(kind='bar',color='orange',legend='daily',grid=True)
# cases confirmed before number 250 positive cases confirmed



plt.figure(figsize=(8,2))

df= Confirmed_positive_cases[:corona[corona['SNo']==250]['Date'].values[0]]

df.cumsum().plot(legend='accumulated')

df.plot(kind='bar',color='orange',legend='daily',grid=True)
# cases confirmed after number 250 positive cases confirmed

plt.figure(figsize=(10,5))

df= Confirmed_positive_cases[corona[corona['SNo']==250]['Date'].values[0]:]

df.cumsum().plot(legend='accumulated')

df.plot(kind='bar',color='orange',legend='daily',grid=True)