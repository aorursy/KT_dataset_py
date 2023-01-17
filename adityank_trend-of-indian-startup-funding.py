#Render Matplotlib Plots Inline

%matplotlib inline



#Import the standard Python Scientific Libraries

import pandas as pd

import numpy as np

from scipy import stats

import matplotlib.pyplot as plt

import seaborn as sns



#Suppress Deprecation and Incorrect Usage Warnings 

import warnings

warnings.filterwarnings('ignore')



#Load MCQ Responses into a Pandas DataFrame

data = pd.read_csv('../input/startup_funding.csv', encoding="ISO-8859-1", low_memory=False)
data['year'] = pd.to_datetime(data['Date'], errors='coerce').apply(lambda x: str(x).split('-')[0] if x != np.nan else np.nan)
plt.figure(figsize=(10, 5))

sns.countplot(data=data,x='year')

a = data[['StartupName','IndustryVertical']].groupby('IndustryVertical').count().sort_values('StartupName',ascending=False).head(8)

a.reset_index(inplace=True)

plt.pie(a['StartupName'],labels=a['IndustryVertical'])

plt.show()
b= data.SubVertical.value_counts().sort_values(ascending=False).head(15)

b.plot(kind='barh',figsize=(15, 10))
location = data.CityLocation.value_counts().head(5)

location.plot(kind='barh',figsize=(10, 5))
InvestorsName = data.InvestorsName.value_counts().head(10)

InvestorsName.plot(kind='barh',figsize=(15, 10))
InvestmentType  = data.InvestmentType.value_counts().head(5)

InvestmentType.plot(kind='barh',figsize=(15, 5))
AmountInUSD = data.AmountInUSD.value_counts().head(5)

AmountInUSD.plot(kind='barh',figsize=(15, 5))
amount = data[['StartupName','AmountInUSD']].groupby('AmountInUSD').count().sort_values('StartupName',ascending=False).head(10)

amount