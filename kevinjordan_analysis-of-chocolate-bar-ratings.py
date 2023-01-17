import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns
data = pd.read_csv("../input/flavors_of_cacao.csv")

print(data.shape)

print(data.head())

print(data.columns.values)
data.columns =data.columns.str.replace('\n', ' ').str.replace('\xa0', '')

print(data.info())
#print(data.isnull().sum())

data.fillna(0,inplace=True)

print(data.isnull().sum())
data['Cocoa Percent']=data['Cocoa Percent'].apply(lambda x: x[:-1]).astype('float')
data.info()
print(data.corr())



sns.heatmap(data.corr())
data.columns.values
print(data['Rating'].value_counts().sort_values(ascending=True))

sns.countplot(x ='Rating', data=data)
rating_BroadBeanOrigin_median = data.groupby(["Broad Bean Origin"])['Rating'].median()

print(rating_BroadBeanOrigin_median.sort_values(ascending=False).head(10))

rating_BroadBeanOrigin_median.sort_values(ascending=False).head(10).plot('barh')
rating_company_median= data.groupby(["Company (Maker-if known)"])['Rating'].median()

print(rating_company_median.sort_values(ascending=False).head(10))

rating_company_median.sort_values(ascending=False).head(10).plot('barh')
sns.kdeplot(data["Cocoa Percent"],color="green", shade=True)

sns.kdeplot(data["Rating"],color="green", shade=True)

print(data['Rating'].max())
sns.kdeplot(data["REF"],color="green", shade=True)

print(data['REF'].max())
print(data.info())

sns.kdeplot(data["Review Date"],color="green", shade=True)