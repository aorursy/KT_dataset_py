import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

pd.set_option('display.max_columns', None)

pd.set_option('display.max_rows', 20)

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
data = pd.read_csv('/kaggle/input/melb_data.csv')
data.head()
data.info()
data.describe()
data.isnull().sum().sort_values(ascending = False)

missing_values  = data.isnull().sum().sort_values(ascending = False)

missing_df = pd.concat([missing_values], axis = 1, keys = ['Total'])

f, ax = plt.subplots(figsize = (12,6))

sns.barplot(x = missing_values.index, y = missing_values)

plt.xticks(rotation = 90)

plt.title('Missing Values in Features')

plt.ylabel('Total missing values')
cat_features = ['Suburb', 'Address','Type', 'Method','SellerG','Postcode','CouncilArea','Regionname']

data[cat_features] = data[cat_features].astype('category')
pd.to_datetime(data.YearBuilt, format='%Y', errors = 'coerce').dt.to_period('Y')
pd.to_datetime(data.Date, infer_datetime_format=True)
timeline_features = ['Yearbuilt', 'Date']
continous_features = []

for col in data:

    if col not in cat_features and col not in timeline_features:

        continous_features.append(col)
plt.subplots(figsize = (9,6))

sns.heatmap(data[continous_features].corr(), annot = True, center = 0)
data[continous_features].corr().Price[(data[continous_features].corr().Price >= 0.4) | (data[continous_features].corr().Price < -0.1)]
corrdf_ = data[continous_features].corr().Price[(data[continous_features].corr().Price >= 0.4) | (data[continous_features].corr().Price < 0)]

corr_df = pd.concat([corrdf_], axis = 1, keys = ['Corr']).sort_values(by = 'Corr', ascending = False)

indexlist =[]

for x in corr_df.index:

    indexlist.append(x)
sns.pairplot(data[indexlist] )
data.YearBuilt = data.YearBuilt[data.YearBuilt > 1800]
sns.relplot(x = 'Date', y ='Price', kind= 'line', data = data, aspect = 3)

plt.xticks(rotation =90)
f,ax=plt.subplots(figsize = (17,7))

sns.regplot(x = 'YearBuilt', y ='Price', data = data, ax = ax)

plt.xticks(rotation =90)
sns.relplot(y = 'YearBuilt', x ='Date',hue = 'Price' ,kind= 'scatter', data = data, aspect = 3)

plt.xticks(rotation =90)
data['Age'] = pd.DatetimeIndex(data['Date']).year - data.YearBuilt # year of selling - year built
f,ax=plt.subplots(figsize = (17,7))

sns.regplot(x = 'Age', y = 'Price' ,data = data, ax = ax)

plt.xticks(rotation =90)
for cat in cat_features:

    print(cat , data[cat].nunique())
for cat in cat_features:

    try:

        if data[cat].nunique() < 15:

            catdf = data[cat][data.Price >=  data.Price.quantile(0.75)].value_counts()

            listof_index = []

            for index in catdf.index:

                listof_index.append(index)

            df = data[(data[cat].isin(listof_index))  &  (data.Price >=  data.Price.quantile(0.75))][[cat,'Price','Method']]

            sns.catplot(x= cat ,y='Price', data = df, kind = 'swarm', hue = 'Method' )

            plt.xticks(rotation = 90)



        else:

            catdf = data[cat][data.Price >=  data.Price.quantile(0.75)].value_counts().iloc[:14]

            listof_index = []

            for index in catdf.index:

                listof_index.append(index)

            df = data[(data[cat].isin(listof_index))  &  (data.Price >=  data.Price.quantile(0.75))][[cat,'Price','Method']]

            sns.relplot(x= cat ,y='Price', data = df ,hue = 'Method')

            plt.xticks(rotation = 90)

    except:

        if data[cat].nunique() < 15:

            catdf = data[cat][data.Price >=  data.Price.quantile(0.75)].value_counts()

            listof_index = []

            for index in catdf.index:

                listof_index.append(index)

            df = data[(data[cat].isin(listof_index))  &  (data.Price >=  data.Price.quantile(0.75))][[cat,'Price']]

            sns.catplot(x= cat ,y='Price', data = df, kind = 'swarm' )

            plt.xticks(rotation = 90)



        else:

            catdf = data[cat][data.Price >=  data.Price.quantile(0.75)].value_counts().iloc[:14]

            listof_index = []

            for index in catdf.index:

                listof_index.append(index)

            df = data[(data[cat].isin(listof_index))  &  (data.Price >=  data.Price.quantile(0.75))][[cat,'Price']]

            sns.relplot(x= cat ,y='Price', data = df )

            plt.xticks(rotation = 90)