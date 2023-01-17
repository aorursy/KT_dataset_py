%matplotlib inline



# imports

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt
# importing data

data = pd.read_csv('../input/7003_1.csv', low_memory=False)
data.head(5)
data.shape
data.columns
data.describe(include='all')
data.dtypes
price_features = ['prices.amountMin', 'prices.amountMax']

data[price_features] = data[price_features].apply(pd.to_numeric, errors='coerce')
data['prices.amountMin'].describe()
fixed_min_price = data[(data['prices.amountMin'] > 5) & (data['prices.amountMin'] < 300)]

plt.figure(figsize=(8,8), dpi=100, facecolor='w', edgecolor='k')

fixed_min_price.groupby('brand')['prices.amountMin'].mean().plot(kind='bar')
fixed_max_price = data[(data['prices.amountMax'] > 5) & (data['prices.amountMax'] < 300)]

plt.figure(figsize=(8,6), dpi=100, facecolor='w', edgecolor='k')

fixed_max_price.groupby('brand')['prices.amountMax'].mean().plot(kind='bar')
most_expensve = data.sort_values(by='prices.amountMax', ascending=False).head(120).tail(100)

plt.figure(figsize=(8,6), dpi=100, facecolor='w', edgecolor='k')

most_expensve.groupby('brand')['prices.amountMax'].mean().plot(kind='bar')
fixed_max_price['prices.amountMax'].describe()
data.sort_values('prices.amountMax', ascending=False).head(10)
data['prices.color'].value_counts().sum()
data['features'].str.strip().value_counts().head(3)
def get_values(x):

    y={}

    error=0

    if type(x) is str and len(x) > 0:

        try:

            parsed_x = eval(x)

            for i in parsed_x:

                y[i['key']] = i['value'][0]

        except:

            error += 1

    return y

        



features = data['features'].apply(lambda x: get_values(x))
len(features)

features_df= pd.DataFrame(list(features))

len(features_df.columns)
features_df = features_df.groupby(features_df.columns, axis=1).sum()
features_df.describe(include='all')
features_df.isnull().sum().sort_values().head(10)
features_df['Gender'].value_counts()
features_df['Color'].value_counts()
features_df['UV Rating'].value_counts()
features_df['Age Group'].value_counts()
features_df['yearBuilt'].value_counts()
features_df['Fabric Material'].value_counts()
data['prices.currency'].value_counts()
data['manufacturer'].value_counts().sum()
data['categories'].value_counts().head(10)
data['flavors'].value_counts().sum()
data['keys'].value_counts().sum()
data['manufacturerNumber'].value_counts().sum()
data['merchants'].value_counts().sum()