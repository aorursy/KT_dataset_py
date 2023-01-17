import pandas as pd

import plotly as py

import seaborn as sns

import matplotlib.pyplot as plt





%matplotlib inline
csv_data = pd.read_csv('../input/renfe.csv', delimiter=',')

type(csv_data['origin'])
csv_data.head()
csv_data.shape
csv_data.count()
csv_data.isnull().sum()
cleandf = csv_data
cleandf['fare'].fillna(value='Promo', inplace=True)
cleandf['fare'].value_counts(dropna=False)
cleandf['train_class'].fillna(value='Turista', inplace=True)
cleandf['train_class'].value_counts(dropna=False)
price_mean = round(cleandf['price'].mean(), 2)

price_mean
cleandf['price'].fillna(value=price_mean, inplace=True)
cleandf['price'].value_counts(dropna=False).head(5)
round(cleandf['price'].mean(),2)
cleandf.isnull().sum()
cleandf = cleandf.drop(['Unnamed: 0','insert_date'], axis = 1)
print(cleandf['origin'].unique())

print(len(cleandf['origin'].unique()))
print(cleandf['destination'].unique())

print(len(cleandf['destination'].unique()))
connection = cleandf['origin'] + '-' + cleandf['destination']
print(connection.unique())

print(len(connection.unique()))
connection.value_counts()
cleandf['connection'] = connection
cleandf.drop(['origin','destination'],axis=1,inplace=True)
cleandf.columns
cleandf = cleandf[['connection', 'start_date', 'end_date', 'train_type', 'price', 'train_class', 'fare']]

       
cleandf.head()
cleandf.dtypes
cleandf['start_date'] = pd.to_datetime(cleandf.start_date)
cleandf['end_date'] = pd.to_datetime(cleandf.end_date)
cleandf.dtypes
time_travel = cleandf['end_date'] - cleandf['start_date']
cleandf.head()
cleandf1 = cleandf
cleandf1['time_travel'] = time_travel
cleandf1.drop(['start_date','end_date'], axis=1, inplace=True)
cleandf1.columns
cleandf1 = cleandf1[['connection', 'time_travel', 'train_type', 'train_class', 'fare', 'price']]
cleandf1.head()
cleandf1['price'].value_counts(dropna=False).head(5)
timetravel = cleandf1['time_travel'].value_counts(dropna=False).head(10)
df = cleandf1

df.dtypes

df['price'].hist()

plt.figure()
sns.catplot(x='train_type',

            kind='count',

            order=df.train_type.value_counts().index,

            aspect=2,

            data=df)
sns.set(style='whitegrid', rc={'figure.figsize':(17, 5)})

ax = sns.boxplot(x='train_type', y='price', data=df, fliersize=0.5, width=0.5, )

plt.show()
sns.catplot(x='train_class',

            kind='count',

            order=df.train_class.value_counts().index,

            aspect=2,

            data=df)
sns.set(style='whitegrid', rc={'figure.figsize':(15, 5)})

ax = sns.boxplot(x='train_class', y='price', data=df, fliersize=0.5, width=0.5, )

plt.show()
sns.catplot(x='fare',

            kind='count',

            order=df.fare.value_counts().index,

            aspect=1.5,

            data=df)
sns.set(style='whitegrid', rc={'figure.figsize':(10, 5)})

ax = sns.boxplot(x='fare', y='price', data=df, fliersize=0.5, width=0.5, )

plt.show()