# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.

import matplotlib.pyplot as plt

import seaborn as sns
df = pd.read_csv('../input/craigslist-carstrucks-data/craigslistVehicles.csv')
# First 5 rows of our data

df.head()
#Delete some columns

df = df.drop(columns=['image_url', 'lat', 'long', 'city_url', 'desc', 'city', 'VIN'])
#Find and delete duplicates

df.drop_duplicates(subset='url')

df.shape
df[df.isnull().sum(axis=1) < 9].shape, df[df.isnull().sum(axis=1) >= 9].shape
#Let's leave lines with less than 9 missing values

df = df[df.isnull().sum(axis=1) < 9]

df.shape
#let's take a look how many missing values we have in our dataset

df.isnull().sum()
df[df.price == 0].shape
df = df[df.price != 0]

df.shape
plt.figure(figsize=(8, 8))

sns.boxplot(y= 'price', data=df)
#delete data with prices above 100k

df = df[df.price < 100000]

df.shape
plt.figure(figsize=(8, 10))

sns.boxplot(y= 'price', data=df)
plt.figure(figsize=(15, 13))

year_plot = sns.countplot(x = 'year', data=df)

year_plot.set_xticklabels(year_plot.get_xticklabels(), rotation=90,fontsize=8);
df = df[df.year > 1985]

df.shape
plt.figure(figsize=(15, 13))

ax = sns.countplot(x = 'year', data=df, palette='Set1')

ax.set_xticklabels(ax.get_xticklabels(), rotation=90,fontsize=10);
df.odometer.quantile(.999)
df = df[~(df.odometer > 500000)]

df.shape
plt.figure(figsize = (8, 12))

sns.boxplot(y = 'odometer', data = df[~(df.odometer > 500000)])
sns.set(style="ticks", color_codes='palette')

sns.pairplot(df, hue= 'condition');
df.isnull().sum()
mean= df[df['year'] == 2010]['odometer'].mean()

mean
df.odometer = df.groupby('year')['odometer'].apply(lambda x: x.fillna(x.mean()))
df.odometer.isnull().sum()
df['condition'].isnull().sum()
df.loc[(df['year'] >= 2019)]['condition'].isnull().sum()
df.loc[df.year>=2019, 'condition'] = df.loc[df.year>=2019, 'condition'].fillna('new')
df.loc[(df['year'] >= 2019)]['condition'].isnull().sum()
df['condition'].unique()
excellent_odo_mean = df[df['condition'] == 'excellent']['odometer'].mean()

good_odo_mean = df[df['condition'] == 'good']['odometer'].mean()

like_new_odo_mean = df[df['condition'] == 'like new']['odometer'].mean()

salvage_odo_mean = df[df['condition'] == 'salvage']['odometer'].mean()

fair_odo_mean = df[df['condition'] == 'fair']['odometer'].mean()

print('excelent {}, good {}, like_new {}, salvage {}, fair {}'.format(excellent_odo_mean, good_odo_mean,

                                                                like_new_odo_mean, salvage_odo_mean,

                                                                fair_odo_mean))
df.loc[df['odometer'] <= like_new_odo_mean, 'condition'] = df.loc[df['odometer'] <= like_new_odo_mean, 'condition'].fillna('like new')

df.loc[df['odometer'] >= fair_odo_mean, 'condition'] = df.loc[df['odometer'] >= fair_odo_mean, 'condition'].fillna('fair')

df.loc[((df['odometer'] > like_new_odo_mean) & 

       (df['odometer'] <= excellent_odo_mean)), 'condition'] = df.loc[((df['odometer'] > like_new_odo_mean) & 

       (df['odometer'] <= excellent_odo_mean)), 'condition'].fillna('excellent')

df.loc[((df['odometer'] > excellent_odo_mean) & 

       (df['odometer'] <= good_odo_mean)), 'condition'] = df.loc[((df['odometer'] > excellent_odo_mean) & 

       (df['odometer'] <= good_odo_mean)), 'condition'].fillna('good')

df.loc[((df['odometer'] > good_odo_mean) & 

       (df['odometer'] <= fair_odo_mean)), 'condition'] = df.loc[((df['odometer'] > good_odo_mean) & 

       (df['odometer'] <= fair_odo_mean)), 'condition'].fillna('salvage')
df.isnull().sum()
df['cylinders'].unique()
df['cylinders'].value_counts().head()
df['cylinders'].isnull().sum()
df['cylinders'] = df['cylinders'].fillna(df['cylinders'].value_counts().index[0])
df['transmission'] = df['transmission'].fillna(df['transmission'].value_counts().index[0])

df['title_status'] = df['title_status'].fillna(df['title_status'].value_counts().index[0])

df['fuel'] = df['fuel'].fillna(df['fuel'].value_counts().index[0])

df['size'] = df['size'].fillna(df['size'].value_counts().index[0])
df = df.dropna(subset=['make'])
df = df.fillna('Unkown')
df.isnull().sum()
df = df.drop(columns=['url'])
df.head()
from sklearn.preprocessing import LabelEncoder
labels = ['manufacturer', 'make', 'condition', 'cylinders', 'fuel', 'title_status', 'transmission', 

          'drive', 'size', 'type', 'paint_color']

les = {}



for l in labels:

    les[l] = LabelEncoder()

    les[l].fit(df[l])

    tr = les[l].transform(df[l]) 

    df.loc[:, l + '_feat'] = pd.Series(tr, index=df.index)



labeled = df[ ['price'

                ,'odometer'

                ,'year'] 

                 + [x+"_feat" for x in labels]]
labeled.head()
X = labeled.drop(columns=['price'])

y = labeled['price']

print(X.shape, y.shape)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state=42)

print(X_train.shape, X_test.shape)
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()  

scaler.fit(X_train)    

X_train_normed = pd.DataFrame(scaler.transform(X_train))

X_test_normed = pd.DataFrame(scaler.transform(X_test))
from sklearn.ensemble import RandomForestRegressor
rdf = RandomForestRegressor()

rdf.fit(X_train_normed, y_train)
from sklearn.metrics import mean_squared_error as MSE
y_pred = rdf.predict(X_test_normed)

rmse2 = np.sqrt(MSE(y_test, y_pred))

print("RMSE = {:.2f}".format((rmse2)))
accuracy = rdf.score(X_test_normed,y_test)

print(accuracy*100,'%')
from sklearn.model_selection import GridSearchCV

parameters = {'n_estimators': [100]}
clf = GridSearchCV(rdf, parameters, cv=5, n_jobs= -1)
clf.fit(X_train_normed, y_train)
clf.best_estimator_
y_pred = clf.best_estimator_.predict(X_test_normed)

rmse2 = np.sqrt(MSE(y_test, y_pred))

print("RMSE = {:.2f}".format(rmse2))

accuracy = clf.score(X_test_normed,y_test)

print(accuracy*100,'%')