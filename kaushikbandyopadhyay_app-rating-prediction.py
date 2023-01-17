import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns
#load data into a df

df_app = pd.read_csv('/kaggle/input/googleplaystore 2.csv')

df_app.head()
#find out no of nulls per column

df_app.isnull().sum()
#dropping nulls value

df_app.dropna(axis=0,inplace=True)

#reset index since we dropped rows.

df_app.reset_index(drop=True,inplace=True)
df_app['Size'].tail(15)
#fix Size column by converting MB into KB and extracting numerical value

m_index = df_app['Size'].loc[df_app['Size'].str.contains('M')].index.tolist()

converter = pd.DataFrame(df_app.loc[m_index,'Size'].apply(lambda x: x.strip('M')).astype(float).apply(lambda x : x * 1000)).astype(str)

df_app.loc[m_index,'Size'] = converter
df_app['Size'].tail(10)
# Size cleaning

df_app['Size'] = df_app['Size'].apply(lambda x: x.strip('k'))

df_app[df_app['Size'] == 'Varies with device'] = 0

df_app['Size'] = df_app['Size'].astype(float)
df_app.dtypes
#change data type

df_app['Reviews'] = df_app['Reviews'].astype(int)
#remove '+' and ',' from numerical value

df_app['Installs'] = df_app['Installs'].astype(str).apply(lambda x: x.strip('\+'))
df_app['Installs'] = [int(i.replace(',','')) for i in df_app['Installs']]
df_app['Installs'].tail(15)
#Remove ‘$’ sign, and convert to numeric

df_app['Price'] = df_app['Price'].astype(str).apply(lambda x: x.strip('\$'))

df_app['Price'] = df_app['Price'].astype(float)
#Reviews should not be more than installs as only those who installed can review the app. If there are any such records, drop them.

df_app.drop(df_app[df_app['Reviews'] > df_app['Installs']].index,axis=0,inplace=True)
np.sum(df_app['Reviews'] > df_app['Installs'])
#For free apps (type = “Free”), the price should not be >0. Drop any such rows.

free = df_app.loc[df_app['Type'] == 'Free'].index
np.sum(df_app.loc[free,'Price'] > 0)
df_app['Price'].tail()
df_app.Price.plot.box()

#boxplot for price reveils that there are some apps with high price.
df_app.Reviews.plot.box()

#we observe that very few apps have high rating.
df_app.Rating.plot.hist()

#frequency of ratings are distributed more towards higher ratings
df_app.Size.plot.hist()

#frequency of apps with low size is more and decreases significantly as the app size increases
#From the box plot, it seems like there are some apps with very high price. Drop records above 200 as they maybe junk apps

df_app.drop(df_app[df_app['Price'] >=200].index,inplace=True)
#Very few apps have very high number of reviews. These are all star appsthat don’t help with the analysis and, in fact, will skew it. Drop records having more than 2 million reviews.

df_app.drop(df_app[df_app['Reviews'] > 2000000].index,inplace=True)
#different percentile for Installs and keeping only those records above a threshold

df_app.Installs.quantile([0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99])
df_app = df_app[df_app.Installs > 10000000.0].copy()
df_app.head()
sns.scatterplot(x="Rating", y="Size", data=df_app)

#its quite clear that heavy apps dont have a better rating.
sns.scatterplot(x="Rating", y="Reviews", data=df_app)

#more number of reviews does not contribute to better rating. Mid-range reviews have better ratings that high number of reviews.
sns.boxenplot(x="Rating", y="Content Rating", data=df_app)

#its clear that Content rating for Teen are liked better
sns.boxplot(x='Rating',y='Category',data=df_app)

#Game category is most popular having the better share of ratings.
#create a copy of data frame

inp1 = df_app.copy()
inp1.head()
inp1[['Installs','Reviews']]
inp1.info()
#drop unwanted cols

inp1.drop(columns = { 'App','Last Updated','Current Ver','Android Ver'},

inplace=True)
inp1.head()
#Apply log transformation (np.log1p) to Reviews and Installs.

inp1['Reviews'] = np.log1p(inp1['Reviews'])
inp1['Installs'] = np.log1p(inp1['Installs'])
inp1.head()
#Get dummy columns for Category, Genres, and Content Rating. This needs to be done as the models do not understand categorical data, and all data should be numeric.

dum_cols = ['Category','Genres','Content Rating']

inp2 = pd.get_dummies(inp1,columns=dum_cols,drop_first=True)

inp2
inp2.pop('Type')
#Train test split and apply 70-30 split.

from sklearn.model_selection import train_test_split

df_train, df_test = train_test_split(inp2, train_size = 0.7, random_state = 100)

y_train = df_train.Rating

X_train = df_train

y_test = df_test.Rating

X_test = df_test

from sklearn.linear_model import LinearRegression

lr = LinearRegression()

lr.fit(X_train, y_train)
#Reporting r2 for the model

from sklearn.metrics import r2_score

y_train_pred= lr.predict(X_train)

r2_score(y_train, y_train_pred)

#R2 of 1 indicates that the regression predictions perfectly fit the data.
y_test_pred= lr.predict(X_test)

r2_score(y_test, y_test_pred)

#R2 of 1 indicates that the regression predictions perfectly fit the data.