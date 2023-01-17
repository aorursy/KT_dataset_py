# Loading packages

import pandas as pd #Analysis 

import matplotlib.pyplot as plt #Visulization

import seaborn as sns #Visulization

import numpy as np #Analysis 

from scipy.stats import norm #Analysis 

from sklearn.preprocessing import StandardScaler #Analysis 

from scipy import stats #Analysis 

import warnings 

warnings.filterwarnings('ignore')

%matplotlib inline

import gc

import plotly.graph_objs as go

import plotly.offline as py

from plotly import tools
df_train = pd.read_csv('../input/train.csv')

df_test  = pd.read_csv('../input/test.csv')
print("train.csv. Shape: ",df_train.shape)

print("test.csv. Shape: ",df_test.shape)
df_train.head(10)
df_test.head(10)
f, ax = plt.subplots(figsize=(8, 6))

sns.distplot(df_train['price'])
df_train['price'] = np.log1p(df_train['price'])

f, ax = plt.subplots(figsize=(8, 6))

sns.distplot(df_train['price'])
df_train["bedrooms"].drop_duplicates()
df_train["bathrooms"].drop_duplicates()
data = pd.concat([df_train['price'], df_train['sqft_living']], axis=1)

f, ax = plt.subplots(figsize=(8, 6))

fig = sns.regplot(x='sqft_living', y="price", data=data)
df_train[df_train["sqft_living"]>13000]
df_train["floors"].drop_duplicates()
fig, ax = plt.subplots(figsize=(8, 6))

sns.boxplot(x="floors", y="price", data=df_train)

plt.title("Box Plot")

plt.show()
df_train["waterfront"].drop_duplicates()
fig, ax = plt.subplots(figsize=(8, 6))

sns.boxplot(x="waterfront", y="price", data=df_train)

plt.title("Box Plot")

plt.show()
df_train["view"].drop_duplicates()
fig, ax = plt.subplots(figsize=(8, 6))

sns.boxplot(x="view", y="price", data=df_train)

plt.title("Box Plot")

plt.show()
df_train["condition"].drop_duplicates()
fig, ax = plt.subplots(figsize=(8, 6))

sns.boxplot(x="condition", y="price", data=df_train)

plt.title("Box Plot")

plt.show()
df_train["grade"].drop_duplicates()
fig, ax = plt.subplots(figsize=(8, 6))

sns.boxplot(x="grade", y="price", data=df_train)

plt.title("Box Plot")

plt.show()
df_train["yr_built"].describe()
df_train[df_train["yr_renovated"]!=0]["yr_renovated"].describe()
df_train.plot(kind = "scatter", x = "long", y = "lat", alpha = 0.1, s = df_train["sqft_living"]*0.02, 

             label = "sqft_living", figsize = (10, 8), c = "price", cmap = plt.get_cmap("jet"), colorbar = True, sharex = False)
df_train[["sqft_living","sqft_lot","sqft_living15","sqft_lot15","yr_renovated","lat","long"]].head(30)
from scipy.stats import spearmanr



df_train_noid = df_train.drop("id",1)



plt.figure(figsize=(21,21))

sns.set(font_scale=1.25)

sns.heatmap(df_train_noid.corr(method='spearman'),fmt='.2f', annot=True, square=True , annot_kws={'size' : 15})
cor = df_train_noid.corr(method='spearman')

cor["price"].nlargest(n=20).index
df_train[df_train["sqft_living"]>13000]
data = pd.concat([df_train['price'], df_train['grade']], axis=1)

f, ax = plt.subplots(figsize=(8, 6))

fig = sns.boxplot(x='grade', y="price", data=data)
df_train[df_train["grade"]==3]
df_train.loc[(df_train['price']>14.7) & (df_train['grade'] == 8)]
df_train.loc[(df_train['price']>15.5) & (df_train['grade'] == 11)]
df_train.plot(kind = "scatter", x = "long", y = "lat", alpha = 0.1, s = df_train["sqft_living"]*0.02, 

             label = "sqft_living", figsize = (10, 8), c = "price", cmap = plt.get_cmap("jet"), colorbar = True, sharex = False)
df_train = df_train.loc[df_train['id']!=2302]
skew_columns = ['bedrooms', 'sqft_living', 'sqft_lot', 'sqft_above', 'sqft_basement']



for c in skew_columns:

    df_train[c] = np.log1p(df_train[c].values)

    df_test[c] = np.log1p(df_test[c].values)
for df in [df_train,df_test]:

    df['date'] = df['date'].apply(lambda x: x[0:8])
df_train.head(4)
a = [(1,3)]

a
for df in [df_train,df_test]:

    df['total_rooms'] = df['bedrooms'] + df['bathrooms']

    df['sqft_ratio'] = df['sqft_living'] / df['sqft_lot']

    df['sqft_ratio15'] = df['sqft_living15'] / df['sqft_lot15']

    df['been_renovated'] = df['yr_renovated'].apply(lambda x: 0 if x == 0 else 1)

    df['date'] = df['date'].astype('int')

for df in [df_train,df_test]:

    df['location'] = zip(df['lat'],df['long'])# 위치 변수
df_train["location"][1]