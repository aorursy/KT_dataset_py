# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import plotly.plotly as py

import seaborn as sns

import missingno as msno

from wordcloud import WordCloud

import nltk

from nltk.tokenize import word_tokenize

from nltk.corpus import stopwords

from catboost import CatBoostRegressor

from sklearn.preprocessing import MinMaxScaler

from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import train_test_split



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
df = pd.read_csv('../input/winemag-data-130k-v2.csv')

df.info()
df.head()
df.describe()
df.duplicated(['title']).sum()
dfUW = df.drop_duplicates(subset=['title'])

dfUW.info()
df.drop(['Unnamed: 0'], axis=1, inplace=True)

df.country = df.country.astype('category')

df.designation = df.designation.astype('category')

df.province = df.province.astype('category')

df.region_1 = df.region_1.astype('category')

df.region_2 = df.region_2.astype('category')

df.variety = df.variety.astype('category')

df.winery = df.winery.astype('category')
df.isna().sum()
msno.matrix(df)
fig, ax = plt.subplots(figsize=(4,8))

sns.boxplot(y=df.points)
fig, ax = plt.subplots(figsize=(6,4))

sns.distplot(df.points)
fig, ax = plt.subplots(figsize=(4,8))

sns.boxplot(y=df.price)
fig, ax = plt.subplots(figsize=(12,6))

sns.distplot(df.price.dropna())
sns.jointplot("price", "points", data=df, kind="reg")
df.price.quantile(0.01)
df.price.quantile(0.99)
df = df[df.price <= 1500]

sns.jointplot("price", "points", data=df, kind="reg")
ccounts = dfUW.country.value_counts()
fig, ax = plt.subplots(figsize=(12,6))

sns.distplot(ccounts)
dfUW.country.value_counts().plot(kind='bar', fontsize=14, figsize=(16,6))
g = sns.catplot(y="country", x="points", kind='box', data=dfUW)

g.fig.set_size_inches(8,10)
g = sns.catplot(y="country", x="points", kind='box', data=dfUW[dfUW.points >= dfUW.points.quantile(0.9)])

g.fig.set_size_inches(8,10)
pcounts = dfUW.province.value_counts()

p10popular = pcounts.nlargest(10).index.tolist()
fig, ax = plt.subplots(figsize=(12,6))

sns.distplot(pcounts)
pcounts[pcounts >= pcounts.quantile(0.9)].plot(kind='bar', fontsize=14, figsize=(16,6))
g = sns.catplot(y="province", x="points", kind='box', data=dfUW[dfUW.points >= dfUW.points.quantile(0.95)])

g.fig.set_size_inches(12,15)
g = sns.catplot(y="province", x="points", kind='box', data=dfUW[dfUW.province.isin(p10popular)])

g.fig.set_size_inches(12,15)
dfUW['provincePopularity'] = pd.Series(['popular', 'non-popular'], dtype='category')

ppopular = pcounts[pcounts.isin(pcounts.nlargest(15))].index.tolist()

dfUW.loc[dfUW.province.isin(ppopular), 'provincePopularity'] = 'popular'

dfUW.loc[~dfUW.province.isin(ppopular), 'provincePopularity'] = 'non-popular'

g = sns.catplot(x="provincePopularity", y="points", kind='box', data=dfUW)

g.fig.set_size_inches(4,4)
vcounts = dfUW.variety.value_counts()

v10popular = vcounts.nlargest(10).index.tolist()

v10rare = vcounts.nsmallest(10).index.tolist()
fig, ax = plt.subplots(figsize=(12,6))

sns.distplot(vcounts)
vcounts[vcounts >= vcounts.quantile(0.9)].plot(kind='bar', fontsize=14, figsize=(16,6))
dfUW['varietyPopularity'] = pd.Series(['popular', 'non-popular'], dtype='category')

vpopular = vcounts[vcounts.isin(vcounts.nlargest(15))].index.tolist()

dfUW.loc[dfUW.variety.isin(vpopular), 'varietyPopularity'] = 'popular'

dfUW.loc[~dfUW.variety.isin(vpopular), 'varietyPopularity'] = 'non-popular'

g = sns.catplot(x="varietyPopularity", y="points", kind='box', data=dfUW)

g.fig.set_size_inches(4,4)
wcounts = dfUW.winery.value_counts()
fig, ax = plt.subplots(figsize=(12,6))

sns.distplot(wcounts)
dfUW['wineryPopularity'] = pd.Series(['popular', 'non-popular'], dtype='category')

wpopular = wcounts[wcounts.isin(wcounts.nlargest(10))].index.tolist()

dfUW.loc[dfUW.winery.isin(wpopular), 'wineryPopularity'] = 'popular'

dfUW.loc[~dfUW.winery.isin(wpopular), 'wineryPopularity'] = 'non-popular'

dfUW.wineryPopularity.value_counts()

g = sns.catplot(x="wineryPopularity", y="points", kind='box', data=dfUW)

g.fig.set_size_inches(4,4)
df.description =  df.description.str.lower()

reviews = df.description.str.cat(sep=' ')



stop_words = stopwords.words('english')

stop_words.append('the')

stop_words.append('it')

stop_words = set(stop_words)



tokens = word_tokenize(reviews)

tokens = [w for w in tokens if not w in stop_words]



vocabulary = set(tokens)



frequency_dist = nltk.FreqDist(tokens)

sorted(frequency_dist,key=frequency_dist.__getitem__, reverse=True)



wordcloud = WordCloud(background_color='white').generate_from_frequencies(frequency_dist)



plt.imshow(wordcloud)

plt.axis("off")

plt.show()
features = ['country','province', 'winery', 'variety', 'price']

target = ['points']



data = dfUW[features + target].dropna()



X_train, X_test, y_train, y_test = train_test_split(data[features], data['points'], test_size=0.33, random_state=42)



model=CatBoostRegressor(iterations=100, depth=3, learning_rate=0.1, loss_function='RMSE')

model.fit(X_train, y_train,cat_features=[0,1,2,3], eval_set=(X_test, y_test), plot=True)

plt.show()