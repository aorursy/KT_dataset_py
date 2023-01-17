# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
df15 = pd.read_csv('../input/2015.csv')
df16 = pd.read_csv('../input/2016.csv')
df17 = pd.read_csv('../input/2016.csv')
df15.head()
print('shapes: 2015 = {}, 2016 = {}, 2017 = {}'.format(df15.shape, df16.shape, df17.shape))
df15.sort_values('Happiness Score', ascending=False).head(5)
df15.set_index('Country', inplace=True)
df16.set_index('Country', inplace=True)
df17.set_index('Country', inplace=True)
df15.isnull().sum()
df15.dtypes
fig1, ax1 = plt.subplots(ncols=3, figsize=(14,5))
data = [df15, df16, df17]
year = ['2015', '2016', '2017']
for i, df in enumerate(data):
    p1 = df['Happiness Score'].sort_values(ascending=False)[:4]
    p2 = df['Happiness Score'].sort_values(ascending=False)[-4:]
    both = p1.append(p2)
    both.plot(kind='bar', ax=ax1[i])
    ax1[i].set_title('Happiest and Unhappiest Countries {}'.format(year[i]))
ax1[0].set_ylabel('Happiness Score')
df15.loc['Thailand', ['Happiness Score','Happiness Rank']]
sns.pairplot(df15, vars=['Happiness Score', 'Economy (GDP per Capita)', 'Family', 'Freedom', 'Generosity'], kind='reg')
sns.heatmap(df15.drop(['Happiness Rank', 'Standard Error'], axis=1).corr(), annot=True, fmt='.1f')
sns.stripplot(x='Region', y='Freedom', data=df15)
plt.xticks(rotation=90)
sns.boxplot(x='Region', y='Generosity', data=df15)
plt.xticks(rotation=90)
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

model = LinearRegression()
df15.columns
predictors = list(df15.drop(['Region','Happiness Rank', 'Standard Error', 'Happiness Score', 'Dystopia Residual'],axis=1).columns)
X15 = df15[predictors].copy()
y15 = df15['Happiness Score']
X15train, X15test, y15train, y15test = train_test_split(X15, y15, test_size=0.3, random_state=42)
model.fit(X15train,y15train)
model.coef_
preds = model.predict(X15test)
mean_absolute_error(y15test,preds)
plt.scatter(y15test,preds)
plt.xlabel('data')
plt.ylabel('prediction')
plt.title('Predicted Happiness Score vs Actual Score 2015')
