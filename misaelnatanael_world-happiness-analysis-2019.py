import numpy as np

import pandas as pd

from pandas import Series, DataFrame

import matplotlib.pyplot as plt

import seaborn as sns



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
df1 = pd.read_csv('../input/world-happiness/2019.csv')

df1 = df1.rename(columns={"Country or region": "Country"})

df1
df2 = pd.read_csv('../input/world-happiness/2015.csv')

df = pd.merge(df1, df2, on='Country', how='left')

df.drop(df.columns[10:],axis=1,inplace=True)

df = df.rename(columns={"Generosity_x": "Generosity"})

cols = ['Overall rank','Country','Region','Score','GDP per capita','Social support','Healthy life expectancy',

        'Freedom to make life choices','Generosity','Perceptions of corruption']

df = df[cols]

df
by_region = df.groupby('Region').agg(Mean_Score=('Score', 'mean')).reset_index()

plt.figure(figsize=(20,10))

sns.barplot(x='Mean_Score', y='Region', data=by_region.sort_values(by='Mean_Score',ascending=False))

plt.title('Mean Happiness Score in Each Region (2019)')
df[df['Region']=='Australia and New Zealand']
df[df['Region']=='North America']
from sklearn.linear_model import LinearRegression

x,y = df['GDP per capita'], df['Score']

X = np.array(x).reshape(-1,1)

plt.figure(figsize=(10,7))

sns.scatterplot(x, y)

plt.title('Relation between Happiness Score and GDP per Capita')



lr = LinearRegression()

lr.fit(X,y)

plt.plot(x, y, '.')

plt.plot(x, lr.intercept_ + lr.coef_ * x, '-')

plt.text(1.4, 4.5, 'R-squared = %0.2f' % lr.score(X,y))

plt.show()
x,y = df['Social support'], df['Score']

X = np.array(x).reshape(-1,1)

plt.figure(figsize=(10,7))

sns.scatterplot(x, y)

plt.title('Relation between Happiness Score and Social Support')



lr = LinearRegression()

lr.fit(X,y)

plt.plot(x, y, '.')

plt.plot(x, lr.intercept_ + lr.coef_ * x, '-')

plt.text(0.3, 2.4, 'R-squared = %0.2f' % lr.score(X,y))

plt.show()
x,y = df['Healthy life expectancy'], df['Score']

X = np.array(x).reshape(-1,1)

plt.figure(figsize=(10,7))

sns.scatterplot(x, y)

plt.title('Relation between Happiness Score and Healthy Life')



lr = LinearRegression()

lr.fit(X,y)

plt.plot(x, y, '.')

plt.plot(x, lr.intercept_ + lr.coef_ * x, '-')

plt.text(0.9, 4, 'R-squared = %0.2f' % lr.score(X,y))

plt.show()
x,y = df['Freedom to make life choices'], df['Score']

X = np.array(x).reshape(-1,1)

plt.figure(figsize=(10,7))

sns.scatterplot(x, y)

plt.title('Relation between Happiness Score and Freedom of Life')



lr = LinearRegression()

lr.fit(X,y)

plt.plot(x, y, '.')

plt.plot(x, lr.intercept_ + lr.coef_ * x, '-')

plt.text(0, 4.2, 'R-squared = %0.2f' % lr.score(X,y))

plt.show()
x,y = df['Generosity'], df['Score']

X = np.array(x).reshape(-1,1)

plt.figure(figsize=(10,7))

sns.scatterplot(x, y)

plt.title('Relation between Happiness Score and Generosity')



lr = LinearRegression()

lr.fit(X,y)

plt.plot(x, y, '.')

plt.plot(x, lr.intercept_ + lr.coef_ * x, '-')

plt.text(0.47, 5.9, 'R-squared = %0.2f' % lr.score(X,y))

plt.show()
x,y = df['Perceptions of corruption'], df['Score']

X = np.array(x).reshape(-1,1)

plt.figure(figsize=(10,7))

sns.scatterplot(x, y)

plt.title('Relation between Happiness Score and Perceptions of Corruption')



lr = LinearRegression()

lr.fit(X,y)

plt.plot(x, y, '.')

plt.plot(x, lr.intercept_ + lr.coef_ * x, '-')

plt.text(0.31, 6, 'R-squared = %0.2f' % lr.score(X,y))

plt.show()
six_vars = df[['Score','GDP per capita','Social support','Healthy life expectancy','Freedom to make life choices','Perceptions of corruption','Generosity']]

plt.figure(figsize=(10,8))

sns.heatmap(six_vars.corr(method='pearson'), cmap = 'RdBu_r', annot = True)

plt.show()
x,y = df['GDP per capita'], df['Healthy life expectancy']

X = np.array(x).reshape(-1,1)

plt.figure(figsize=(10,7))

sns.scatterplot(x, y)

plt.title('Relation between Health and GDP per Capita')



lr = LinearRegression()

lr.fit(X,y)

plt.plot(x, y, '.')

plt.plot(x, lr.intercept_ + lr.coef_ * x, '-')

plt.text(1.3, 0.5, 'R-squared = %0.2f' % lr.score(X,y))

plt.show()
X = (six_vars.dropna().iloc[:,1:7])

y = (six_vars.dropna().iloc[:,0])

lr = LinearRegression()

lr.fit(X,y)

print("multiple linear regression R-squared: %f" % (lr.score(X,y)))
from sklearn.feature_selection import SelectKBest

from sklearn.feature_selection import f_regression

fs = SelectKBest(score_func=f_regression, k='all')

fs.fit(X,y)

indices = np.argsort(fs.scores_)[::-1]

for f in range(X.shape[1]):

    print("%d. feature %s (%f)" % (f + 1, six_vars.columns[indices[f]+1], fs.scores_[indices[f]]))



cc = DataFrame({'feature score':Series(fs.scores_),'features':Series(X.columns)})    

plt.figure(figsize=(17,7))

sns.barplot(y='feature score',x='features',data=cc.sort_values(by='feature score',ascending=False))
from sklearn.ensemble import ExtraTreesRegressor

forest = ExtraTreesRegressor(n_estimators=100,

                              random_state=0)

forest.fit(X, y)

importances = forest.feature_importances_

indices = np.argsort(importances)[::-1]

for f in range(X.shape[1]):

    print("%d. feature %s (%f)" % (f + 1, six_vars.columns[indices[f]+1], importances[f]))



cc = DataFrame({'feature score':Series(importances),'features':Series(X.columns)})    

plt.figure(figsize=(17,7))

sns.barplot(y='feature score',x='features',data=cc.sort_values(by='feature score',ascending=False))