import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import pandas as pd

%matplotlib inline



from sklearn.linear_model import (LinearRegression, Ridge, Lasso, RandomizedLasso)

from sklearn.preprocessing import MinMaxScaler

from sklearn.ensemble import RandomForestRegressor

from sklearn.decomposition import PCA



from sklearn.model_selection import train_test_split

from sklearn import metrics


df = pd.read_csv("../input/kc_house_data.csv")


numerical_cols = df[['price', 'bedrooms', 'bathrooms', 'sqft_living',

       'sqft_lot', 'floors', 'waterfront', 'view', 'condition', 'grade',

       'sqft_above', 'sqft_basement', 'yr_built', 'yr_renovated', 'zipcode',

       'lat', 'long', 'sqft_living15', 'sqft_lot15']]
df.info()
df.describe()
plt.figure(figsize=(16, 6))

sns.heatmap(df.corr(), annot=True, fmt='.0g' , cmap='coolwarm')
order=df.groupby('grade').mean().sort_values(by='sqft_lot', ascending=True).index.values



sns.barplot(x='grade', y='sqft_lot', data=df, order=order)
df['total_sqft'] = df['sqft_living'] + df['sqft_lot']
df['date'] = pd.to_datetime(df['date'])
sns.set_style('whitegrid')



plt.figure(figsize=(16, 4))



sns.lineplot(x='date', y='price', data=df)



plt.xticks(rotation = 90)



plt.show()
plt.figure(figsize=(16, 4))



sns.lineplot(x='yr_built', y='condition', data=df)
sns.lmplot(x='condition', y='price', data=df)
plt.figure(figsize=(16, 4))



sns.boxplot(x='bedrooms', y='price', data=df)
sns.barplot(x='waterfront', y='price', data=df)
plt.figure(figsize=(8, 4))



sns.barplot(x='view', y='price', data=df)
df['basement'] = df['sqft_basement'] > 0
plt.figure(figsize=(16, 4))



sns.lineplot(x='yr_renovated', y='condition', data=df[df['yr_renovated'] > 0])

plt.figure(figsize=(16, 4))



sns.lineplot(x='yr_built', y='price', data=df)

sns.barplot(x='basement', y='price', data=df)
sns.lmplot(x='sqft_living', y='price', data=df)
sns.lmplot(x='lat', y='price', data=df)
y = df['price']

x = df[['bedrooms', 'bathrooms', 'sqft_living',

       'sqft_lot', 'floors', 'waterfront', 'view', 'condition', 'grade',

       'sqft_above', 'sqft_basement', 'yr_built', 'yr_renovated', 'zipcode',

        'sqft_living15', 'sqft_lot15']]

col_names = x.columns.values
lr = LinearRegression(normalize=True)

lr.fit(x, y)

lr_coef = lr.coef_




def scale_coef (coef, model_name):



    minmax = MinMaxScaler()



    coef = minmax.fit_transform(np.array([np.abs(coef)]).T).T[0]



    coef = pd.DataFrame(data=coef, columns=[model_name], index=col_names)

    

    return round(coef, 2)
l = scale_coef(lr_coef, 'lr')
l


ridge = Ridge(alpha = 7)

ridge.fit(x, y)

r = scale_coef(ridge.coef_, 'ridge')





lasso = Lasso(alpha=.05)

lasso.fit(x, y)

ls = scale_coef(lasso.coef_, 'lasso')
random_forest = RandomForestRegressor(n_jobs=-1, n_estimators=50, verbose=3)

random_forest.fit(x, y)

rf = scale_coef(random_forest.feature_importances_, 'random forest');
fs = pd.concat([l, r, ls, rf], axis=1)
fs['mean'] = (fs['lr'] + fs['ridge'] + fs['lasso'] + fs['random forest']) / 4
fs['mean'] = round(fs['mean'], 2)
order = fs.sort_values(by='mean', ascending=False).index.values
fs
plt.figure(figsize=(16, 4))



sns.barplot(y='index', x='mean', data=fs.reset_index(), order=order)