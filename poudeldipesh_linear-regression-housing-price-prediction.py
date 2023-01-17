import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

% matplotlib inline

import seaborn as sns

import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression

from sklearn.model_selection import cross_val_score





#Reading the CSV file

a= pd.read_csv('../input/kc_house_data.csv')
a.head()
print('The datatype of the features are as follows:')

print(a.dtypes)
print("The missing values on each feature are as follows:")

print(a.isnull().sum())
print("A quick glance at the dataset:")

print(a.describe())
print("Let's check the correlation of the features in relation to price.")

corr_price= a.corr()['price']

print(corr_price.sort_values(ascending=False))
correlation_price = a.corr()

plt.figure(figsize=(14, 12))

heatmap = sns.heatmap(correlation_price, annot=True, linewidths=0, vmin=-1, cmap="YlGn")
sns.lmplot(x='sqft_living',y='price',data=a,fit_reg=True);

sns.lmplot(x='sqft_lot',y='price',data=a,fit_reg=True);

sns.lmplot(x='grade',y='price',data=a,fit_reg=True);

sns.lmplot(x='sqft_above',y='price',data=a,fit_reg=True);

sns.lmplot(x='bathrooms',y='price',data=a,fit_reg=True);

sns.lmplot(x='bedrooms',y='price',data=a,fit_reg=True);

sns.lmplot(x='view',y='price',data=a,fit_reg=True);

sns.lmplot(x='sqft_living15',y='price',data=a,fit_reg=True);

sns.lmplot(x='sqft_basement',y='price',data=a,fit_reg=True); #weak corealation, therefore we cannot use to predict price

sns.lmplot(x='sqft_lot15',y='price',data=a,fit_reg=True);

sns.lmplot(x='yr_renovated',y='price',data=a,fit_reg=True);

sns.lmplot(x='lat',y='price',data=a,fit_reg=True);

sns.lmplot(x='long',y='price',data=a,fit_reg=True);
X= a.drop(['price','date','id'], axis=1)

y = a[['price']]
#Building our linear regression model



X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=2,  test_size=0.10)

regression_model = LinearRegression()

regression_model.fit(X_train, y_train)

print('Accuracy score with cross validation is:')

scores = cross_val_score(regression_model, X, y, cv=10).mean()

print(scores)

print("")

print("Linear regression accuracy score without CV:")

print(regression_model.score(X_test, y_test))

print("")
