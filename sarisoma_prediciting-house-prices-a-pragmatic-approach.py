import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
%matplotlib inline
df_train = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv')
df_train.head(3)
df_train.info()
# Determine correlation and derive feature variables from it.
# Variables providing less than 0.50 to be discarded. 
df_train.corr()[['SalePrice']].sort_values(by='SalePrice', ascending=False).head(11)
# Fill missing values for the variables of interest:
df_train['GarageYrBlt'].fillna(round(df_train['GarageYrBlt'].median(), 1), inplace=True)
df_train['MasVnrArea'].fillna(0.0, inplace=True)
# A simple regression plot to visualize correlations on each feature variable to be used:
def regplot(x):
    sns.regplot(x, y=df_train['SalePrice'], data=df_train)
regplot(x='OverallQual')
plt.ylim(0, )
regplot(x='GrLivArea')
plt.ylim(0, )
regplot(x= 'GarageCars')
plt.ylim(0, )
regplot(x='GarageArea')
plt.ylim(0, )
regplot(x= 'TotalBsmtSF')
plt.ylim(0, )
regplot(x='1stFlrSF')
plt.ylim(0, )
regplot(x='FullBath')
plt.ylim(0, )
regplot(x= 'TotRmsAbvGrd')
plt.ylim(0, )
regplot(x = 'YearBuilt')
plt.ylim(0, )
regplot(x= 'YearRemodAdd')
plt.ylim(0, )
X = df_train[['OverallQual', 'GrLivArea', 'GarageCars', 'GarageArea', 'TotalBsmtSF', '1stFlrSF', 'FullBath',
         'TotRmsAbvGrd', 'YearBuilt', 'YearRemodAdd', 'GarageYrBlt', 'MasVnrArea', 'Fireplaces', 'BsmtFinSF1']]
# verify for null values
X.info()
# define target variable
y = df_train[['SalePrice']]
y.info()
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
lr = LinearRegression().fit(X_train, y_train)
print('lr.coef_: {}'.format(lr.coef_))
print('lr.intercept_: {}'.format(lr.intercept_))
print('training set score: {}'.format(lr.score(X_train, y_train)))
print('test set score: {}'.format(lr.score(X_test, y_test)))
from sklearn.linear_model import Ridge
ridge = Ridge(alpha=.1)
ridge.fit(X_train, y_train)
print('training set score: {}'.format(ridge.score(X_train, y_train)))
print('test set score: {}'.format(ridge.score(X_test, y_test)))
from sklearn.linear_model import Lasso
lasso = Lasso(alpha=1, max_iter=100000)
lasso.fit(X_train, y_train)
print('training set score: {}'.format(lasso.score(X_train, y_train)))
print('test set score: {}'.format(lasso.score(X_test, y_test)))
print('number of features used: {}'.format(np.sum(lasso.coef_ != 0)))
from sklearn.tree import DecisionTreeRegressor
tree = DecisionTreeRegressor()
tree.fit(X_train, y_train)
print('training set score: {}'.format(tree.score(X_train, y_train)))
print('test set score: {}'.format(tree.score(X_test, y_test)))
from sklearn.ensemble import GradientBoostingRegressor
gbrt = GradientBoostingRegressor(random_state=0)
gbrt.fit(X_train, y_train)
print('training set score: {}'.format(gbrt.score(X_train, y_train)))
print('test set score: {}'.format(gbrt.score(X_test, y_test)))
# this plot / algorithm referenced from the book: Le Machine Learning avec Python / O'Reilly / 2018

def plot_feature_importance(model):
    n_features = X.shape[1]
    plt.barh(range(n_features), model.feature_importances_, align='center')
    plt.yticks(np.arange(n_features), X.columns)
    plt.xlabel('Feature Importance')
    plt.ylabel('Feature')
    plt.ylim(-1, n_features)
plot_feature_importance(gbrt)