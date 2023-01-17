#invite people for the Kaggle party
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler
from scipy import stats
import warnings
warnings.filterwarnings('ignore')
%matplotlib inline
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
df_train = pd.read_csv('../input/train.csv')
df_train
df_train.columns
df_train['SalePrice'].describe()
sns.distplot(df_train['SalePrice']);
var = 'GrLivArea'
data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)
data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000));
var = 'TotalBsmtSF'
data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)
data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000));
var = 'OverallQual'
data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)
f, ax = plt.subplots(figsize=(8, 6))
fig = sns.boxplot(x=var, y="SalePrice", data=data)
fig.axis(ymin=0, ymax=800000);
var = 'YearBuilt'
data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)
f, ax = plt.subplots(figsize=(16, 8))
fig = sns.boxplot(x=var, y="SalePrice", data=data)
fig.axis(ymin=0, ymax=800000);
plt.xticks(rotation=90);
corrmat = df_train.corr()
f, ax = plt.subplots(figsize=(12, 9))
sns.heatmap(corrmat, vmax=.8, square=True);
topK = 10 #number of variables for heatmap
cols = corrmat.nlargest(topK, 'SalePrice')['SalePrice'].index
cm = np.corrcoef(df_train[cols].values.T)
sns.set(font_scale=1.25)
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)
plt.show()
sns.set()
cols = ['TotalBsmtSF', '1stFlrSF', 'TotRmsAbvGrd', 'GrLivArea']
sns.pairplot(df_train[cols], size = 2.5)
plt.show();
sns.set()
cols = ['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath', 'YearBuilt']
sns.pairplot(df_train[cols], size = 2.5)
plt.show();
total = df_train.isnull().sum().sort_values(ascending=False)
percent = (df_train.isnull().sum()/df_train.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_data.head(20)
df_train = df_train.drop((missing_data[missing_data['Total'] > 1]).index,1) # Drop the columns
df_train = df_train.drop(df_train.loc[df_train['Electrical'].isnull()].index) # Drop the row
df_train.isnull().sum().max()
df_train
len(df_train.columns)
#scatter plot
plt.scatter(df_train['TotalBsmtSF'], df_train['SalePrice']);
#scatter plot
plt.scatter(df_train[df_train['TotalBsmtSF']>0]['TotalBsmtSF'], df_train[df_train['TotalBsmtSF']>0]['SalePrice']);
from sklearn.ensemble import RandomForestRegressor
X = df_train[:62]
y = np.log1p(df_train["SalePrice"])
y.describe()
X = pd.get_dummies(df_train)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size=.2,
                                                    random_state=0)
regr = RandomForestRegressor(max_depth=10, random_state=0)
regr.fit(X_train, y_train)
print(regr.feature_importances_)
X.iloc[0]
regr.predict([X.iloc[1]])
y[1]
regr.score(X_train, y_train)
regr.score(X_test, y_test)
from sklearn.model_selection import learning_curve
tsz = np.linspace(0.1, 1, 10)
train_sizes, train_scores, test_scores = learning_curve(regr, X, y, train_sizes=tsz)
fig = plt.figure()
plt.plot(train_sizes, train_scores.mean(axis=1), 'ro-', label="Train Scores")
plt.plot(train_sizes, test_scores.mean(axis=1), 'go-', label="Test Scores")
plt.title('Learning Curve: Random Forest Tree')
plt.ylim((0.5, 1.0))
plt.legend()
plt.draw()
plt.show()
regr.predict([X_test.iloc[1]])
y_test[1]
y_test_predicts = regr.predict(X_test)
y_train_predicts = regr.predict(X_train)
from sklearn.metrics import mean_squared_error
mean_squared_error(y_test, y_test_predicts)  
mean_squared_error(y_train, y_train_predicts)
