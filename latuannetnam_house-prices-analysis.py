# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import numpy as np

from scipy.stats import norm

from sklearn.preprocessing import StandardScaler

from sklearn.tree import DecisionTreeRegressor

from sklearn.linear_model import LinearRegression

from sklearn.model_selection import cross_val_score, train_test_split

from sklearn.metrics import mean_squared_error, make_scorer

from math import sqrt

from scipy import stats

import datetime

import warnings

warnings.filterwarnings('ignore')

%matplotlib inline



# Input data files are available in the DATA_DIR directory.

DATA_DIR="../input"

# Any results you write to the current directory are saved as output.
# Load data. Download from:https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data

train_data = pd.read_csv(DATA_DIR + "/train.csv")

eval_data =  pd.read_csv(DATA_DIR + "/test.csv")
# exploring train data

train_data.head(5)
train_data_columns = train_data.columns.values

print(train_data_columns)

print('Train data columns:', len(train_data_columns))
eval_data_columns = eval_data.columns.values

print(eval_data_columns)

print('Evaluation data columns:',len(eval_data_columns))
print("Train data size:",len(train_data))

print("Test data size:", len(eval_data))

print("Missing columns in test_data:", np.setdiff1d(train_data_columns, eval_data_columns))
# Check if any cell has NULL value

isnull_data = train_data.isnull().any()

print(isnull_data[isnull_data == True].sort_index())
train_data['SalePrice'].isnull().any()
train_data['SalePrice'].describe()
#histogram

#train_data['SalePrice'].hist(bins=20)

target = train_data['SalePrice']

target_log = np.log(target)

plt.figure(figsize=(10,5))

plt.subplot(1,2,1)

sns.distplot(target, bins=50)

plt.title('Original Data')

plt.xlabel('Sale Price')



plt.subplot(1,2,2)

sns.distplot(target_log, bins=50)

plt.title('Natural Log of Data')

plt.xlabel('Natural Log of Sale Price')

plt.tight_layout()
correlation = train_data.corr()['SalePrice'].sort_values()[-10:]

#print(correlation)
correlation.plot.bar(figsize=(20,8), sort_columns = True)
#saleprice correlation matrix

corrmat = train_data.corr()

k = 10 #number of variables for heatmap

cols = corrmat.nlargest(k, 'SalePrice')['SalePrice'].index

print(cols.values)

cm = np.corrcoef(train_data[cols].values.T)

sns.set(font_scale=1.25)

hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)

plt.show()
var = 'OverallQual'

qual_data = pd.concat([train_data['SalePrice'], train_data[var]], axis=1)

f, ax = plt.subplots(figsize=(8, 6))

fig = sns.boxplot(x=var, y="SalePrice", data=qual_data)

fig.axis(ymin=0, ymax=800000);

var = 'GarageCars'

qual_data = pd.concat([train_data['SalePrice'], train_data[var]], axis=1)

f, ax = plt.subplots(figsize=(8, 6))

fig = sns.boxplot(x=var, y="SalePrice", data=qual_data)

fig.axis(ymin=0, ymax=800000);

#train_data.plot.scatter(x='GrLivArea', y='SalePrice')

#train_data.plot.scatter(x='GarageArea', y='SalePrice')

#train_data.plot.scatter(x='TotalBsmtSF', y='SalePrice')

#scatterplot

sns.set()

cols = ['SalePrice', 'GrLivArea','GarageArea', 'TotalBsmtSF','1stFlrSF']

sns.pairplot(train_data[cols], size = 4)

plt.show();
#missing data

features = ['SalePrice', 'GrLivArea','GarageArea', 'TotalBsmtSF', '1stFlrSF']

sub_data = train_data[features]

sub_data.head(5)

null_data = sub_data.isnull()

total = null_data.sum().sort_values(ascending=False)

percent = (null_data.sum()/null_data.count()).sort_values(ascending=False)

missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

missing_data.head(20)
#GrLivArea

#deleting points

train_data.sort_values(by = 'GrLivArea', ascending = False)[:2]

train_data = train_data.drop(train_data[train_data['Id'] == 1299].index)

train_data = train_data.drop(train_data[train_data['Id'] == 524].index)

train_data.plot.scatter(x='GrLivArea', y='SalePrice')
# GarageArea

#deleting points

var = 'GarageArea'

train_data.sort_values(by = var, ascending = False)[:3]

train_data = train_data.drop(train_data[train_data['Id'] == 582].index)

train_data = train_data.drop(train_data[train_data['Id'] == 1191].index)

train_data = train_data.drop(train_data[train_data['Id'] == 1062].index)

train_data.plot.scatter(x=var, y='SalePrice')
#TotalBsmtSF

#deleting points

var = 'TotalBsmtSF'

train_data.sort_values(by = var, ascending = False)[:3]

train_data = train_data.drop(train_data[train_data['Id'] == 333].index)

train_data = train_data.drop(train_data[train_data['Id'] == 497].index)

train_data = train_data.drop(train_data[train_data['Id'] == 441].index)

train_data.plot.scatter(x=var, y='SalePrice')
#1stFlrSF

#deleting points

var = '1stFlrSF'

train_data.sort_values(by = var, ascending = False)[:1]

train_data = train_data.drop(train_data[train_data['Id'] == 1025].index)

train_data.plot.scatter(x=var, y='SalePrice')
features = ['GrLivArea','GarageArea', 'TotalBsmtSF', '1stFlrSF']

label = ['SalePrice']

X = train_data[features]

X.head(5)

Y = np.log(train_data[label])

Y.head(5)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.33, random_state=324)

print("train size:", len(X_train))

print("test size:", len(X_test))

print("Split ratio", len(X_test)/len(X_train))
regressor = LinearRegression()

regressor.fit(X_train, Y_train)
Y_prediction = regressor.predict(X_test)

Y_prediction[:5]
RMSE1 = sqrt(mean_squared_error(y_true = Y_test, y_pred = Y_prediction))

print("RMSE1:", RMSE1)
scorer = make_scorer(mean_squared_error, greater_is_better = False)

RMSE2 = np.sqrt(-cross_val_score(regressor, X_test, Y_test, scoring = scorer, cv = 10))

print("RMSE2:", RMSE2.mean())
Y_test.describe()
#Plot residuals

plt.scatter(Y_prediction, Y_prediction - Y_test, c = "lightgreen", marker = "s", label = "Validation data")

plt.title("Linear regression")

plt.xlabel("Predicted values")

plt.ylabel("Residuals")
plt.scatter(Y_prediction, Y_test, c = "lightgreen", marker = "s", label = "Validation data")

plt.title("Linear regression")

plt.xlabel("Predicted values")

plt.ylabel("Real values")

plt.legend(loc = "upper left")

plt.show()
X_eval = eval_data[features]
isnull_data = X_eval.isnull().any()

isnull_data[isnull_data == True].sort_index()
#filling Null daa

X_eval['GarageArea'].fillna(X_eval['GarageArea'].mean(), inplace=True)

X_eval['GarageArea'][:5]
#filling Null daa

X_eval['TotalBsmtSF'].fillna(X_eval['TotalBsmtSF'].mean(), inplace=True)

X_eval['TotalBsmtSF'][:5]
X_eval.isnull().any()
Y_eval_log = regressor.predict(X_eval)

# Transform SalePrice to normal

Y_eval = np.exp(Y_eval_log.ravel())

print(type(Y_eval))

print(Y_eval[:5])
eval_output = pd.DataFrame({'Id': eval_data['Id'], 'SalePrice': Y_eval})

print(len(eval_output))

eval_output.head()
today = str(datetime.date.today())

print(today)

eval_output.to_csv(today+'-submission.csv',index=False)