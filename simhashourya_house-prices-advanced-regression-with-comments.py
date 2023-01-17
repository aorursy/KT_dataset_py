# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
#Importing The dataset

train_df = pd.read_csv("../input/house-prices-advanced-regression-techniques/train.csv")

train_df

#Dropping and saving  the ID column

train_df.drop("Id", axis =1, inplace = True)

train_df.head()

#Checking for outliers

import seaborn as sns

fig , ax = plt.subplots()

ax.scatter(x = train_df['GrLivArea'], y = train_df['SalePrice'])

plt.ylabel('SalePrice', fontsize=10)

plt.xlabel('GrLivArea', fontsize=10)

plt.show()
train_df = train_df.drop(train_df[(train_df['GrLivArea']>4000) & (train_df['SalePrice']<300000)].index)



#Check the scatter plot again after removing outliers

fig, ax = plt.subplots()

ax.scatter(train_df['GrLivArea'], train_df['SalePrice'])

plt.ylabel('SalePrice', fontsize=10)

plt.xlabel('GrLivArea', fontsize=10)

plt.show()
from scipy.stats import norm

sns.distplot(train_df['SalePrice'] , fit=norm)

mean = np.mean(train_df)

sd = np.std(train_df)

#Q-Q plot

from scipy import stats

fig = plt.figure()

result = stats.probplot(train_df['SalePrice'], plot= plt)

plt.show()
#Transforming the Target variable(Box-Cox)

train_df['SalePrice'] = np.log1p(train_df['SalePrice'])

sns.distplot(train_df['SalePrice'], fit =norm)

#getting the new Q-Q plot

fig = plt.figure()

result = stats.probplot(train_df['SalePrice'], plot= plt)

plt.show()
#importing the test data

test_df = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')

test_df.head()
#Missing Data

train_df_na = (train_df.isnull().sum()/len(train_df))*100

train_df_na = train_df_na.drop(train_df_na[train_df_na == 0].index).sort_values(ascending = False)[:20]

missing_data = pd.DataFrame({'Missing %': train_df_na})

missing_data.head(20)
f, ax = plt.subplots(figsize=(15,10))

plt.xticks(rotation='90')

sns.barplot(x=train_df_na.index, y= train_df_na)

plt.xlabel('Features', fontsize=12)

plt.ylabel('Percentage of missing values', fontsize=12)

plt.title('Missing value percentage of each feature', fontsize=15)
#Dropping features with high missing values %

train_df = train_df.drop((missing_data[missing_data['Missing %'] > 5.56]).index,1)

train_df.isnull().sum().sort_values(ascending=False).head(20)

total_test = (test_df.isnull().sum()/len(test_df)) *100

total_test = total_test.drop(total_test[total_test == 0].index).sort_values(ascending = False)[:20]

missing_data = pd.DataFrame({'Missing %': total_test})

missing_data.head(20)
#Dropping features with high missing values %

test_df = test_df.drop((missing_data[missing_data['Missing %'] > 5.56]).index,1)

test_df.isnull().sum().sort_values(ascending=False).head(20)
#Train

categorical_feature_mask = train_df.dtypes==object

# filter categorical columns using mask and turn it into alist

categorical_cols = train_df.columns[categorical_feature_mask].tolist()

from sklearn.preprocessing import LabelEncoder

label = LabelEncoder()

train_df[categorical_cols] = train_df[categorical_cols].apply(lambda x: label.fit_transform(x.astype(str)))

train_df.head()
categorical_feature_mask = test_df.dtypes==object

# filter categorical columns using mask and turn it into alist

categorical_cols = test_df.columns[categorical_feature_mask].tolist()

from sklearn.preprocessing import LabelEncoder

label = LabelEncoder()

test_df[categorical_cols] = test_df[categorical_cols].apply(lambda x: label.fit_transform(x.astype(str)))

test_df.head()
train_df.isnull().sum().sort_values(ascending=False).head(20)

test_df.isnull().sum().sort_values(ascending=False).head(20)

#Training Dataset

for col in ('GarageYrBlt', 'MasVnrArea'):

    train_df[col] = train_df[col].fillna(0)
#Test Dataset

for col in ('GarageYrBlt', 'MasVnrArea','BsmtFullBath','BsmtHalfBath', 'GarageCars','GarageArea','BsmtUnfSF','TotalBsmtSF','BsmtFinSF1','BsmtFinSF2'):

    test_df[col] = test_df[col].fillna(0)
#saleprice correlation matrix

k = 15 #number of variables for heatmap

plt.figure(figsize=(10,8))

corrmat = train_df.corr()

# picking the top 15 correlated features

cols = corrmat.nlargest(k, 'SalePrice')['SalePrice'].index

cm = np.corrcoef(train_df[cols].values.T)

sns.set(font_scale=1.25)

hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)

plt.show()
train_df = train_df[cols]

cols
test_df=test_df[cols.drop('SalePrice')]

test_df.head()

#Group by neighborhood and fill in missing value by the median LotFrontage of all the neighborhood

#train_df['LotFrontage'] = train_df.groupby("Neighborhood")['LotFrontage'].transform(lambda x: x.fillna(x.median()))
from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(train_df.drop('SalePrice', axis= 1), train_df['SalePrice'], test_size=0.25, random_state=1)

y_train = y_train.values.reshape(-1,1)

y_test = y_test.values.reshape(-1,1)
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

sc_y = StandardScaler()

X_train = sc.fit_transform(X_train)

X_test = sc.fit_transform(X_test)

y_train = sc.fit_transform(y_train)

y_test = sc_y.fit_transform(y_test)
from sklearn.linear_model import LinearRegression

lm = LinearRegression()

lm.fit(X_train,y_train)

y_pred = lm.predict(X_test)

y_pred = y_pred.reshape(-1,1)

plt.figure(figsize=(10,10))

plt.scatter(y_pred,y_test)

plt.xlabel("Observed")

plt.ylabel("Predicted")

plt.show()
from sklearn import metrics

print("MAE", metrics.mean_absolute_error(y_test,y_pred))

print("MSE", metrics.mean_squared_error(y_test,y_pred))

print("RMSE", np.sqrt(metrics.mean_squared_error(y_test,y_pred)))

print('R-Squared', metrics.r2_score(y_test,y_pred))
from sklearn import ensemble

from sklearn.utils import shuffle

from sklearn.metrics import r2_score, mean_squared_error

params = {'n_estimators': 500, 'max_depth': 3, 'min_samples_split': 2,

          'learning_rate': 0.01, 'loss':'ls'}

clf = ensemble.GradientBoostingRegressor(**params)

clf.fit(X_train,y_train)

clf_pred = clf.predict(X_test)

clf_pred = clf_pred.reshape(-1,1)

plt.figure(figsize=(10,10))

plt.scatter(clf_pred,y_test, c="brown")

plt.xlabel("Observed")

plt.ylabel("Predicted")

plt.show()
print('MAE:', metrics.mean_absolute_error(y_test, clf_pred))

print('MSE:', metrics.mean_squared_error(y_test, clf_pred))

print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, clf_pred)))

print('R-Squared', metrics.r2_score(y_test,clf_pred))
#Decision Tree

from sklearn.tree import DecisionTreeRegressor

dr = DecisionTreeRegressor()

dr.fit(X_train,y_train)

dr_pred = dr.predict(X_test)

dr_pred= dr_pred.reshape(-1,1)

plt.figure(figsize=(10,10))

plt.scatter(dr_pred,y_test, c="yellow")

plt.xlabel("Observed")

plt.ylabel("Predicted")

plt.show()
print('MAE:', metrics.mean_absolute_error(y_test, dr_pred))

print('MSE:', metrics.mean_squared_error(y_test, dr_pred))

print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, dr_pred)))

print('R-Squared', metrics.r2_score(y_test,dr_pred))
#Support Vector Machine

from sklearn.svm import SVR

svr = SVR(kernel='rbf')

svr.fit(X_train,y_train)

svr_pred = svr.predict(X_test)

svr_pred= svr_pred.reshape(-1,1)

plt.figure(figsize=(10,10))

plt.scatter(svr_pred,y_test, c="black")

plt.xlabel("Observed")

plt.ylabel("Predicted")

plt.show()
print('MAE:', metrics.mean_absolute_error(y_test, svr_pred))

print('MSE:', metrics.mean_squared_error(y_test, svr_pred))

print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, svr_pred)))

print('R-Squared', metrics.r2_score(y_test,svr_pred))
sub = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')

test_id = sub['Id']

sub = pd.DataFrame(test_id, columns=['Id'])

test_df = sc.fit_transform(test_df)

test_df.shape

test_prediction_clf=clf.predict(test_df)

test_prediction_clf= test_prediction_clf.reshape(-1,1)

test_prediction_clf

test_prediction_clf =sc_y.inverse_transform(test_prediction_clf)

test_prediction_clf = pd.DataFrame(test_prediction_clf, columns=['SalePrice'])

test_prediction_clf.head()
result = pd.concat([sub,test_prediction_clf], axis=1)

result.head()

result.to_csv('submission.csv',index=False)
