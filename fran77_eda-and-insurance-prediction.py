# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

from matplotlib import pyplot as plt

# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train = pd.read_csv('/kaggle/input/health-insurance-cross-sell-prediction/train.csv')

test = pd.read_csv('/kaggle/input/health-insurance-cross-sell-prediction/test.csv')

subm= pd.read_csv('/kaggle/input/health-insurance-cross-sell-prediction/sample_submission.csv')
train.head()
len_train = len(train)

len_train
plt.figure(figsize=(8, 6))

sns.countplot(train.Gender)
plt.figure(figsize=(12, 6))

train.groupby('Age')['id'].count().plot()
# Pick at 25 and an another one at 45
train.Driving_License.value_counts()
# Some of them don't have driving license
train.Region_Code.value_counts()
# If customer already has Vehicle Insurance
plt.figure(figsize=(8, 6))

sns.countplot(train.Previously_Insured)
plt.figure(figsize=(8, 6))

sns.countplot(train.Vehicle_Age)
# If customer got his/her vehicle damaged in the past
train.Vehicle_Damage.value_counts()
# The amount customer needs to pay as premium in the year
plt.figure(figsize=(12, 6))

train.groupby('Annual_Premium')['id'].count().plot()
train.Annual_Premium.describe()
# Zooming

plt.figure(figsize=(12, 6))

train[train.Annual_Premium < 20000].groupby('Annual_Premium')['id'].count().plot()
train.Policy_Sales_Channel.value_counts()[:20]
train.Vintage.value_counts()[:20]
plt.figure(figsize=(8, 6))

sns.countplot(train.Response)
train.isnull().sum()
# No null values
sns.catplot(x="Gender", y="Response", kind="bar", data=train)
sns.catplot(x="Driving_License", y="Response", kind="bar", data=train)
# More probabilities to have a positive response with a driving license
sns.catplot(x="Previously_Insured", y="Response", kind="bar", data=train)
# A very little amount of positive response if previously insured
sns.catplot(x="Vehicle_Damage", y="Response", kind="bar", data=train)
# More probabilities to have a positive response if the customer got his/her vehicle damaged in the past
sns.catplot(x="Vehicle_Age", y="Response", kind="bar", data=train)
# The more old the vehicule, the more probabilities to have a positive response
train.groupby('Response')['Age'].mean()
data = pd.concat([train, test])

data = data.drop('id', axis=1)
# Convert categorical features
data['Gender'] = data.Gender.astype('category').cat.codes

data['Vehicle_Age'] = data.Vehicle_Age.astype('category').cat.codes

data['Vehicle_Damage'] = data.Vehicle_Damage.astype('category').cat.codes
train = data[:len_train]

test = data[len_train:]
plt.figure(figsize=(8, 6))

sns.distplot(data.Age, bins=10)
data['AgeBin'] = pd.cut(data['Age'], 10)

data[['AgeBin', 'Response']].groupby(['AgeBin'], as_index=False).mean().sort_values(by='AgeBin', ascending=True)
data.Age.describe()
data.loc[(data['Age'] >= 20) & (data['Age'] <= 30), 'Age'] = 0

data.loc[(data['Age'] > 30) & (data['Age'] <= 40), 'Age'] = 1

data.loc[(data['Age'] > 40) & (data['Age'] <= 50), 'Age'] = 2

data.loc[(data['Age'] > 50) & (data['Age'] <= 60), 'Age'] = 3

data.loc[(data['Age'] > 60) & (data['Age'] <= 70), 'Age'] = 4

data.loc[(data['Age'] > 70) & (data['Age'] <= 80), 'Age'] = 5

data.loc[ data['Age'] > 80, 'Age'] = 6
data['Annual_Premium'].describe()
# data with annual premium < 100000

data_premimum = data[data.Annual_Premium <100000]
data_premimum['PremBin'] = pd.cut(data_premimum['Annual_Premium'], 10)

data_premimum[['PremBin', 'Response']].groupby(['PremBin'], as_index=False).mean().sort_values(by='PremBin', ascending=True)
data.loc[ data['Annual_Premium'] <= 10000, 'Annual_Premium'] = 0

data.loc[(data['Annual_Premium'] > 10000) & (data['Annual_Premium'] <= 20000), 'Age'] = 1

data.loc[(data['Annual_Premium'] > 20000) & (data['Annual_Premium'] <= 30000), 'Annual_Premium'] = 2

data.loc[(data['Annual_Premium'] > 30000) & (data['Annual_Premium'] <= 40000), 'Annual_Premium'] = 3

data.loc[(data['Annual_Premium'] > 40000) & (data['Annual_Premium'] <= 50000), 'Annual_Premium'] = 4

data.loc[(data['Annual_Premium'] > 50000) & (data['Annual_Premium'] <= 60000), 'Annual_Premium'] = 5

data.loc[(data['Annual_Premium'] > 60000) & (data['Annual_Premium'] <= 70000), 'Annual_Premium'] = 6

data.loc[(data['Annual_Premium'] > 70000) & (data['Annual_Premium'] <= 80000), 'Annual_Premium'] = 7

data.loc[(data['Annual_Premium'] > 80000) & (data['Annual_Premium'] <= 90000), 'Annual_Premium'] = 8

data.loc[(data['Annual_Premium'] > 90000) & (data['Annual_Premium'] <= 100000), 'Annual_Premium'] = 9

data.loc[ data['Age'] > 100000, 'Age'] = 10
train = data[:len_train]

test = data[len_train:]
train = train.drop('AgeBin', axis=1)

test = test.drop(['Response', 'AgeBin'], axis=1)
from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier

from sklearn import tree

from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import KFold, cross_val_score, train_test_split

from sklearn.metrics import accuracy_score

from sklearn.model_selection import GridSearchCV

from xgboost import XGBClassifier

from lightgbm import LGBMClassifier
# plot the heatmap

train_corr = train.corr()

plt.figure(figsize=(10,8))

sns.heatmap(train_corr, 

        xticklabels=train_corr.columns,

        yticklabels=train_corr.columns, cmap=sns.diverging_palette(220, 20, n=200))
train_corr[['Response']]
train = train.drop(['Region_Code', 'Vintage'], axis=1)

test = test.drop(['Region_Code', 'Vintage'], axis=1)
X = train.drop("Response", axis=1)

Y = train["Response"]
# Split 20% test, 80% train



X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size = 0.2, random_state=0)
# Logistic Regression



log = LogisticRegression(max_iter=100)

log.fit(X_train, Y_train)

Y_pred_log = log.predict(X_val)

acc_log = accuracy_score(Y_pred_log, Y_val)

acc_log
rf = RandomForestClassifier()



# search the best params

grid = {'n_estimators':[100,200], 'max_depth': [10,20]}



clf_rf = GridSearchCV(rf, grid, cv=10)

clf_rf.fit(X_train, Y_train)



Y_pred_rf = clf_rf.predict(X_val)

# get the accuracy score

acc_rf = accuracy_score(Y_pred_rf, Y_val)

print(acc_rf)
clf_rf.best_params_
knn = KNeighborsClassifier()



# values we want to test for n_neighbors

param_grid = {'n_neighbors': np.arange(1, 20)}



clf_knn = GridSearchCV(knn, param_grid, cv=5)



#fit model to data

clf_knn.fit(X_train, Y_train)



Y_pred_knn = clf_knn.predict(X_val)

# get the accuracy score

acc_knn = accuracy_score(Y_pred_rf, Y_val)

print(acc_knn)
# LGBM Classifier



lgbm = LGBMClassifier(random_state=0)

lgbm.fit(X_train, Y_train)

Y_pred_lgbm = lgbm.predict(X_val)

acc_lgbm = accuracy_score(Y_pred_lgbm, Y_val)

acc_lgbm
clf_xgb = XGBClassifier(learning_rate=0.02, n_estimators=100, max_depth = 10)



clf_xgb.fit(X_train, Y_train)



Y_pred_xgb = clf_xgb.predict(X_val)

# get the accuracy score

acc_xgb = accuracy_score(Y_pred_xgb, Y_val)

print(acc_xgb)
# We will use the Random Forest model
clf_rf.fit(X, Y)



Y_test = clf_rf.predict(test)
subm['Response']= Y_test
subm.to_csv('submission.csv',index=False)