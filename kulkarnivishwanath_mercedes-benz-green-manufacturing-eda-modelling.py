# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import seaborn as sns

import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor

from sklearn.linear_model import Lasso

from sklearn.metrics import r2_score

from sklearn.model_selection import train_test_split,RandomizedSearchCV,GridSearchCV



import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

pd.set_option('display.max_rows', 500)

pd.set_option('display.max_columns', 500)
train = pd.read_csv("../input/mercedes-benz-greener-manufacturing/train.csv")

test = pd.read_csv("../input/mercedes-benz-greener-manufacturing/test.csv")



print ("Training Dataset Shape {}".format(train.shape))

print ("Testing Dataset Shape {}".format(test.shape))
Submission = test[['ID']]

test.drop('ID',axis=1,inplace=True)

train.drop('ID',axis=1,inplace=True)
train.info()
test.info()
train.dtypes.value_counts()

# Most of the Variables are of int type followed by object.
train.isna().sum()[train.isna().sum() > 0]

# There is no column with missing values.
test.isna().sum()[test.isna().sum() > 0]

# There is no column with missing values.
train['y'].describe()
plt.figure(figsize=(10,8))

sns.distplot(train['y'])
sns.boxplot(train['y'])

# There is one outlier which is way above the other values. Makes sense to delete it
train.drop(train[train['y'] > 250].index[0],axis=0,inplace=True)
train.head()
lst_train = []

for col in train.columns:

    if train[col].nunique() == 1:

        lst_train.append(col)

        print ("Column {} has Single Unique Value".format(col))
lst_test = []

for col in test.columns:

    if test[col].nunique() == 1:

        lst_test.append(col)

        print ("Column {} has Single Unique Value".format(col))
for col in train.columns:

    if train[col].dtype == "object":

        print ("Unique Values in {} column".format(col),train[col].nunique())
for col in test.columns:

    if test[col].dtype == "object":

        print ("Unique Values in {} column".format(col),test[col].nunique())
plt.figure(figsize=(12,8))

sns.boxplot(y=train['y'],x=train['X0'])

plt.xlabel("X0 Values")

plt.ylabel("Values")

plt.title("Box Plot of Dependent Variable with X0 Categorical Variable")
plt.figure(figsize=(12,8))

sns.boxplot(y=train['y'],x=train['X1'])

plt.xlabel("X1 Values")

plt.ylabel("Values")

plt.title("Box Plot of Dependent Variable with X1 Categorical Variable")
plt.figure(figsize=(12,8))

sns.boxplot(y=train['y'],x=train['X2'])

plt.xlabel("X2 Values")

plt.ylabel("Values")

plt.title("Box Plot of Dependent Variable with X2 Categorical Variable")
plt.figure(figsize=(12,8))

sns.boxplot(y=train['y'],x=train['X3'])

plt.xlabel("X3 Values")

plt.ylabel("Values")

plt.title("Box Plot of Dependent Variable with X3 Categorical Variable")
plt.figure(figsize=(12,8))

sns.boxplot(y=train['y'],x=train['X4'])

plt.xlabel("X4 Values")

plt.ylabel("Values")

plt.title("Box Plot of Dependent Variable with X4 Categorical Variable")
test['X4'].value_counts()
train.groupby('X4')['y'].agg({"count","min",'max','mean','median'})
plt.figure(figsize=(12,8))

sns.boxplot(y=train['y'],x=train['X5'])

plt.xlabel("X5 Values")

plt.ylabel("Values")

plt.title("Box Plot of Dependent Variable with X5 Categorical Variable")
plt.figure(figsize=(12,8))

sns.boxplot(y=train['y'],x=train['X6'])

plt.xlabel("X6 Values")

plt.ylabel("Values")

plt.title("Box Plot of Dependent Variable with X6 Categorical Variable")
plt.figure(figsize=(12,8))

sns.boxplot(y=train['y'],x=train['X8'])

plt.xlabel("X8 Values")

plt.ylabel("Values")

plt.title("Box Plot of Dependent Variable with X8 Categorical Variable")
df = train[[col for col in train.columns if ((train[col].dtype == 'int') & (col not in ['ID','y']))]].columns



lst_train = []

for col in train.columns:

    if ((train[col].dtype == 'int') & (col not in ['ID','y'])):

        lst_train.append(sum(train[col])/len(train))



df_new = pd.DataFrame(lst_train,df,columns=['Ratio_Train']).sort_index()



lst_test = []

for col in test.columns:

    if ((test[col].dtype == 'int') & (col !='ID')):

        lst_test.append(sum(test[col])/len(test))



df_new['Ratio_Test'] = lst_test



df_new.sort_index()
# Let's check the correlation between the variables and eliminate the one's that have high correlation

# Threshold for removing correlated variables

threshold = 0.9



# Absolute value correlation matrix

corr_matrix = train.corr().abs()

corr_matrix.head()



# Some of the columns/rows here are represented as NaN's because these variables have a single unique value (either 1 or 0)
# Upper triangle of correlations

upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))

upper.head()
# Select columns with correlations above threshold

to_drop = [column for column in upper.columns if any(upper[column] > threshold)]



print('There are %d columns to remove.' % (len(to_drop)))

print ("Following columns can be dropped {}".format(to_drop))
train = train.drop(columns = to_drop)

test = test.drop(columns = to_drop)



print('Training shape: ', train.shape)

print('Testing shape: ', test.shape)
# As we saw in the Boxplots above, this variable is dominated by a single value. Hence deleting this variable.

train.drop('X4',axis=1,inplace=True)

test.drop('X4',axis=1,inplace=True)
train.head()
test.head()
# Since there are lot of levels in the categorical variables, it makes sense to convert them to category data type instead of creating dummy variables

# which will only add to the complexity by creating large number of features.

for col in train.columns:

    if train[col].dtype == "object":

        train[col] = train[col].astype('category')

        train[col] = train[col].cat.codes
for col in test.columns:

    if test[col].dtype == "object":

        test[col] = test[col].astype('category')

        test[col] = test[col].cat.codes
X = train[[col for col in train.columns if col!='y']]

y = train['y']
rf = RandomForestRegressor(n_jobs=-1,random_state=42)

rf.fit(X,y)
feature_importances = rf.feature_importances_

feature_importances = pd.DataFrame({'feature': list(X.columns), 'importance': feature_importances}).sort_values('importance', ascending = False)

feature_importances
# Dropping the variables which have 0 feature importance. Random Forests are a good way to measure the feature importance of the variables. 

zero_features = list(feature_importances[feature_importances['importance'] == 0.0]['feature'])

print('There are %d features with 0.0 importance' % len(zero_features))

feature_importances
train.drop(zero_features,axis=1,inplace=True)

test.drop(zero_features,axis=1,inplace=True)
X = train[[col for col in train.columns if col!='y']]

y = train['y']
rf_2 = RandomForestRegressor(n_jobs=-1,random_state=42)

rf_2.fit(X,y)
#Submission['y'] = rf_2.predict(test)

#Submission.to_csv("Submission_1.csv",index=None)
rf_3 = RandomForestRegressor(n_jobs=-1,random_state=42)

params = {"n_estimators":list(range(50,501,100)),

          "max_features":[0.5,'sqrt','log2'],

          "min_samples_leaf":[1,3,5,10,25],

          "min_samples_split":[2,5,7,10,15],

          "max_depth":[3,5,7,9]}

r_search = RandomizedSearchCV(estimator=rf_3,param_distributions=params,cv=5,scoring='r2')

r_search.fit(X,y)
r_search.best_estimator_,r_search.best_params_,r_search.best_score_
feature_importances = r_search.best_estimator_.feature_importances_

feature_importances = pd.DataFrame({'feature': list(X.columns), 'importance': feature_importances}).sort_values('importance', ascending = False)

feature_importances
zero_features = list(feature_importances[feature_importances['importance'] == 0.0]['feature'])

print('There are %d features with 0.0 importance' % len(zero_features))

feature_importances
train.drop(zero_features,axis=1,inplace=True)

test.drop(zero_features,axis=1,inplace=True)
X = train[[col for col in train.columns if col!='y']]

y = train['y']



rf_4 = RandomForestRegressor(n_jobs=-1,random_state=42)

params = {"n_estimators":list(range(50,501,100)),

          "max_features":[0.5,'sqrt','log2'],

          "min_samples_leaf":[1,3,5,10,25],

          "min_samples_split":[2,5,7,10,15],

          "max_depth":[3,5,7,9]}

r_search2 = RandomizedSearchCV(estimator=rf_4,param_distributions=params,cv=5,scoring='r2')

r_search2.fit(X,y)



r_search2.best_estimator_,r_search2.best_params_,r_search2.best_score_
Submission['y'] = r_search2.predict(test)

Submission.to_csv("Submission_2.csv",index=None)