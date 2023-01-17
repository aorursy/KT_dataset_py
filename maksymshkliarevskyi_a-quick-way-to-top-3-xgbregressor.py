import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns



from sklearn.model_selection import train_test_split, KFold, GridSearchCV

from sklearn.preprocessing import LabelEncoder

from xgboost import XGBRegressor

from sklearn.metrics import mean_absolute_error



from scipy.stats import zscore



import warnings

warnings.filterwarnings('ignore')



# Input data files are available in the read-only "../input/" directory

# Running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
X = pd.read_csv('../input/home-data-for-ml-course/train.csv', index_col='Id')

X_test = pd.read_csv('../input/home-data-for-ml-course/test.csv', index_col='Id')



y = X.SalePrice

X.drop(['SalePrice'], axis=1, inplace=True)
sum(X.columns != X_test.columns)
print('Train data: {}'.format(X.shape))

print('Train data targets: {}'.format(len(y)))

print('*' *25)

print('Test data: {}'.format(X_test.shape))

print('*' *25, '\n')

X.info()
# Concatenation of train and test sets for data preprocessing

X_all = pd.concat([X, X_test], axis=0)



# A first look at columns with missing values

cols_with_na = X_all.isna().sum()[X_all.isna().sum() > 0].sort_values(ascending = False)



plt.figure(figsize=(8, 8))

sns.barplot(x = cols_with_na.values, y = cols_with_na.index, color = 'green', edgecolor = 'black')

plt.title('NA values')
# Remove columns with more than 10% missing values

na_limit = X_all.shape[0] * 0.1

drop_cols = cols_with_na[cols_with_na > na_limit].index



X_all.drop(drop_cols, axis=1, inplace=True)



# Check the sum of missing values in each column

cols_with_na = X_all.isna().sum()[X_all.isna().sum() > 0].sort_values(ascending = False)



plt.figure(figsize=(8, 8))

sns.barplot(x = cols_with_na.values, y = cols_with_na.index, color = 'green', edgecolor = 'black')

plt.title('NA values')
# Columns for replacing

numeric_missing = [col for col in X_all.columns if X_all[col].dtype in ['float64', 'int64'] and col in cols_with_na.index]

object_missing = [x for x in cols_with_na.index if x not in numeric_missing]



# Have a look at object_missing columns

fig = plt.figure(figsize=(20, 15))

p = 1

for i in object_missing:

    fig.add_subplot(3, 6, p)

    X_all[i].hist()

    plt.title(i)

    p += 1

plt.show()
# Based on visualization above we divide object columns by replacement type

mode_missing = ['Functional', 'Utilities', 'SaleType', 'Electrical']

none_missing = [x for x in object_missing if x not in mode_missing]

print('Mode: {}'.format(mode_missing))

print('None: {}'.format(none_missing))
# Replace all missing values

for col in numeric_missing:

    X_all[col] = X_all[col].fillna(0)



for col in mode_missing:

    X_all[col] = X_all[col].fillna(X_all[col].mode()[0])



for col in none_missing:

    X_all[col] = X_all[col].fillna('none')

    

# Check the final result

X_all.isna().sum()[X_all.isna().sum() > 0]
# One-Hot encoding

print(X_all.shape)

X_all = pd.get_dummies(X_all).reset_index(drop=True)

print(X_all.shape)
# Label encoding

# Create the list of categorical columns

# categorical_cols = [cname for cname in X_all.columns if X_all[cname].dtype == 'object']



# encoder = LabelEncoder()

# for col in categorical_cols:

#     X_all[col] = encoder.fit_transform(X_all[col])
numeric_features = [col for col in X_all.columns if X_all[col].dtype in ['float64', 'int64']]



mean = X_all[numeric_features].mean(axis=0)

std = X_all[numeric_features].std(axis=0)



X_all[numeric_features] -= mean # centering

X_all[numeric_features] /= std # scaling
# Return train and test sets 

X_new = X_all.iloc[:1460, :]

X_test_new = X_all.iloc[1460: , :]



# Create data sets for training (80%) and validation (20%)

X_train, X_valid, y_train, y_valid = train_test_split(X_new, y, train_size=0.8, test_size=0.2, random_state=0)
X_new.head(5)
# The basic model

params = {'random_state': 0}



model = XGBRegressor(**params)



model.fit(X_train, y_train, verbose=False)



preds = model.predict(X_valid)

print('Valid MAE of the basic model: {}'.format(mean_absolute_error(preds, y_valid)))
# The best model

params = {'n_estimators': 4000,

          'max_depth': 6,

          'min_child_weight': 3,

          'learning_rate': 0.02,

          'subsample': 0.7,

          'random_state': 0}



model = XGBRegressor(**params)



model.fit(X_train, y_train, verbose=False)



preds = model.predict(X_valid)

print('Valid MAE of the best model: {}'.format(mean_absolute_error(preds, y_valid)))
params = {'n_estimators': 4000,

          'max_depth': 6,

          'min_child_weight': 3,

          'learning_rate': 0.02,

          'subsample': 0.7,

          'random_state': 0}



model = XGBRegressor(**params)



model.fit(X_new, y, verbose=False)
predictions = model.predict(X_test_new)



output = pd.DataFrame({'Id': X_test.index,

                       'SalePrice': predictions})

output.to_csv('../../kaggle/working/submission.csv', index=False)