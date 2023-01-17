# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn import metrics
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
train_set = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv')
test_set = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/test.csv')
sample_sub = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/sample_submission.csv')
sns.distplot(train_set['SalePrice'])
missing_train = train_set.isnull().sum()
missing_train = missing_train[missing_train > 0]
missing_train.sort_values(inplace = True)
missing_train.plot.bar()
missing_test = test_set.isnull().sum()
missing_test = missing_test[missing_test > 0]
missing_test.sort_values(inplace = True)
missing_test.plot.bar()
# Take k features with highest correlation
k = 10
label = 'SalePrice'
columns = correlation_matrix.nlargest(k, label)[label].index
correlation_matrix = np.corrcoef(train_set[columns].values.T)

f, ax = plt.subplots(figsize=(20, 20))

heatmap = sns.heatmap(correlation_matrix, cbar = True, annot = True, square = True, yticklabels = columns.values, xticklabels = columns.values)

plt.show()
##### fill NaN values
for i in train_set:
    if train_set[i].dtype == 'object':
        train_set[i] = train_set[i].fillna(train_set[i].mode())
    else:
        train_set[i] = train_set[i].fillna(train_set[i].median())
def label_encoder(df_train, df_test):
    le_count = 0
    
    for col in df_train:
        if df_train[col].dtype == 'object':
            if len(list(df_train[col].unique())) <= 2:
                le = LabelEncoder()
                le.fit(list(df_train[col].unique())+list(df_test[col].unique()))

                df_train[col] = le.transform(df_train[col].astype(str))
                df_test[col] = le.transform(df_test[col].astype(str))
                le_count +=1;
               
    
    print("Total label encoded columns : %d " %le_count)

label_encoder(train_set, test_set)
train_set.head()
train_len = len(train_set)

data = pd.concat(objs = [train_set, test_set], axis = 0)

data = pd.get_dummies(data)
import copy
train_set = copy.copy(data[:train_len])

test_set = copy.copy(data[train_len:])

test_set = test_set.drop([label], axis = 1)
# Extract labels
corr = train_set.corr().sort_values(label)
cols = corr[label][corr[label].values > 0.05].index.values
train_label = train_set[label]
cols = np.delete(cols, len(cols) - 1)
train_sample = train_set[cols]
test_sample = test_set[cols]
# Imputation
imputer = SimpleImputer(strategy = 'median')
imputer.fit(train_sample)

train_sample = imputer.transform(train_sample)
test_sample = imputer.transform(test_sample)
# Normalization
scaler = StandardScaler()

scaler.fit(train_sample)

train_sample = scaler.transform(train_sample)
test_sample = scaler.transform(test_sample)
# train test split

X_train, X_test , y_train, y_test = train_test_split(train_sample, train_label, train_size = 0.8, random_state = 64)
X_train.shape, X_test.shape, test_sample.shape
rf = RandomForestRegressor(n_estimators = 10000, random_state = 64)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)
sns.set()
plt.figure(figsize = (16, 16))
plt.scatter(y_test, y_pred)
plt.title('Actual vs Predicted')
plt.xlabel('Actual Sale Price')
plt.ylabel('Predicted Sale Price')

plt.plot()
test_pred = rf.predict(test_sample)

submission = pd.DataFrame()

submission['ID'] = testID

submission['SalePrice'] = test_pred
submission.head()

submission.to_csv('house_price_rf.csv', index = False)
