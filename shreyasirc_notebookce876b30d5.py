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
df_train = pd.read_csv('/kaggle/input/kharagpur-data-analytics-group-selections-2020/train.csv')
df_test = pd.read_csv('/kaggle/input/kharagpur-data-analytics-group-selections-2020/test.csv')
df_train.head()
df_train = df_train.drop(columns='Unnamed: 0')

# printing the numeric columns
num_cols = df_train._get_numeric_data().columns
num_cols
X = df_train.drop(['id','poor'], axis=1)
df_train.poor = df_train.poor.astype(int)

y = df_train['poor']
X.shape
from sklearn.preprocessing import LabelEncoder
labels = df_train['poor']
encoder = LabelEncoder()
encoded_labels = encoder.fit_transform(labels)
encoded_labels
# splitting the dataframe from train.csv further into train and test sets to see whether our model is giving a good accuracy
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, encoded_labels, test_size = 0.3, random_state = 0, stratify=encoded_labels)
X_train.shape, X_test.shape
categorical = [col for col in X_train.columns if X[col].dtypes == 'O']
for col in categorical:
    if X_train[col].isnull().mean()>0:
        print(col, (X_train[col].isnull().mean()))
for col in categorical:
    if X_test[col].isnull().mean()>0:
        print(col, (X_train[col].isnull().mean()))
# applying one-hot encoding on categorical variables
import category_encoders as ce
encoder = ce.OneHotEncoder(cols=list(categorical))

X_train = encoder.fit_transform(X_train)

X_test = encoder.transform(X_test)



X_train.shape, X_test.shape
from sklearn.preprocessing import  MinMaxScaler
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
X_train.shape
from sklearn.naive_bayes import GaussianNB


# instantiate the model
gnb = GaussianNB()


# fit the model
gnb.fit(X_train, y_train)
y_pred = gnb.predict(X_test)
from sklearn.metrics import accuracy_score

print('Model accuracy score: {0:0.4f}'. format(accuracy_score(y_test, y_pred)))
X_train.shape
X_test.shape
from xgboost import XGBClassifier
xgb = XGBClassifier(n_estimators=100)
xgb.fit(X_train, y_train)

y_pred = xgb.predict(X_test)
from sklearn.metrics import accuracy_score

print('Model accuracy score: {0:0.4f}'. format(accuracy_score(y_test, y_pred)))
X_train.shape
X.shape
X = encoder.transform(X)
X.shape
# printing the numeric columns
num_cols = df_train._get_numeric_data().columns
num_cols
y = encoded_labels
df_test = df_test.drop(columns='Unnamed: 0')
X_test_set = df_test.drop(['id'], axis=1)
X_test_set.shape

X_test_set = encoder.transform(X_test_set)

X_test_set.shape
X.shape
from xgboost import XGBClassifier
xgb = XGBClassifier(n_estimators=100)
xgb.fit(X,y)

y_pred_test = xgb.predict(X_test_set)
y_pred_test
test_id = df_test['id']
import pandas as pd
df = pd.DataFrame(data={"id": test_id, "Poor": y_pred_test})
df.to_csv("./predictions.csv", sep=',',index=False)
