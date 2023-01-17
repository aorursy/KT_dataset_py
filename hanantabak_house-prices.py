# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from xgboost import XGBRegressor

from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_absolute_error

from sklearn.ensemble import IsolationForest





# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
X = pd.read_csv('../input/home-data-for-ml-course/train.csv')

X_test_full = pd.read_csv('../input/home-data-for-ml-course/test.csv')



X.dropna(axis=0, subset=['SalePrice'],inplace=True)

y = X.SalePrice

X.drop(['SalePrice'],axis=1,inplace=True)

X.dropna(axis=1,inplace=True)
X_train_full,X_valid_full,y_train,y_valid = train_test_split(X,y,train_size=0.8,test_size=0.2,random_state=0)

cat_cols = [col for col in X_train_full.columns if X_train_full[col].nunique()<10 and X_train_full[col].dtype == "object"]

num_cols = [col for col in X_train_full.columns if X_train_full[col].dtype in ['int64','float64']]

my_cols = cat_cols + num_cols

X_train = X_train_full[my_cols]

X_valid = X_valid_full[my_cols]

X_test = X_test_full[my_cols]

X_train = pd.get_dummies(X_train)

X_valid = pd.get_dummies(X_valid)

X_test = pd.get_dummies(X_test)

X_train, X_valid = X_train.align(X_valid, join='left', axis=1)

X_train, X_test = X_train.align(X_test, join='left', axis=1)

iso = IsolationForest(contamination=0.1)

yhat = iso.fit_predict(X_train)

X_train, y_train = X_train[np.where(yhat == 1, True, False)], y_train[np.where(yhat == 1, True, False)]
my_model = XGBRegressor(n_estimators=500, learning_rate=0.05)

my_model.fit(X_train,y_train,early_stopping_rounds=5, eval_set=[(X_valid,y_valid)],verbose=False)

pred = my_model.predict(X_valid)

print('mae =', mean_absolute_error(y_valid,pred))

pred2 = my_model.predict(X_test)

output = pd.DataFrame({'Id':X_test.Id, 'SalePrice':pred2})

output.to_csv('submission3.csv',index=False)
X.isna().sum()