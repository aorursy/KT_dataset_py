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
train = pd.read_csv('/kaggle/input/train.csv')

test = pd.read_csv('/kaggle/input/test.csv')

#creating dataframe for the required output

submission = pd.DataFrame()

submission['id'] = test['id']
train.isna().sum()
train.head()
train.drop(['id'],1,inplace=True)

test.drop(['id'],1,inplace=True)
from sklearn.model_selection import train_test_split

y = train['target']

train.drop(['target'],1,inplace=True)

#train validation split in 80:20 ratio

X_train, X_val, y_train, y_val = train_test_split(train, y, test_size=0.20, random_state=42, shuffle=True)  
from xgboost.sklearn import XGBRegressor

xgb_reg = XGBRegressor(learning_rate=0.09, max_depth=6,seed=0)
xgb_reg.fit(X_train, y_train)
from sklearn.metrics import mean_squared_error, r2_score

y_pred = xgb_reg.predict(X_val)

rmse = np.sqrt(mean_squared_error(y_pred, y_val))

print(rmse)

r2 = r2_score(y_val, y_pred)

print("R2 Score:", r2)
xgb_reg = XGBRegressor(learning_rate=0.09, max_depth=6, min_child_weight=40,seed=0)

xgb_reg.fit(train, y)
y_pred = xgb_reg.predict(test)

submission['target'] = y_pred

submission = submission.sort_values(by=['id'],ascending=False)

submission.to_csv('scorecheck.csv',index=False)