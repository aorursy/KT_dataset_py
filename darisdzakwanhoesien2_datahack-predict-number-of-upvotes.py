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
submission = pd.read_csv("../input/predict-number-of-upvotes/sample_submission_OR5kZa5.csv")

train = pd.read_csv("../input/codefest/train_NIR5Yl1.csv")

test = pd.read_csv("../input/codefest/test_8i3B3FC.csv")
train
test
test = test.merge(submission, on='ID')

test
train_test = pd.concat([train,test], ignore_index=True)

train_test
train_test.isnull().sum()
non_number_train_test_columns = train_test.dtypes[train_test.dtypes == object].index.values

non_number_train_test_columns
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

train_test['Tag'] = le.fit_transform(train_test['Tag']).astype(np.int64)

train_test = train_test.drop(['ID','Username'],axis=1)

train_test
x_train = train_test.iloc[:len(train)*9//10].drop(['Upvotes'], axis=1)

x_val = train_test.iloc[len(train)*9//10:].drop(['Upvotes'], axis=1)



y_train = train_test.iloc[:len(train)*9//10]['Upvotes']

y_val = train_test.iloc[len(train)*9//10:]['Upvotes']
import time

from xgboost import XGBRegressor

ts = time.time()



model = XGBRegressor(

    max_depth=10,

    n_estimators=1000,

    min_child_weight=0.5, 

    colsample_bytree=0.8, 

    subsample=0.8, 

    eta=0.1,

#     tree_method='gpu_hist',

    seed=42)



model.fit(

    x_train, 

    y_train, 

    eval_metric="rmse", 

    eval_set=[(x_train, y_train), (x_val, y_val)], 

    verbose=True, 

    early_stopping_rounds = 20)



time.time() - ts
x_test = train_test.iloc[len(train):].drop(['Upvotes'], axis=1)



Y_pred = model.predict(x_val).clip(0, 20)

Y_test = model.predict(x_test)

np.round(Y_test)
submission['Upvotes'] = np.round(Y_test)

submission.to_csv('submission.csv',index=False)

submission