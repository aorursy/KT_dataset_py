# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import lightgbm as lgb

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train_data = pd.read_csv("/kaggle/input/titanic/train.csv")
train_data.head()
test_data = pd.read_csv("/kaggle/input/titanic/test.csv")
test_data.head()
y = train_data["Survived"]

features = ["Pclass", "Sex", "SibSp", "Parch"]
X_train = pd.get_dummies(train_data[features])
X_test = pd.get_dummies(test_data[features])

train_age = train_data["Age"]
test_age = test_data["Age"]

X_train = pd.concat([train_age, X_train],axis=1)
X_test = pd.concat([test_age, X_test],axis=1)
lgb_train = lgb.Dataset(X_train,y)
print(lgb_train)
lgb_params = {"objective":"binary", 
              "metric":"binary_error",
              "learning_rate": 0.05, 
              "max_depth": 3,
              "feature_fraction": 0.7
              }
gbm = lgb.train(lgb_params,
                lgb_train,
                num_boost_round=500,
                verbose_eval=50)
pred= gbm.predict(X_test)
pred_temp = np.where(pred < 0.5, 0, pred)
pred_temp = np.where(pred >= 0.5,1, pred_temp)
print(pred_temp)
pred_df = pd.DataFrame(pred_temp)
pred_df= pred_df.rename(columns={0: 'pred'})
pred_df = pred_df.astype(int)
test_Pass = test_data["PassengerId"]
pred_df = pd.concat([test_Pass, pred_df],axis=1)
pred_df= pred_df.rename(columns={"pred": 'Survived'})
pred_df.head()
pred_df.to_csv('my_submission.csv', index=False)