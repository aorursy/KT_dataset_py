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


import lightgbm as lgb   

from sklearn.metrics import mean_squared_error

from sklearn.model_selection import GridSearchCV
df_test = pd.read_csv('/kaggle/input/Kannada-MNIST/test.csv')

df_train = pd.read_csv('/kaggle/input/Kannada-MNIST/train.csv')

df_vaild = pd.read_csv('/kaggle/input/Kannada-MNIST/Dig-MNIST.csv')
df_train.head()
df_vaild.head()

y_train = df_train['label']

x_train = df_train.drop(['label'],axis=1)

y_vaild = df_vaild['label']

x_vaild = df_vaild.drop(['label'],axis=1)
lgb_params = {

            "objective" : "multiclass",

            "metric" : "multi_logloss",

            "num_class" : 10,

            "max_depth" : 5,

            "num_leaves" : 30,

            "learning_rate" : 0.1,

            "bagging_fraction" : 0.4,

            "feature_fraction" : 0.7,

            "lambda_l1" : 0,

            "lambda_l2" : 0,}





lgtrain = lgb.Dataset(x_train, label=y_train)

lgtest = lgb.Dataset(x_vaild, label=y_vaild)

lgb_clf = lgb.train(lgb_params, lgtrain, 1000, 

                    valid_sets=[lgtrain, lgtest], 

                    early_stopping_rounds=5, 

                    verbose_eval=10)

res = lgb_clf.predict(df_test).argmax(axis=1)

submission = pd.read_csv('../input/Kannada-MNIST/sample_submission.csv')

submission['label'] = res

submission.to_csv('submission.csv', index=False)