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
train = pd.read_csv('../input/learn-together/train.csv')

train.head()
test = pd.read_csv('../input/learn-together/test.csv')

test.head()
sample_submission = pd.read_csv('../input/learn-together/sample_submission.csv')

sample_submission.head()
train.describe()
import pandas_profiling as pdp

pdp.ProfileReport(train)
X_train = train.drop(['Cover_Type'], axis=1)

Y_train = train['Cover_Type']
import lightgbm as lgb

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score



train_x, valid_x, train_y, valid_y = train_test_split(X_train, Y_train, test_size=0.33, random_state=0)



gbm = lgb.LGBMClassifier(objective='binary')



gbm.fit(train_x, train_y, eval_set = [(valid_x, valid_y)],

       early_stopping_rounds=20,

       verbose=10

       )
oof = gbm.predict(valid_x, num_iteration=gbm.best_iteration_)

print('score', round(accuracy_score(valid_y, oof)*100,2))
test_pred = gbm.predict(test, num_iteration=gbm.best_iteration_)

sample_submission['Cover_Type'] = test_pred

sample_submission.to_csv('submission.csv', index=False)
!ls