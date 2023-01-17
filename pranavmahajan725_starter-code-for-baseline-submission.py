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
train_df = pd.read_csv('/kaggle/input/cte-ml-hack-2019/train_real.csv')

test_df = pd.read_csv('/kaggle/input/cte-ml-hack-2019/test_real.csv')
train_df.head()
train_df.dtypes
X_train = train_df.drop(['Id', 'label', 'Soil'], axis=1)

Y_train = train_df['label']
X_train.head()
Y_train.head()
X_test = test_df.drop(['Id', 'Soil'], axis=1)

X_test.head()
# we can try to fit the base model 

# we can try logistic regression

from sklearn.linear_model import LogisticRegressionCV

from sklearn.metrics import roc_auc_score



clf=LogisticRegressionCV(cv=5, max_iter = 1000).fit(X_train, Y_train)
train_res=clf.predict(X_train)

train_res
test_res = clf.predict(X_test)

test_res
submission_df = pd.DataFrame()

submission_df['Id'] = test_df['Id']
submission_df['Predicted'] = test_res.tolist()
submission_df.tail()
submission_df.to_csv('vanilla_logistic_submission.csv',index=False)
!ls