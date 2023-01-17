# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
df_train = pd.read_csv("../input/train.csv", low_memory = True,nrows=20000000)
df_test=pd.read_csv("../input/test.csv")
df_train.head()
df_test.head()
df_train=df_train.drop(['attributed_time'],axis=1)
df_train.columns
df_train.info()
# Any missing values
df_train.isnull().sum()
df_train['is_attributed'].value_counts()
df_train=df_train.drop(['click_time'],axis=1)
df_train.columns
df_train.head()
X=df_train.drop(['is_attributed'],axis=1)
y=df_train['is_attributed']
from sklearn.linear_model import LogisticRegression
model=LogisticRegression()
model.fit(X,y)
X_test=df_test.drop(['click_id','click_time'],axis=1)
y_pred=model.predict(X_test)
set(y_pred)
y_predict= pd.DataFrame(y_pred, columns=['is_attributed'])
y_predict.head()
id=df_test['click_id']
test_submission = pd.read_csv("../input/sample_submission.csv")
test_submission.head()
test_submission['is_attributed'] = y_predict
test_submission.head()
test_submission.to_csv('submission_logistic_.csv', index=False)