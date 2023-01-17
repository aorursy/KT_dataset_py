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
pd.set_option('display.max.columns',3000)
# load train data

train=pd.read_csv('/kaggle/input/knit-hacks/train.csv')

test=pd.read_csv('/kaggle/input/knit-hacks/test.csv')
# see target distribution

train['Col2'].value_counts()/train.shape[0]
## Unbalanced problem
# check missing values 

train.isnull().sum().any()
# check catogorical columns

cat_col=[col for col in train.columns if train[col].dtype=='O']
# drop cat

train=train.drop(cat_col,axis=1)
# from test also

test_new=test.copy(deep=True)
test_new=test_new.drop(cat_col,axis=1)
# fill missing

train=train.fillna(0)
# test also

test_new=test_new.fillna(0)
# installation

! pip install -U sliced
train_copy=train.copy(deep=True)
train_copy=train_copy.drop('Col2',axis=1)
### remove zero variance features

from sklearn.feature_selection import VarianceThreshold

vt=VarianceThreshold(threshold=0)

vt_x=vt.fit_transform(train_copy)
vt_test=vt.transform(test_new)
from sliced import SlicedInverseRegression

sir = SlicedInverseRegression(n_directions=2)

sir.fit(vt_x,train['Col2'])
X_sir=sir.transform(vt_x)
X_test=sir.transform(vt_test)
from sklearn.linear_model import LogisticRegression
lr=LogisticRegression().fit(X_sir,train['Col2'])
submission=pd.DataFrame({'Col1':test.Col1,'Col2':lr.predict(X_test)})
submission.to_csv('submission_reduction.csv',index=False)