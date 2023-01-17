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
import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

from pandas import DataFrame

from imblearn.over_sampling import SMOTE
test = pd.read_csv("/Users/kimura/Desktop/test.csv",index_col=0)

train = pd.read_csv("/Users/kimura/Desktop/train.csv",index_col=0)
train.groupby('Class').count()
X = train.drop('Class', axis=1).values

y = train['Class'].values
from collections import Counter

from sklearn.datasets import make_classification

from imblearn.over_sampling import SMOTE

print ( 'Original dataset shape  %s ' % Counter ( y ))

sm = SMOTE ( random_state = 42 )

X, y = sm . fit_resample ( X , y )

print ( 'Resampled dataset shape  %s ' % Counter ( y ))
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier()

model.fit(X, y)
predict['Class'].value_counts()
import lightgbm as lgb

from sklearn import datasets

from sklearn.model_selection import train_test_split

import numpy as np



params = {

    'n_estimators': 10000

}



model = lgb.LGBMClassifier(**params)

model.fit(X, y,eval_set=[(X, y)], early_stopping_rounds=100)
X = test.values

p = model.predict(X)

predict = pd.read_csv('/Users/kimura/Desktop/sampleSubmission.csv')

predict['Class'] = p

predict.to_csv('LGBM2.csv',index=None)