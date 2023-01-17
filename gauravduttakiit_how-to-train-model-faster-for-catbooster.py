import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

pd.set_option('display.max_columns', None)
train=pd.read_csv('/kaggle/input/fashion-class-classification/fashion-mnist_train.csv')

train.head()
train.info()
test=pd.read_csv('/kaggle/input/fashion-class-classification/fashion-mnist_test.csv')

test.head()
test.info()
X_train=train.drop('label',axis=1)

y_train=train['label']

X_test=test.drop('label',axis=1)

y_test=test['label']
from catboost import CatBoostClassifier

model=CatBoostClassifier(task_type="GPU")
model.fit(X_train,y_train)