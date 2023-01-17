import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

import seaborn as sns



train = pd.read_csv(os.path.join('../input', 'train.csv'))

test  = pd.read_csv(os.path.join('../input', 'test.csv'))
train.info()
train.head(10)
train['Name_Len'] = train['Name'].apply(lambda x: len(x))

train[['Survived','Fare','Pclass']].groupby(pd.qcut(train['Name_Len'],6)).mean()
train['Survived'].groupby(train['Sex']).mean()
bins = [0,10,20,30,40,50,60,70,80]

train['Survived'].groupby(pd.cut(train['Age'], bins)).mean()
