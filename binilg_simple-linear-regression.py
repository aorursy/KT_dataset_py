import pandas as pd

import numpy as np

import seaborn as sns

%matplotlib inline
train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')

id_test = test['id'].values
test = test.drop('id',axis=1)
test.shape
train.head()
test.head()
#looking for missing data in train

sns.heatmap(train.isnull(),yticklabels=False,cbar=False,cmap='viridis')
#looking for missing data in test

sns.heatmap(test.isnull(),yticklabels=False,cbar=False,cmap='viridis')
train.info()
test.info()
y = train['target']

X = train.drop('target',axis=1)

X = X.drop('id',axis=1)
X.shape
from sklearn.linear_model import LinearRegression
logmodel = LinearRegression()
logmodel.fit(X,y)
pred = logmodel.predict(test)
sub = pd.DataFrame()

sub['id'] = id_test

sub['target'] = pred

sub.loc[sub['target']<0, 'target']=0
sub.to_csv('sub.csv',index=False)