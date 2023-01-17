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
train = pd.read_csv('/kaggle/input/liverpool-ion-switching/train.csv')

submission = pd.read_csv('/kaggle/input/liverpool-ion-switching/sample_submission.csv')
train.shape
train.head(10)
train_time=train['time'].values
train_time
train_time_0 = train_time[:50000]
train_time_0.shape
for i in range(1,100):

    train_time_0 = np.hstack([train_time_0, train_time[i*50000:(i+1)*50000]])
train_time_0.shape
train['time'] = train_time_0
test = pd.read_csv('/kaggle/input/liverpool-ion-switching/test.csv')

test.head()
train_time_0 = train_time[:50000]

for i in range(1,40):

    train_time_0 = np.hstack([train_time_0, train_time[i*50000:(i+1)*50000]])

test['time'] = train_time_0
X = train[['time', 'signal']].values

y = train['open_channels'].values
from sklearn.tree import DecisionTreeClassifier

my_tree=DecisionTreeClassifier(random_state=0)

my_tree.fit(X,y)
X_test = test[['time', 'signal']].values
submission.head()
test_preds = my_tree.predict(X_test)

test_preds
submission['open_channels'] = test_preds
submission.head()
submission['open_channels'].mean()
submission['time'] = [format(submission.time.values[x], '.4f') for x in range(2000000)]
submission.time.values[:20]
submission.to_csv('submission.csv', index=False)