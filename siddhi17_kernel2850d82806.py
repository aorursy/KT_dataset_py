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
train_df = pd.read_csv('../input/train.csv')

test_df = pd.read_csv('../input/test.csv')

labels = train_df['target'].values

data = train_df.drop(['id','target'],axis=1).values



data_test = test_df.drop(['id'],axis=1).values
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler 



X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2)



scaler = StandardScaler() 

X_train = scaler.fit_transform(X_train)

X_test = scaler.transform(X_test)

data_test = scaler.transform(data_test)
from sklearn.linear_model import LogisticRegression

logisticRegr = LogisticRegression()



logisticRegr.fit(X_train, y_train)
from sklearn import metrics



pred = logisticRegr.predict(X_test)

print(metrics.accuracy_score(y_test, pred))
ids = test_df['id']

y_pred = logisticRegr.predict(data_test)

out = pd.DataFrame()

out['id'] = ids

out['target'] = y_pred

print(out.info())

out.head(10)
out.to_csv('submission.csv',index=False)
