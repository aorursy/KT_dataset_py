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
from sklearn.model_selection import train_test_split, GridSearchCV

from sklearn.ensemble import RandomForestClassifier



data_path = '../input/'
df_train = pd.read_csv(data_path + 'train.csv', header=None)

df_test = pd.read_csv(data_path + 'test.csv', header=None)

df_trainlabels = pd.read_csv(data_path + 'trainLabels.csv', header=None)



print(df_train.shape)

print(df_test.shape)

print(df_trainlabels.shape)
df_train.info()
df_train.head()
df_trainlabels.head()
X, y = df_train, np.ravel(df_trainlabels)

X_train,X_test,y_train,y_test = train_test_split(X, y, test_size=0.3, random_state=0)
clf = RandomForestClassifier()



clf.fit(X_train, y_train)



prd = clf.predict(df_test)
prd.shape
Ids = np.arange(1,prd.shape[0]+1)

Ids
submission = pd.DataFrame({'Id': Ids, 'Solution':prd})

submission.to_csv('submission.csv', index=False)