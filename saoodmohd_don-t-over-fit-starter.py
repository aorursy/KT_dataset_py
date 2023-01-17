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
import pandas as pd

data = pd.read_csv('../input/train.csv')

labels = data['target'].values

data = data.drop(['id','target'],axis=1).values
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2)

print(X_train.shape,y_train.shape,X_test.shape,y_test.shape)
from sklearn.linear_model import LogisticRegression

clf = LogisticRegression(solver='saga',multi_class ='multinomial')

clf.fit(X_train, y_train)
def evaluate(pred,labels):

    return sum(pred == labels)/len(pred)
pred = clf.predict(X_test)

print("Accuracy:",evaluate(pred,y_test))
from sklearn.ensemble import RandomForestClassifier

rfc = RandomForestClassifier(n_jobs=-1, n_estimators=5)

rfc.fit(X_train, y_train)
pred = rfc.predict(X_test)

print("Accuracy:",evaluate(pred,y_test))
unknown = pd.read_csv("../input/test.csv")

ids = unknown['id']

unknown = unknown.drop(['id'],axis=1).values

pred = clf.predict(unknown)
out = pd.DataFrame()

out['id'] = ids

out['target'] = pred

print(out.info())

out.head(10)
out.to_csv('./out.csv',index=False)