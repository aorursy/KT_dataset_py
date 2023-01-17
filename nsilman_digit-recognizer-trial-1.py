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
train=pd.read_csv('../input/train.csv')

test=pd.read_csv('../input/test.csv')



X=train.drop(train[['label']], axis=1)

y=train['label']

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
from sklearn.neural_network import MLPClassifier

from sklearn.model_selection import GridSearchCV



clf=MLPClassifier(activation='relu', alpha=0.001, hidden_layer_sizes=(700,))

clf.fit(X_train, y_train)

ypred=clf.predict(test)


ypre=pd.DataFrame(ypred)



ypre.index=test.index

ypre.columns=['Label']

ypre.index.rename('ImageId', inplace=True)

ypre.index+=1

ypre.to_csv('6dr.csv')