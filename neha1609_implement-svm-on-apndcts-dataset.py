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
data= pd.read_csv("/kaggle/input/apndcts/apndcts.csv")

X=data.iloc[:, :-1]

Y= data.iloc[:, 7]

print(data.describe())

print (X)

print(Y)
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.4, random_state=100)

print(y_test)
from sklearn import svm

from sklearn.model_selection import cross_val_score

clf = svm.SVC(kernel='linear',gamma='auto', C=1)
pre = clf.fit(X_train,y_train)
clf.score(X_test,y_test)