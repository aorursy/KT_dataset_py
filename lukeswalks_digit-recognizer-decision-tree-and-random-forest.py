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
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import matplotlib.image as mpimg

import seaborn as sns

from sklearn.ensemble import RandomForestClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import classification_report,confusion_matrix

%matplotlib inline



# Load the data

train = pd.read_csv("/kaggle/input/train.csv")

test = pd.read_csv("/kaggle/input/test.csv")
from sklearn.model_selection import train_test_split

X = train.drop(columns='label')

y = train['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=101)
#Decision Tree

dtree = DecisionTreeClassifier()

dtree.fit(X_train,y_train)

predictions = dtree.predict(X_test)

print(classification_report(y_test,predictions))

print(confusion_matrix(y_test,predictions))
#1000 Tree

rfc1000 = RandomForestClassifier(n_estimators=1000)

rfc1000.fit(X_train,y_train)

predictions = rfc1000.predict(X_test)

print(classification_report(y_test,predictions))

print(confusion_matrix(y_test,predictions))
#1000 Tree on Test

test['label']  = rfc1000.predict(test)

submission = pd.DataFrame()

submission['Label'] = test['label']

submission.index = np.arange(1, 28001)

submission.index.name = 'ImageId'

submission.to_csv("submission.csv")