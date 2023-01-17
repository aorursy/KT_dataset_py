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

df = pd.read_csv('../input/breast-cancer-wisconsin-data/data.csv')
df.head()
df.isnull().sum()
df = df.drop('id', axis = 1)

df = df.drop('Unnamed: 32', axis = 1)
df.head()
from sklearn.preprocessing import LabelEncoder

label_encoder = LabelEncoder()

df['diagnosis'] = label_encoder.fit_transform(df['diagnosis'])
df.head()
X = df.iloc[:, 1:]

y = df.iloc[:, 0:1]
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state = 0)
from sklearn.svm import SVC

classifier = SVC(kernel = 'linear')

classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)

print(y_pred)
from sklearn.metrics import confusion_matrix

from sklearn.metrics import accuracy_score

print(confusion_matrix(y_test, y_pred))

accuracy_score(y_test, y_pred)*100