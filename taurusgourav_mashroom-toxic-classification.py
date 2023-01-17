# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline

df = pd.read_csv('/kaggle/input/mushroom-classification/mushrooms.csv')
df.head()
df.columns
print('The dataset has {} columns and {} rows'.format(df.shape[1], df.shape[0]))
df.isnull().sum()
#Label Encoder

from sklearn.preprocessing import LabelEncoder

label = LabelEncoder()

for col in df.columns:

    df[col] = label.fit_transform(df[col])

df.head()

y = df['class']

X = df.drop(['class'], axis =1, inplace = False)
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

scaler.fit_transform(X)
from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix

from sklearn.metrics import classification_report

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.3)
from sklearn.neighbors import KNeighborsClassifier
model = KNeighborsClassifier(n_neighbors=5)

model.fit(X_train, y_train)
pred = model.predict(X_test)
print(confusion_matrix(y_test, pred))
print(classification_report(y_test, pred))
from sklearn.model_selection import cross_val_score

from sklearn import svm
clf = svm.SVC(kernel='linear', C=1)

scores = cross_val_score(clf, X,y, cv=5)

scores
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
#logistic regression
from sklearn.linear_model import LogisticRegression
lmodel = LogisticRegression()

lmodel.fit(X_train, y_train)
lm_predict = lmodel.predict(X_test)
confusion_matrix(y_test, lm_predict)
print(classification_report(y_test, lm_predict))