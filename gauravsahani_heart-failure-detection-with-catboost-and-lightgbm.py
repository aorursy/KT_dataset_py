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

import numpy as npo

import matplotlib.pyplot as plt

import seaborn as sns

import sklearn
df=pd.read_csv('../input/heart-failure-clinical-data/heart_failure_clinical_records_dataset.csv')
df.head()
df.corr()
plt.matshow(df.corr())

plt.show()
df.info()
df=df.drop(['time'],axis=1)
X=df.drop(['DEATH_EVENT'],axis=1)

y=df[['DEATH_EVENT']]
import keras

from keras.models import Sequential

from keras.layers import Dense

from keras.layers import LeakyReLU,PReLU,ELU

from keras.layers import Dropout
X.shape
y.shape
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 1)
from catboost import CatBoostClassifier

CB = CatBoostClassifier(iterations=124,learning_rate=0.3018,depth=3)





CB.fit(X_train, y_train,eval_set=(X_test, y_test))
pred = CB.predict(X_test)



from sklearn.metrics import accuracy_score,confusion_matrix

print(accuracy_score(y_test, pred))

print(confusion_matrix(y_test, pred))
import lightgbm as lgb

clf = lgb.LGBMClassifier(max_depth=-2, min_child_samples=28, n_estimators=365,

                         num_leaves=16, learning_rate=0.07592)

clf.fit(X_train, y_train)
pred1 = clf.predict(X_test)



from sklearn.metrics import accuracy_score,confusion_matrix

print(accuracy_score(y_test, pred1))

print(confusion_matrix(y_test, pred1))
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, pred1)

print('Confusion matrix\n\n', cm)

print('\nTrue Positives(TP) = ', cm[0,0])

print('\nTrue Negatives(TN) = ', cm[1,1])

print('\nFalse Positives(FP) = ', cm[0,1])

print('\nFalse Negatives(FN) = ', cm[1,0])
cm_matrix = pd.DataFrame(data=cm, columns=['Actual Positive:1', 'Actual Negative:0'], 

                                 index=['Predict Positive:1', 'Predict Negative:0'])



sns.heatmap(cm_matrix, annot=True, fmt='d', cmap='YlGnBu')
from sklearn.metrics import classification_report

print(classification_report(y_test, pred1))