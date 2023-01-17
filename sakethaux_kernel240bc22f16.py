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
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt
df = pd.read_csv("/kaggle/input/mushroom-classification/mushrooms.csv")

df.head()
df.shape
df['class'] = df['class'].apply(lambda x : 1 if x=='p' else 0)
df.columns.values
df = pd.get_dummies(df,columns=['cap-shape', 'cap-surface', 'cap-color', 'bruises',

       'odor', 'gill-attachment', 'gill-spacing', 'gill-size',

       'gill-color', 'stalk-shape', 'stalk-root',

       'stalk-surface-above-ring', 'stalk-surface-below-ring',

       'stalk-color-above-ring', 'stalk-color-below-ring', 'veil-type',

       'veil-color', 'ring-number', 'ring-type', 'spore-print-color',

       'population', 'habitat'])
y = df['class']

X = df.drop(columns=['class'])
from sklearn.linear_model import LogisticRegression

logreg = LogisticRegression()
from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.30,random_state=0)
logreg.fit(X_train,y_train)
y_pred = logreg.predict(X_test)
from sklearn.metrics import confusion_matrix

confusion_matrix(y_test,y_pred)
logreg.score(X_test,y_test)