# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.metrics import classification_report



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df = pd.read_csv('../input/mushroom-classification/mushrooms.csv')

df.head(3)
from sklearn.tree import DecisionTreeClassifier

mushrooms=df.applymap(lambda x: ord(x))

mushrooms.head(3)
X = mushrooms.drop('class',axis='columns')

y = mushrooms['class']

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.33)
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(n_estimators=2)

model.fit(X_train, y_train)
model.score(X_test, y_test)
y_predicted = model.predict(X_test)

classification_report(y_test,y_predicted)


from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_predicted)

cm


%matplotlib inline

import matplotlib.pyplot as plt

import seaborn as sn

plt.figure(figsize=(8,5))

sn.heatmap(cm, annot=True)

plt.xlabel('Predicted')

plt.ylabel('Truth')
from sklearn.neural_network import MLPClassifier

clf = MLPClassifier(hidden_layer_sizes=18, activation='logistic', max_iter=320)

clf.fit(X_train, y_train)
clf.score(X_test, y_test)
y_pred_red = clf.predict(X_test)

classification_report(y_test,y_pred_red)
cm_red = confusion_matrix(y_test, y_pred_red)

cm_red
%matplotlib inline

import matplotlib.pyplot as plt

import seaborn as sn

plt.figure(figsize=(8,5))

sn.heatmap(cm_red, annot=True)

plt.xlabel('Predicted')

plt.ylabel('Truth')
from sklearn.metrics import mean_squared_error

mean_squared_error(y_test, y_pred_red)