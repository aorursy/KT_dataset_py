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
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
data=pd.read_csv('/kaggle/input/breast-cancer-wisconsin-data/data.csv')
data.head()
columns=data.columns
columns
np.unique(data['diagnosis'])


data.loc[data.diagnosis == "M", "diagnosis"] = 1
data.loc[data.diagnosis == "B", "diagnosis"] = 0
data.info()
del data['Unnamed: 32']
del data['id']
data.isnull().sum()
data.describe()
data.corr()
y=data.pop('diagnosis')
x=data
y=y.astype(int)
x=x.astype(float)
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score,confusion_matrix
from sklearn.metrics import accuracy_score


from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
x_train,x_test,y_train,y_test=train_test_split(x,y)
model1=RandomForestClassifier()
model1.fit(x_train,y_train)
y_pre=model1.predict(x_test)
print(f1_score(y_test,y_pre))
print()
print(confusion_matrix(y_test,y_pre))
print()
print(accuracy_score(y_test,y_pre))
model2=LogisticRegression()
model2.fit(x_train,y_train)
y_pre=model1.predict(x_test)
print(f1_score(y_test,y_pre))
print()
print(confusion_matrix(y_test,y_pre))
print()
print(accuracy_score(y_test,y_pre))
model3=SVC()
model3.fit(x_train,y_train)
y_pre=model1.predict(x_test)
print(f1_score(y_test,y_pre))
print()
print(confusion_matrix(y_test,y_pre))
print()
print(accuracy_score(y_test,y_pre))
model1=MLPClassifier(hidden_layer_sizes=(150,150))
model1.fit(x_train,y_train)
y_pre=model1.predict(x_test)
print(f1_score(y_test,y_pre))
print()
print(confusion_matrix(y_test,y_pre))
print()
print(accuracy_score(y_test,y_pre))