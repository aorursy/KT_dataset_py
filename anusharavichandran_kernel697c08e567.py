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
import matplotlib.pyplot as plt
import pandas as pd
dataset = pd.read_csv('../input/creditcardcsv/creditcard.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values
from sklearn.compose import ColumnTransformer 
from sklearn.preprocessing import  StandardScaler 
column_trans = ColumnTransformer(transformers=[('scaler',StandardScaler(),[0])],remainder='passthrough') 
X=np.array(column_trans.fit_transform(X))
print(X[0][0])
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
print(cm)
accuracy_score(y_test, y_pred)
print(classifier.predict(column_trans.transform([[8808,-4.617217204,1.695693653,-3.114372201,4.328198553,-1.873256991,-0.989908136,-4.577264627,0.472216158,0.472016953,-5.576022636,4.802322761,-10.83316447,0.104303876,-9.405423062,-0.807477869,-7.552342204,-9.80256179,-4.120628835,1.74050729,-0.039045934,0.481829697,0.146023056,0.117038528,-0.217564599,-0.138776044,-0.424452881,-1.002041426,0.890780288,1.1
]])))
print(classifier.predict(column_trans.transform([[1,-1.358354062,-1.340163075,1.773209343,0.379779593,-0.503198133,1.800499381,0.791460956,0.247675787,-1.514654323,0.207642865,0.624501459,0.066083685,0.717292731,-0.165945923,2.345864949,-2.890083194,1.109969379,-0.121359313,-2.261857095,0.524979725,0.247998153,0.771679402,0.909412262,-0.689280956,-0.327641834,-0.139096572,-0.055352794,-0.059751841,378.66
]])))
from sklearn.metrics import classification_report

print(classification_report(y_test, y_pred))