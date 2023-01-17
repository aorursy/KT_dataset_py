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
df0 = pd.read_csv("/kaggle/input/emg-4/2.csv")

df1 = pd.read_csv("/kaggle/input/emg-4/3.csv")

df2 = pd.read_csv("/kaggle/input/emg-4/0.csv")

df3 = pd.read_csv("/kaggle/input/emg-4/1.csv")



column_names = ['signal_' + str(i) for i in range(65)]
df0.columns = column_names

df1.columns = column_names

df2.columns = column_names

df3.columns = column_names



df0['label'] = 0

df1['label'] = 1

df2['label'] = 2

df3['label'] = 3
df = pd.concat([df0,df1,df2,df3],axis=0)

df.drop(['signal_64'],axis=1,inplace = True)
X = df.drop(['label'],axis=1)

X = X.iloc[:,:]

y = df['label']
from sklearn.model_selection import train_test_split



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)
from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.linear_model import LogisticRegression

dt_params = None

rf_params = None

lr_params = None



dt = DecisionTreeClassifier()

rf = RandomForestClassifier()

lr = LogisticRegression()
rf.fit(X_train,y_train)

y_pred = rf.predict(X_test)

y_pred_proba = rf.predict_proba(X_test)
y_pred_proba
from sklearn.metrics import classification_report



print(classification_report(y_test,y_pred))
import scikitplot as skplt

import matplotlib.pyplot as plt



y_true = [1,0,1,1,0]

y_probas = [1,0,1,0.5,0.8]

skplt.metrics.plot_roc_curve(y_test, y_pred_proba)

plt.show()