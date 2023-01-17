# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import seaborn as sns

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split,GridSearchCV

import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler

from imblearn.over_sampling import SMOTE

from sklearn.metrics import accuracy_score,classification_report

%matplotlib inline
df = pd.read_csv('../input/creditcard.csv')
df.head()
sns.countplot(df['Class'])

print(df['Class'].value_counts())
df['NormalizedAmount'] = StandardScaler().fit_transform(df['Amount'].values.reshape(-1,1))

df.drop(['Time','Amount'],1,inplace = True)
x = df.drop('Class',1)

y = df['Class']
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.3,random_state = 42)
sm = SMOTE(random_state=42)

x_array,y_array = sm.fit_resample(x_train,y_train)
x_train = pd.DataFrame(x_array,columns=x.columns)

y_train = pd.Series(y_array)
x_train.head()
params= {'C':np.linspace(1,10,10)}

grid = GridSearchCV(LogisticRegression(),params,cv=10,verbose=3,n_jobs=1)
grid.fit(x_array,y_array)
grid.best_estimator_
prediction = grid.predict(x_test)
accuracy_score(y_test,prediction)
from sklearn.metrics import roc_auc_score, auc,roc_curve

fpr, tpr, thresholds = roc_curve(y_test, prediction, pos_label=1)

auc_algo =  auc(fpr, tpr)

print(auc_algo)