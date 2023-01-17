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
import matplotlib.pyplot as plt
import numpy as np
data=pd.read_csv('../input/heart-failure-clinical-data/heart_failure_clinical_records_dataset.csv')
data.head()
from sklearn.preprocessing import StandardScaler
data['DEATH_EVENT'].value_counts()
data.corr()
dr=['anaemia','diabetes','creatinine_phosphokinase','smoking','platelets','high_blood_pressure']
train=data.drop(dr,axis=1)
X=train.drop('DEATH_EVENT',axis=1)
Y=train['DEATH_EVENT']
X
s=StandardScaler()
X=s.fit_transform(X)
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y)
from sklearn.ensemble import RandomForestClassifier
rf=RandomForestClassifier()
rf.fit(X_train,Y_train)
rf.score(X_train,Y_train)
rf.score(X_test,Y_test)
from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier()
dt.fit(X_train,Y_train)
dt.score(X_train,Y_train)
dt.score(X_test,Y_test)
from imblearn.over_sampling import SMOTE
oversample=SMOTE()
Xos, yos = oversample.fit_resample(X, Y)
from sklearn.model_selection import train_test_split
Xos_train,Xos_test,Yos_train,Yos_test=train_test_split(Xos,yos)
rf.fit(Xos_train,Yos_train)
rf.score(Xos_train,Yos_train)
rf.score(Xos_test,Yos_test)
y_pr=rf.predict(Xos_test)
from sklearn.metrics import confusion_matrix
confusion_matrix(y_pr,Yos_test)
