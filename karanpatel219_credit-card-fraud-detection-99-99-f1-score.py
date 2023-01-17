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
data=pd.read_csv('../input/creditcardfraud/creditcard.csv')
data
data['Class'].value_counts()
data.drop('Time',axis=1,inplace=True)
data
from imblearn.over_sampling import SMOTE
Y=data['Class']
X=data.drop('Class',axis=1)
X
us=SMOTE()
Xs,Ys=us.fit_resample(X,Y)
Ys.value_counts()
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
Xsc=sc.fit_transform(Xs)
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(Xs,Ys)
from sklearn.ensemble import RandomForestClassifier
model=RandomForestClassifier()
model.fit(X_train,Y_train.values)
model.score(X_train,Y_train)
model.score(X_test,Y_test)
ypre=model.predict(X_test)
ypre['Predicted']=pd.DataFrame(ypre)
from sklearn.metrics import f1_score
f1_score(Y_test,ypre)
