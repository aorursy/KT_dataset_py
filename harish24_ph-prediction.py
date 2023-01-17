# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



        

# Any results you write to the current directory are saved as output.
phs = pd.read_csv('/kaggle/input/ph-recognition/ph-data.csv')

ph = phs.copy()
ph.describe()
ph.info()
sns.boxplot('label','blue',data=ph)
sns.catplot('label','blue',data=ph)
sns.barplot('label','blue',data=ph)
sns.boxplot('label','red',data=ph)
sns.barplot('label','red',data=ph)
corr = ph.corr()
print(corr)
#sns.pairplot(ph,palette='set2')
ph['label']=ph['label'].astype('object')
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split



x = ph[['blue','green','red']]

y = ph['label']
y = y.astype('int64')
X_train,X_test,y_train,y_test = train_test_split(x,y,test_size=0.3,random_state=42)
from sklearn.metrics import accuracy_score,precision_score,recall_score

from sklearn.model_selection import cross_val_predict



rfm = RandomForestClassifier(max_features=0.2,criterion='entropy',n_estimators=2000,warm_start=True,bootstrap=True)

rfm.fit(X_train,y_train)



rfm_pred = cross_val_predict(rfm,X_train,y_train,cv=10)

rfm_acc = accuracy_score(y_train,rfm_pred)



print(rfm_acc)

y_pred = rfm.predict(X_test)

accuracy = accuracy_score(y_test,y_pred)

print(accuracy)