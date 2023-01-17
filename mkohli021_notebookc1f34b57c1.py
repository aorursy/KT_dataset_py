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
df=pd.read_csv("../input/chances-of-heart-attack/heart.csv")
#prints first 5 rows
df.head()
#prints last 5 rows
df.tail()
df.shape
df.describe()
#describe the stats of the data for numerical variables
print(np.var(df,axis=0))

Var.to_csv("HeartVar.csv")

cor=df.corr()
cor
df.drop(['slope','thal',
'restecg',
'exang',
'sex',
'fbs'], axis=1, inplace=True)
df.head()
Q1=df.quantile(0.25)
Q3=df.quantile(0.75)
iqr=Q3-Q1
iqr
df['target'].value_counts()
df2=df[:]
df2.drop(['target'], axis=1, inplace=True)
X=df2
X.head()
y=df['target']
y.head()

from sklearn import model_selection
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
logReg=LogisticRegression()
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.30,random_state=10)
logReg=logReg.fit(X_train,y_train)
prob=logReg.predict_proba(X_train)
prob
prob=pd.DataFrame(prob)
prob1=prob.iloc[:,1]
prob=pd.DataFrame(prob1)
prob.columns=['p_d']
prob
y_train.value_counts()
y_pred=prob['p_d'].apply(lambda x: 1 if x>0.43 else 0)
y_pred.value_counts()
from sklearn import metrics
cnf_metrix=metrics.confusion_matrix(y_train, y_pred)
cnf_metrix
fpr,tpr,_=metrics.roc_curve(y_train, y_pred)
auc=metrics.roc_auc_score(y_train, y_pred)
import matplotlib.pyplot as plt
plt.plot(fpr, tpr, label='data 1, auc='+str(auc))
plt.legend(loc=4)
plt.show()
