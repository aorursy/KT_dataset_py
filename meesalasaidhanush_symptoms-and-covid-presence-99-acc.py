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
df=pd.read_csv('/kaggle/input/symptoms-and-covid-presence/Covid Dataset.csv')
df.head()
df.isnull().any()
df.describe()
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
sns.countplot(df['COVID-19'])
from sklearn.preprocessing import LabelEncoder
l=LabelEncoder()
df=df.apply(l.fit_transform).astype(int)
cor=df.corr()
cor
df.dtypes
sns.heatmap(cor)
x=df.iloc[:,[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19]]
y=df.iloc[:,[20]]
x.head()
y.head()
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=4,test_size=0.2)
print(x_train.shape)
print(x_test.shape)
from sklearn.linear_model import LogisticRegression
lr= LogisticRegression()
lr.fit(x_train,y_train)
from sklearn.metrics import accuracy_score,confusion_matrix
pr=lr.predict(x_test)
print(confusion_matrix(y_test,pr))
print('accuracy is',accuracy_score(y_test,pr))
import xgboost
xgb=xgboost.XGBClassifier()
xgb.fit(x_train,y_train)
pred=xgb.predict(x_test)
confusion=confusion_matrix(y_test,pred)
print('accuracy is',accuracy_score(y_test,pred))
print(confusion)
TN = confusion [0,0]
FP = confusion [0,1]
FN = confusion [1,0]
TP = confusion [1,1]
print((FP+FN)/float(TP+TN+FP+FN))
print(round(1-accuracy_score(y_test, pred),4))
from sklearn.metrics import recall_score
print("RECALL:", recall_score(y_test,pred))
print("CALCULATED RECALL:", (TP)/(TP+FN))

from sklearn.metrics import accuracy_score,f1_score,precision_score,recall_score,roc_auc_score
accuracy = accuracy_score(y_test, pred)
recall = recall_score(y_test, pred)
precision = precision_score(y_test, pred)
f1 = f1_score(y_test, pred)
roc_auc = roc_auc_score(y_test, pred)

print('Accuracy is  :' ,round(accuracy,2)*100)
print('F1 score is :' ,round(f1,2)*100)
print('Precision is  :',round(precision,2)*100)
print('Recall is  :',round(recall,4)*100)
print('Roc Auc is  :',round(roc_auc,2)*100)
