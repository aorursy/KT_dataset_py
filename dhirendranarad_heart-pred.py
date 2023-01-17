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
df=pd.read_csv('/kaggle/input/heart-failure-clinical-data/heart_failure_clinical_records_dataset.csv')

df









df.dtypes
df.columns
df.isnull().sum()
(df.groupby('anaemia')['high_blood_pressure'].count()).plot.bar()
df['high_blood_pressure'].value_counts().plot.bar()
x=df.drop(['DEATH_EVENT'],axis=1)

y=df['DEATH_EVENT']
x.shape,y.shape
from sklearn.model_selection import train_test_split

train_x,test_x,train_y,test_y=train_test_split(x,y,test_size=0.3,random_state=56)
from sklearn.preprocessing import MinMaxScaler

scaler=MinMaxScaler()
train_xscaled=scaler.fit_transform(train_x)

train_xscaled=pd.DataFrame(train_xscaled,columns=train_x.columns)
train_xscaled.head()
test_xscaled=scaler.fit_transform(test_x)

test_xscaled=pd.DataFrame(test_xscaled,columns=test_x.columns)
test_xscaled.head()
from sklearn.linear_model import LogisticRegression as LogReg

from sklearn.metrics import f1_score

from sklearn.metrics import confusion_matrix
logreg=LogReg()

logreg.fit(train_xscaled,train_y)
train_predict=logreg.predict(train_xscaled)

print('f1_score ',f1_score(train_predict,train_y))
test_predict=logreg.predict(test_xscaled)

print('f1_score ',f1_score(test_predict,test_y))
train_predict=logreg.predict_proba(train_xscaled)

train_predict
train_preds=train_predict[:,1]

train_preds
train_preds.mean()
for i in range(0,len(train_preds)):

    if train_preds[i]>0.33:

        train_preds[i]=1

    else:

        train_preds[i]=0
train_preds
print('f1_score ',f1_score(train_preds,train_y))
print('confusion_matrix :',confusion_matrix(test_y,test_predict))
from sklearn.metrics import classification_report as rep

print(rep(test_y,test_predict))