

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

import pandas as pd

import numpy as np  

import matplotlib.pyplot as plt

import seaborn as sns

df=pd.read_csv('/kaggle/input/crop-damage-information-in-india/Crop_Agriculture_Data_2.csv')
df.head(5)

df.info()
df=df.drop(columns=['ID'],axis=1)
df.select_dtypes(include='object')

from sklearn.preprocessing import OrdinalEncoder

oe=OrdinalEncoder()

df=pd.DataFrame(oe.fit_transform(df),columns=list(df))
x=df.drop(columns=['Crop_Damage'],axis=1)

y=df['Crop_Damage']
x.head(20)

print(y.value_counts())

#The data seems quite skewed

y.value_counts()

ratio_target=pd.DataFrame([y.value_counts(),y.value_counts()/y.shape[0]*100])

print(ratio_target)

#skewed data


ratio_target.plot(kind='hist')
from imblearn.over_sampling import SMOTE

smt=SMOTE()

x,y=smt.fit_resample(x,y)
from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3)

import lightgbm

lgbm=lightgbm.LGBMClassifier()

lgbm.fit(x_train,y_train)

y_pred=lgbm.predict(x_test)

from sklearn.metrics import accuracy_score,f1_score

print('Accuracy score :',accuracy_score(y_test,y_pred))

print('f1 score :',f1_score(y_test,y_pred,average='weighted'))

import xgboost 

xgb=xgboost.XGBClassifier()

xgb.fit(x_train,y_train)

y_pred=xgb.predict(x_test)


print('Accuracy score :',accuracy_score(y_test,y_pred))

print('f1 score :',f1_score(y_test,y_pred,average='weighted'))