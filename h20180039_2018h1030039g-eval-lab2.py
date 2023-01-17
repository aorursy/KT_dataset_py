import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns



%matplotlib inline
df=pd.read_csv("/kaggle/input/eval-lab-2-f464/train.csv")
df.head()
df.isnull().sum()
df.describe()
df['class'].value_counts()
df_1=df[df['class'] == 1]

df_2=df[df['class'] == 2]

df_3=df[df['class'] == 3]

df_4=df[df['class'] == 4]

df_5=df[df['class'] == 5]

df_6 = df[df['class'] == 6]

df_7=df[df['class'] == 7]
df_1 = df_1.sample(49, replace=True)

df_2 = df_2.sample(49, replace=True)

df_3 = df_3.sample(49, replace=True)

df_5 = df_5.sample(49, replace=True)

df_6 = df_6.sample(49, replace=True)

df_7 = df_7.sample(49, replace=True)
df_final = pd.concat([df_1, df_2,df_3,df_5,df_6,df_7], axis=0)
df_final['class'].value_counts()
X_data=df_final.drop(['id','class'],axis=1)
Y_data=df_final['class']
from sklearn.model_selection import train_test_split



X_train,X_val,y_train,y_val = train_test_split(X_data,Y_data,test_size=0.35,random_state=42) 

# X_train=X_data

# y_train=Y_data
from lightgbm import LGBMClassifier

from sklearn.metrics import accuracy_score

lgbm=LGBMClassifier(objective='multiclass',learning_rate=0.3) #72.5 accuracy

lgbm.fit(X_train,y_train)

lgb_pred = lgbm.predict(X_val)

lgb_accuracy = accuracy_score(y_val, lgb_pred)

print(f"The accuracy of the Light GBM is {lgb_accuracy*100:.1f} %")
from sklearn.ensemble import ExtraTreesClassifier

ensemble = ExtraTreesClassifier(n_estimators =200)

ensemble.fit(X_train, y_train)

y_pred_ensemble = ensemble.predict(X_val)

ensemble_accuracy = accuracy_score(y_val, y_pred_ensemble)

print(f"The accuracy of the Extra Trees is {ensemble_accuracy*100:.1f} %")
test=pd.read_csv('/kaggle/input/eval-lab-2-f464/test.csv')
data1=test.drop(['id'],axis=1)

# data1=test[features]
pred = lgbm.predict(data1)

pred
predictions = ensemble.predict(data1)

predictions
compare = pd.DataFrame({'id': test['id'], 'class' : pred})

compare.to_csv('submission13.csv',index=False)