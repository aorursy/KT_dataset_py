import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from keras.models import Sequential

from keras.layers.core import Dense,Activation,Dropout

from keras.layers import LSTM

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split

import seaborn as sns

from sklearn.ensemble import RandomForestClassifier

from sklearn import svm

from xgboost import XGBClassifier

from sklearn.neural_network import MLPClassifier

from sklearn.metrics import accuracy_score

from sklearn.tree import DecisionTreeClassifier





from keras.callbacks import EarlyStopping

import math

df=pd.read_csv('../input/Gastric_Ulcer_Train.csv',delimiter=',')

df.head(3)

df=pd.read_csv('../input/Gastric_Ulcer_Train.csv',delimiter=',')

df.head(3)

del df['ID']

del df['Age']

df=pd.concat([df,pd.get_dummies(df['Personality Type'], drop_first=True, prefix=1)], axis=1,sort=False)

df=pd.concat([df,pd.get_dummies(df['Gender'], drop_first=True, prefix=2)], axis=1,sort=False)

df=pd.concat([df,pd.get_dummies(df['Fixed Mealtimes'], drop_first=True, prefix=3)], axis=1,sort=False)

df=pd.concat([df,pd.get_dummies(df['Alcohol Consumption'], drop_first=True, prefix=4)], axis=1,sort=False)

df=pd.concat([df,pd.get_dummies(df['Vegetarian'], drop_first=True, prefix=5)], axis=1,sort=False)

df=pd.concat([df,pd.get_dummies(df['Smoker'], drop_first=True, prefix=6)], axis=1,sort=False)

df=pd.concat([df,pd.get_dummies(df['CLO Test'], drop_first=True, prefix=7)], axis=1,sort=False)

df=pd.concat([df,pd.get_dummies(df['Coronary Disease'], drop_first=True, prefix=8)], axis=1,sort=False)

df=df.drop(['Personality Type','Gender','Fixed Mealtimes','Alcohol Consumption','Vegetarian','Smoker','CLO Test','Coronary Disease'],axis=1)

df.head()

df1=df

target=df['Gastric Ulcer']

sns.countplot(target)

del df['Gastric Ulcer']

sc=StandardScaler()

sc.fit_transform(df)

X_train,X_test,Y_train,Y_test=train_test_split(df,target,test_size=0.3)



model = XGBClassifier(objective="binary:logistic",n_estimators=500,max_depth=8,learning_rate=0.07)

model.fit(X_train, Y_train)

predict3=model.predict(X_test)

c=0

for i in range(len(predict3)):

    if(predict3[i]==Y_test.iloc[i]):

        c+=1

c3=(c/len(predict3))*100

print('XGBoost Accuracy Score is',c3)

    
model= XGBClassifier(objective="binary:logistic",n_estimators=500,max_depth=8,learning_rate=0.07)

model.fit(df,target)
df=pd.read_csv('../input/Gastric_Ulcer_Test.csv',delimiter=',')

del df['ID']

del df['Age']

df=pd.concat([df,pd.get_dummies(df['Personality Type'], drop_first=True, prefix=1)], axis=1,sort=False)

df=pd.concat([df,pd.get_dummies(df['Gender'], drop_first=True, prefix=2)], axis=1,sort=False)

df=pd.concat([df,pd.get_dummies(df['Fixed Mealtimes'], drop_first=True, prefix=3)], axis=1,sort=False)

df=pd.concat([df,pd.get_dummies(df['Alcohol Consumption'], drop_first=True, prefix=4)], axis=1,sort=False)

df=pd.concat([df,pd.get_dummies(df['Vegetarian'], drop_first=True, prefix=5)], axis=1,sort=False)

df=pd.concat([df,pd.get_dummies(df['Smoker'], drop_first=True, prefix=6)], axis=1,sort=False)

df=pd.concat([df,pd.get_dummies(df['CLO Test'], drop_first=True, prefix=7)], axis=1,sort=False)

df=pd.concat([df,pd.get_dummies(df['Coronary Disease'], drop_first=True, prefix=8)], axis=1,sort=False)

df=df.drop(['Personality Type','Gender','Fixed Mealtimes','Alcohol Consumption','Vegetarian','Smoker','CLO Test','Coronary Disease'],axis=1)

df.head()

sc=StandardScaler()

sc.fit_transform(df)



predict1=model.predict(df)

iid=np.arange(0,200,1)+801

ind=np.arange(1,201,1)

g=pd.DataFrame({'Id':iid,'Gastric Ulcer':predict1},index=ind)

g.head()

g.to_csv("output.csv",index=False)
