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
train=pd.read_csv("/kaggle/input/titanic/train.csv")
test=pd.read_csv("/kaggle/input/titanic/test.csv")
gender_sub=pd.read_csv("/kaggle/input/titanic/gender_submission.csv")
print(train.shape)
test.shape
train.head()
test.isnull().sum()
train.isnull().sum()
gender_sub.info()
train.head()
Train=train.drop("Survived",inplace=False,axis=1)
data=pd.concat([Train,test])
print(data.shape)
data.isnull().sum()
#dropping the unneccessary columns and checking for nulls
data.drop(columns=["Name","Ticket","Cabin"],axis=1,inplace=True)
print(data.isnull().sum())

data.Embarked.mode()        
data.Embarked.fillna(value="S",axis=0,inplace=True)
data.Embarked.isnull().sum()
import matplotlib.pyplot as plt
import seaborn as sns
import warnings 
warnings.filterwarnings("ignore")
%matplotlib inline
data.Age.skew()

avg_age=data.Age.mean()
data.Age.fillna(value=avg_age,axis=0,inplace=True)
data.isnull().sum()
# print(data.Fare.skew())
# avg_Fare=data.Fare.mean()
# data.Fare.fillna(value=avg_Fare,axis=0,inplace=True)
# print(data.Fare.skew())
data.drop(columns=["PassengerId","Fare"],axis=1,inplace=True)
print(data.Pclass.unique())
print(data.Embarked.unique())
print(data.Parch.unique())
print(data.SibSp.unique())
Dummy_data=pd.get_dummies(data, columns=["Pclass","Embarked","Sex"])
Dummy_data.head()
##normalizing the data
from sklearn.preprocessing import MinMaxScaler
scale=MinMaxScaler()
col=Dummy_data.columns
Transformed_data=scale.fit_transform(Dummy_data)
Transformed_df=pd.DataFrame(Transformed_data,columns=col)
Transformed_df.head()
#splitting into train and test data
transformed_train=Transformed_df.iloc[:891,:]
transformed_test=Transformed_df.iloc[891:,:]
#splitting into training and validation sets
y=train.Survived
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(transformed_train,y,train_size=0.7,random_state=1)


y.value_counts()
#using Logistic_regression
from sklearn.linear_model import LogisticRegression
model=LogisticRegression()
model.fit(x_train,y_train)
y_pred=model.predict(x_test)

from sklearn.metrics import classification_report,accuracy_score,roc_auc_score
print(roc_auc_score(y_test,y_pred))
print(accuracy_score(y_test,y_pred))
print(classification_report(y_test,y_pred))
#balancing the classes of survived feature
from sklearn.utils import resample
transformed_train["Survived"]=y
min_data=transformed_train[transformed_train["Survived"]==1]
maj_data=transformed_train[transformed_train["Survived"]==0]
mod_min_data=resample(min_data,random_state=1,n_samples= 549)
balanced_train=pd.concat([maj_data,mod_min_data])

Y=balanced_train.Survived
x=balanced_train.drop("Survived",axis=1,inplace=False)
X_train,X_test,Y_train,Y_test=train_test_split(x,Y,train_size=0.7,random_state=1)
from sklearn.linear_model import LogisticRegression
model=LogisticRegression()
model.fit(X_train,Y_train)
Y_pred=model.predict(X_test)
print(roc_auc_score(Y_test,Y_pred))
print(accuracy_score(Y_test,Y_pred))
print(classification_report(Y_test,Y_pred))
#using randomforest_classifier
from sklearn.ensemble import RandomForestClassifier
model1=RandomForestClassifier(n_estimators=500,max_depth=9,max_features="auto",random_state=1)
model1.fit(X_train,Y_train)
Yr_pred=model1.predict(X_test)
print(roc_auc_score(Y_test,Yr_pred))
print(accuracy_score(Y_test,Yr_pred))
print(classification_report(Y_test,Yr_pred))

result=pd.DataFrame(test.PassengerId,columns=["PassengerId"])
result["Survived"]=model1.predict(transformed_test)
result.to_csv("subR.csv",index=False)

X_train.shape
import tensorflow as tf
from tensorflow.keras.layers import Dense,Dropout,Activation
from tensorflow.keras.models import Sequential




model=Sequential()
model.add(Dense(500,input_shape=(11,)))
model.add(Activation("relu"))
model.add(Dense(400))
model.add(Activation("relu"))
model.add(Dense(100))
model.add(Activation("relu"))
model.add(Dense(10))
model.add(Activation("relu"))
model.add(Dense(1))
model.add(Activation("sigmoid"))

model.compile(loss="binary_crossentropy",metrics=["accuracy"],optimizer="adam")
hist=model.fit(X_train,Y_train,batch_size=150,epochs=100,verbose=2,validation_data=(X_test,Y_test))
Yn_pred=model.predict_classes(X_test)
print(roc_auc_score(Y_test,Yn_pred))
print(accuracy_score(Y_test,Yn_pred))
n_result=pd.DataFrame(test.PassengerId,columns=["PassengerId"])
n_result["Survived"]=model.predict_classes(transformed_test)
n_result.to_csv("subN.csv",index=False)
