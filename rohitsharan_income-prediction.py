import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

from sklearn.preprocessing import OrdinalEncoder

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix

from imblearn.over_sampling import SMOTE

from imblearn.under_sampling import NearMiss

np.random.seed(0)
file=pd.read_csv("../input/adult-income-dataset/adult.csv")

file

file.isnull().sum()
data=file

copy=file

encoder=OrdinalEncoder()

encoded_values=encoder.fit_transform(data)

data=pd.DataFrame(data=encoded_values, columns=copy.columns)

data
plt.figure(figsize=(20,20))

sns.boxplot(data=pd.DataFrame(data))

plt.show()
Y=data["income"]

data=data.loc[:,data.columns!="income"]

print(Y.value_counts())

data
model=RandomForestClassifier()

X_train,X_test,Y_train,Y_test=train_test_split(data,Y,test_size=0.25,random_state=0)

model.fit(X_train,Y_train)

y_pred=model.predict(X_test)

print(model.score(X_test,Y_test)*100,"%")
conf=confusion_matrix(y_true=Y_test,y_pred=y_pred)

print(conf)
model=RandomForestClassifier()

smote=SMOTE(random_state=0)

nr=NearMiss()

X_train_new,Y_train_new=smote.fit_sample(X_train,Y_train)

model.fit(X_train_new,Y_train_new)

y_pred=model.predict(X_test)

conf=confusion_matrix(y_true=Y_test,y_pred=y_pred)  

print(model.score(X_test,Y_test)*100,"%")

print(conf)
print(Y_train.value_counts())

tar=pd.DataFrame(Y_train_new)

tar[0].value_counts()
model=RandomForestClassifier()

nr=NearMiss()

X_train_new,Y_train_new=nr.fit_sample(X_train,Y_train)

model.fit(X_train_new,Y_train_new)

y_pred=model.predict(X_test)

conf=confusion_matrix(y_true=Y_test,y_pred=y_pred)  

print(model.score(X_test,Y_test)*100,"%")

print(conf)
print(Y_train.value_counts())

tar=pd.DataFrame(Y_train_new)

tar[0].value_counts()