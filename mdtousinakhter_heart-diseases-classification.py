# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
heart_d=pd.read_csv("../input/heart.csv")

heart_d.head()
heart_d.describe()
heart_d.info()
heart_d.isnull()
import seaborn as sns

g=sns.pairplot(heart_d)
import matplotlib.pyplot as plt

%matplotlib inline

plt.subplots(figsize=(10,8))

sns.heatmap(heart_d.corr(),annot=True,linewidths=0.8,cmap='coolwarm')
plt.scatter(heart_d['chol'],heart_d['thalach'])
sns.jointplot(x='chol',y='thalach',data=heart_d,kind='kde',color="g")
ax=plt.subplots(figsize=(10,8))

sns.boxplot(data=heart_d['trestbps'])
sns.barplot(heart_d['target'],heart_d['trestbps'])
plt.subplots(figsize=(10,8))

sns.boxplot(data=heart_d['chol'])
sns.barplot(heart_d['target'],heart_d['chol'])
sns.boxplot(data=heart_d['oldpeak'])
sns.barplot(heart_d['target'],heart_d['oldpeak'])
from sklearn.model_selection import train_test_split

X=heart_d.drop("target",axis=1)

Y=heart_d["target"]

X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.20,random_state=42)

from sklearn.metrics import accuracy_score
from sklearn import svm

sv=svm.SVC(kernel='linear')

sv.fit(X_train,Y_train)

pred_sv=sv.predict(X_test)

pred_sv.shape
score_svm=accuracy_score(pred_sv,Y_test)

print("The accuracy achieved using svm is"+" "+str((score_svm)*100))
from sklearn.linear_model import LogisticRegression

lg=LogisticRegression()

lg.fit(X_train,Y_train)

pred_lg=lg.predict(X_test)
score_lgr=accuracy_score(pred_lg,Y_test)

print("The accuracy achieved using Logistic regression"+" "+str((score_lgr)*100))

from keras.models import Sequential

from keras.layers import Dense

model=Sequential()

model.add(Dense(11,input_dim=13,activation='relu'))

model.add(Dense(1,activation='sigmoid'))

model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
model.fit(X_train,Y_train,epochs=250)
pred_nn=model.predict(X_test)
round_nn=[round(x[0]) for x in pred_nn]

score_nn=accuracy_score(round_nn,Y_test)*100

print("The accuracy score of the neural network is"+" "+str(score_nn))

score=[score_svm*100,score_lgr*100,score_nn]

algorithms=["Support Vector Machine","Logistic Regression", "Neural Network"]

sns.barplot(algorithms,score)