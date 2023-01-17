# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

import plotly.graph_objects as go
diabetes=pd.read_csv('../input/pima-indians-diabetes-database/diabetes.csv')

diabetes.head()
diabetes.info()
diabetes.describe().T


diabetes.hist(figsize=(18,12))
diabetes.columns
diabetes_copy=diabetes.copy(deep=True)

diabetes_copy[['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin',

       'BMI',]]=diabetes_copy[['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin',

       'BMI',]].replace(0,np.nan)
diabetes_copy.isnull().sum()
diabetes_copy.hist(figsize=(18,12))
diabetes_copy['Glucose'].fillna(diabetes_copy['Glucose'].mean(),inplace=True)

diabetes_copy['BloodPressure'].fillna(diabetes_copy['BloodPressure'].median(),inplace=True)

diabetes_copy['SkinThickness'].fillna(diabetes_copy['SkinThickness'].median(),inplace=True)

diabetes_copy['Insulin'].fillna(diabetes_copy['Insulin'].median(),inplace=True)

diabetes_copy['BMI'].fillna(diabetes_copy['BMI'].median(),inplace=True)
diabetes_copy.isnull().sum()
diabetes_copy.hist(figsize=(18,12))
plt.figure(figsize=(18,12))

g=sns.pairplot(diabetes_copy,hue='Outcome',palette="husl")
plt.figure(figsize=(18,12))

sns.heatmap(diabetes_copy.corr(),annot=True,cmap="YlGnBu")
sns.scatterplot(x='SkinThickness',y='BMI',data=diabetes_copy,hue='Outcome')
diabetes_copy[diabetes_copy['Age']==65]['Pregnancies']
sns.countplot(x=diabetes_copy['Outcome'])
group=diabetes_copy['Outcome'].value_counts()

group=group.reset_index()

group
group['Percent']=(group['Outcome']/diabetes_copy.shape[0])*100

group
from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn.neighbors import KNeighborsClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.svm import SVC

from xgboost import XGBClassifier

from sklearn.metrics import confusion_matrix,classification_report,roc_curve

use=['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin','BMI', 'DiabetesPedigreeFunction', 'Age']
sc=StandardScaler()

X=pd.DataFrame(sc.fit_transform(diabetes_copy[use]),columns=[use])

y=diabetes_copy['Outcome']
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.33,random_state=101,stratify=y)
logmodel=LogisticRegression()

logmodel.fit(X_train,y_train)

logpred=logmodel.predict(X_test)

print(confusion_matrix(y_test,logpred))

print(classification_report(y_test,logpred))
test_score=[]

train_score=[]

for i in range(1,20):

    knn=KNeighborsClassifier(n_neighbors=i)

    knn.fit(X_train,y_train)

    train_score.append(knn.score(X_train,y_train))

    test_score.append(knn.score(X_test,y_test))
max_score=max(train_score)

train_score_index=[i for i,v in enumerate(train_score) if v==max_score]

print('Max train score {}% and k ={}'.format(max_score*100,list(map(lambda x:x+1,train_score_index))))
max_score_test=max(test_score)

test_score_index=[i for i,v in enumerate(test_score) if v==max_score_test]

print('Max train score {}% and k ={}'.format(max_score_test*100,list(map(lambda x:x+1,test_score_index))))
knn=KNeighborsClassifier(n_neighbors=11)

knn.fit(X_train,y_train)

knnpred=knn.predict(X_test)

print(confusion_matrix(y_test,knnpred))

print(classification_report(y_test,knnpred))
forest=RandomForestClassifier()

forest.fit(X_train,y_train)

forestpred=forest.predict(X_test)

print(confusion_matrix(y_test,forestpred))

print(classification_report(y_test,forestpred))
svc=SVC()

svc.fit(X_train,y_train)

svcpred=svc.predict(X_test)

print(confusion_matrix(y_test,svcpred))

print(classification_report(y_test,svcpred))
from sklearn.model_selection import GridSearchCV



param_grid={'n_neighbors':np.arange(1,70)}

knn_tuned=KNeighborsClassifier()

knn_cv=GridSearchCV(knn_tuned,param_grid,cv=5)

knn_cv.fit(X,y)
knn_cv.best_score_
knn_cv.best_params_
from sklearn.ensemble import GradientBoostingClassifier
gb=GradientBoostingClassifier()

gb.fit(X_train,y_train)

gb_pred=gb.predict(X_test)

print(confusion_matrix(y_test,gb_pred))

print(classification_report(y_test,gb_pred))
from xgboost import XGBClassifier

xgb=GradientBoostingClassifier()

xgb.fit(X_train,y_train)

xgb_pred=xgb.predict(X_test)

print(confusion_matrix(y_test,xgb_pred))

print(classification_report(y_test,xgb_pred))
from keras.models import Sequential

from keras.layers import Dense,Flatten,LeakyReLU
model=Sequential()

model.add(Dense(100,input_dim=8,activation="relu",init="he_uniform"))

model.add(Dense(50,activation="relu",init="he_uniform"))

model.add(Dense(1,activation="sigmoid",init="glorot_uniform"))

model.compile(loss="binary_crossentropy",optimizer="adamax",metrics=['accuracy'])

model.fit(X_train,y_train,validation_split=0.20,batch_size=20,epochs=100)
model.evaluate(X_test,y_test)[1]