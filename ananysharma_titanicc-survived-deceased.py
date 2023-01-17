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
train_data = pd.read_csv('/kaggle/input/titanic/train.csv')
train_data['Age'] = train_data['Age'].fillna(train_data['Age'].mean())
train_data.head()
test_data = pd.read_csv('/kaggle/input/titanic/test.csv')
test_data['Age'].fillna(test_data["Age"].median(),inplace = True)
test_data

women = train_data.loc[train_data.Sex == 'female']['Survived']
print(sum(women)/len(women))
train_data.drop(labels = ["Name","Ticket","Cabin"],axis=1,inplace = True)
test_data.drop(labels = ["Name","Ticket","Cabin"],axis=1,inplace = True)
men = train_data.loc[train_data.Sex == 'male']['Survived']
print(sum(men)/len(men))


import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams['figure.figsize'] = (15,8)
sns.countplot(train_data["Survived"],palette = 'dark')
plt.xlabel("Survived or not")
plt.show()
sns.boxplot(x = train_data["Survived"],y = train_data["Fare"],hue = train_data["Survived"],palette = 'dark')
sns.boxplot(x = train_data["Survived"],y = train_data["Age"],hue = train_data["Survived"],palette = 'dark')
train_data
d1 = {'male':0 ,'female':1}
train_data["Sex"] = train_data["Sex"].map(d1)
test_data["Sex"] = test_data["Sex"].map(d1)
map2 = {"S":0 , "C":1 , "Q":2}
train_data["Embarked"] = train_data["Embarked"].map(map2)
sns.heatmap(train_data[:].corr(),cmap = "RdYlGn",annot = True)
sns.scatterplot(train_data["Fare"],train_data["Age"],color='Green')
sns.catplot(x="Survived", y="Age", hue="Sex", kind="swarm", data=train_data, aspect=1,height=8);
sns.factorplot(x="Sex",col="Survived", data=train_data , kind="count",size=7, aspect=.7,palette=['red','green'])
sns.catplot(x="Survived",hue="Pclass", kind="count",col='Sex', data =train_data,color='Violet',aspect=0.7,height=7);
sns.catplot(x="Survived", hue="SibSp", col = 'Sex',kind="count", data=train_data,height=7);
sns.catplot(x="Survived", hue="Parch", col = 'Sex', kind="count", data=train_data,height=7);
emb =train_data.groupby('Embarked').size()

plt.pie(emb.values,labels = ["Cherbourg","Queenstown","Southampton"],startangle=90,autopct='%1.1f%%',shadow = True);
sns.catplot(x="Embarked",hue="Survived", kind="count",col='Sex', data=train_data,aspect=0.7,height=7);
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.metrics import mean_squared_error as mse

y = train_data['Survived']
feat = ['Pclass','Sex','SibSp','Parch']
X  =pd.get_dummies(train_data[feat])
X_test = pd.get_dummies(test_data[feat])
# models=[KNeighborsClassifier(n_neighbors=5)]

# predictions = model.predict(X_test)
# output = pd.DataFrame({'PassengerId':test_data.PassengerId,'Survived':predictions})
# output.to_csv('my_submissions.csv',index = False)
# print("Ho gaya")
# print(train_data.describe())
print(X.shape)
print(y.shape)
Xtrain,Xtest,ytrain,ytest = train_test_split(X,y,random_state=30,stratify = y)
Xtrain.shape
ytrain.shape
def elbow(k):
    test_error = []
   
    for i in k:
        model = KNeighborsClassifier(n_neighbors = i)
        model.fit(Xtrain,ytrain)
        tmp = model.predict(Xtest)
        tmp = f1_score(tmp,ytest)
        error = 1-tmp
        test_error.append(error)
    return test_error    
k = range(6,20,2)
test_e = elbow(k)
plt.plot(k,test_e)
plt.xlabel("K value")
plt.ylabel("Test Error")
plt.show()
def elbowm(k):
    test_error = []
   
    for i in k:
        model = KNeighborsClassifier(n_neighbors = i)
        model.fit(Xtrain,ytrain)
        tmp = model.predict(Xtest)
        tmp = mse(tmp,ytest)
        error = tmp
        test_error.append(error)
    return test_error
k = range(6,20,2)
test_e = elbowm(k)
plt.plot(k,test_e)
plt.xlabel("K value")
plt.ylabel("Test Error")
plt.show()
from sklearn.model_selection import cross_val_score
model = KNeighborsClassifier(n_neighbors=16)
model.fit(Xtrain,ytrain)
preds = model.predict(Xtest)
print(f1_score(preds,ytest))
#     print(model)
#     print('Accuracy of classifier on training set:{}%'.format(round(f1_score(X_train, y_train)*100)))
#     print("Training data:"+round(clf.cross_val_score(X_train,y_train)*100))
predictions =  model.predict(X_test)
predictions

sample = pd.read_csv('/kaggle/input/titanic/gender_submission.csv')
sample
sub = pd.DataFrame({'PassengerId':sample.PassengerId,
                   'Survived':predictions})
sub
sub.to_csv('Submiss.csv',index=False,header=True)
from xgboost import XGBClassifier
import time

xgb = XGBClassifier(n_estimators=100)
training_start = time.perf_counter()
xgb.fit(Xtrain, ytrain)
training_end = time.perf_counter()
prediction_start = time.perf_counter()
preds = xgb.predict(Xtest)
prediction_end = time.perf_counter()
acc_xgb = (preds == ytest).sum().astype(float) / len(preds)*100
xgb_train_time = training_end-training_start
xgb_prediction_time = prediction_end-prediction_start
print("XGBoost's prediction accuracy is: %3.2f" % (acc_xgb))
print("Time consumed for training: %4.3f" % (xgb_train_time))
print("Time consumed for prediction: %6.5f seconds" % (xgb_prediction_time))
params = {
        'min_child_weight': [1, 5, 10],
        'gamma': [0.5, 1, 1.5, 2, 5],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0],
        'max_depth': [3, 4, 5]
        }
xgb = XGBClassifier(learning_rate=0.02, n_estimators=600, objective='binary:logistic',
                    silent=True, nthread=1)
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from datetime import datetime

folds = 5
param_comb = 5

skf = StratifiedKFold(n_splits=folds, shuffle = True, random_state = 1001)

random_search = RandomizedSearchCV(xgb, param_distributions=params, n_iter=param_comb, scoring='roc_auc', n_jobs=4, cv=skf.split(Xtrain,ytrain), verbose=3, random_state=1001 )

# Here we go
# start_time = timer(None) # timing starts from this point for "start_time" variable
random_search.fit(Xtrain, ytrain)
# timer(start_time)
random_search.best_params_
predss = random_search.predict(X_test)
random_search.best_score_
sample = pd.read_csv('/kaggle/input/titanic/gender_submission.csv')
sample
sub = pd.DataFrame({'PassengerId':sample.PassengerId,
                   'Survived':predss})
sub
sub.to_csv('Submiss1.csv',index=False,header=True)
_