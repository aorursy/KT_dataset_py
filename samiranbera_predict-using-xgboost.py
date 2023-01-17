import pandas as pd

import numpy as np

import math

import datetime

import plotly.express as px

from plotly.subplots import make_subplots

import plotly.graph_objects as go

import plotly.graph_objs as go

from plotly.subplots import make_subplots

from sklearn.metrics import r2_score

import warnings

warnings.warn = False
import pandas as pd

gender_submission = pd.read_csv("../input/titanic/gender_submission.csv")

test = pd.read_csv("../input/titanic/test.csv")

train = pd.read_csv("../input/titanic/train.csv")



print("Training set = ",train.shape)

print("Testing set = ",test.shape)

print("Sum of Missing Values (Train/Test)= ",train.isna().sum().sum(),"(",test.isna().sum().sum(),")")
ls=[[i,train[i].isna().sum(),round(train[i].isna().sum()/train.shape[0]*100,2)] for i in train.columns if train[i].isna().sum()>0]

print('Before Imputing Missing Values\n',ls)





train['Age'].fillna(train['Age'].mean(),inplace=True) # Impute Missing Value of Age with mean



train.loc[train.Cabin.isna(),'Cabin'] = 'NAA'         # Impute Missing Value of Cabin with NAA

train['Cabin'] = [i[0] for i in train.Cabin]



#train.Embarked.hist()

train.loc[train.Embarked.isna(),'Embarked'] = 'S'     # Impute Missing Value of Embarked with Most Frequent Value 





ls=[[i,train[i].isna().sum(),round(train[i].isna().sum()/train.shape[0]*100,2)] for i in train.columns if train[i].isna().sum()>0]

print('After Imputing Missing Values\n',ls)
ls=[[i,test[i].isna().sum(),round(test[i].isna().sum()/test.shape[0]*100,2)] for i in test.columns if test[i].isna().sum()>0]

print('Before Imputing Missing Values\n',ls)



test['Age'].fillna(test['Age'].mean(),inplace=True) # Impute Missing Value of Age with mean



test.loc[test.Cabin.isna(),'Cabin'] = 'NAA'         # Impute Missing Value of Cabin with NAA

test['Cabin'] = [i[0] for i in test.Cabin]



test.loc[test.Fare.isna(),'Fare'] = test.Fare[test.Pclass==3].mean() # Impute Missing Value of Pclass = 3



ls=[[i,train[i].isna().sum(),round(train[i].isna().sum()/train.shape[0]*100,2)] for i in train.columns if train[i].isna().sum()>0]

print('After Imputing Missing Values\n',ls)
#train = train[train.Fare<400]
#test = test[test.Fare<400]
diff = max(train.Age) - min(train.Age)

train['Age_Group'] = [math.floor(i) if math.floor(i)>1 else 1 for i in (train.Age-1)/diff+1]

train.drop(columns=['Age'],axis=1,inplace=True)
diff = max(test.Age) - min(test.Age)

test['Age_Group'] = [math.floor(i) if math.floor(i)>1 else 1 for i in (test.Age-1)/diff+1]

test.drop(columns=['Age'],axis=1,inplace=True)
train.Fare = train.Fare/400

train.loc[train.Fare>1,'Fare'] = 1
test.Fare  = test.Fare/400

test.loc[test.Fare>1,'Fare'] = 1
ohe_col = ['Pclass', 'Sex', 'SibSp', 'Parch', 'Cabin', 'Embarked','Age_Group']

train = pd.get_dummies(train,columns=ohe_col,drop_first=True)

test  = pd.get_dummies(test ,columns=ohe_col,drop_first=True)
y_train = train.Survived

X_train = train.drop(columns=['Survived','PassengerId','Name','Ticket'],axis=1)

X_train.head()
X_test_passenger_id = test.PassengerId

passenger_name = test.Name 

X_test  = test.drop(columns=['PassengerId','Name','Ticket'],axis=1)

X_test.head()
A = set(X_train.columns)

B = set(X_test.columns)

C = A-B

D = B-A

keep_columns = list(A - C - D)



X_train = X_train[keep_columns]

X_test  = X_test[keep_columns]
from sklearn.linear_model import LogisticRegression

log_reg = LogisticRegression(solver='liblinear',penalty='l1',max_iter=100,random_state=7)

log_reg.fit(X_train,y_train)
from sklearn.metrics import accuracy_score

y_pred = log_reg.predict(X_train)

print('The accuracy score is ',round(accuracy_score(y_train,y_pred)*100,2))
def get_prediction(y_train,y_pred_pa,threshold_list):

  best_threshold = 0

  best_score = 0

  for i in threshold_list:

    y_pred = [1 if j>=i else 0 for j in y_pred_pa[:,1]]

    temp = accuracy_score(y_train,y_pred)    

    if best_score < temp:

      best_score = temp

      best_threshold = i



  y_pred = [1 if j>=best_threshold else 0 for j in y_pred_pa[:,1]]



  return best_threshold,y_pred
y_pred_pa  = log_reg.predict_proba(X_train)



best_threshold, y_pred = get_prediction(y_train,y_pred_pa,[0.1,0.3,0.5,0.7,0.9])

print('Accuracy Improvement ( Threshold=',best_threshold,'):',round(accuracy_score(y_train,y_pred)*100,2))



best_threshold, y_pred = get_prediction(y_train,y_pred_pa,[0.45,0.5,0.55,0.6,0.65])

print('Accuracy Improvement ( Threshold=',best_threshold,'):',round(accuracy_score(y_train,y_pred)*100,2))





# Thus, optimal threshold 0.6 is chosen

y_pred = [1 if j>=0.6 else 0 for j in y_pred_pa[:,1]]
from sklearn.model_selection import cross_val_score

score_list = cross_val_score(log_reg,X_train,y_train,cv=10)



print('Cross Validation Score','\n',

      'Mean=',score_list.mean())
select_columns = [i for i,j in zip(X_train.columns,np.ravel(log_reg.coef_)) if j!=0]

print('Select columns from LASSO:\n',select_columns)
print('Output:',y_pred[:10])
from xgboost import XGBClassifier

xgb = XGBClassifier(random_state=7)

xgb.fit(X_train,y_train)
y_pred=xgb.predict(X_train)

print(accuracy_score(y_train,y_pred))
from sklearn.model_selection import GridSearchCV

parameters = {'max_depth':[3,5,7],

              'n_estimators':[50,100,200],

              'learning_rate':[0.1,0.2,0.3],                            

             }

             

gd = GridSearchCV(estimator=XGBClassifier(booster='gbtree',random_state=7),

                  param_grid=parameters,                

                  cv=10,

                  scoring='accuracy')

gd.fit(X_train,y_train)
print(gd.best_estimator_)

print(gd.best_params_)

print(gd.best_score_)
xgb = XGBClassifier(learning_rate=0.2, max_depth=5, n_estimators=100,booster='gbtree',random_state=7)

xgb.fit(X_train,y_train)

y_pred = xgb.predict(X_train)



print('Model Accuracy = ',accuracy_score(y_train,y_pred))
y_pred_pa = xgb.predict_proba(X_train)



best_threshold, y_pred = get_prediction(y_train,y_pred_pa,[0.1,0.3,0.5,0.7,0.9])

print('Accuracy Improvement ( Threshold=',best_threshold,'):',round(accuracy_score(y_train,y_pred)*100,2))



best_threshold, y_pred = get_prediction(y_train,y_pred_pa,[0.45,0.5,0.55,0.6])

print('Accuracy Improvement ( Threshold=',best_threshold,'):',round(accuracy_score(y_train,y_pred)*100,2))
y_pred_pa = xgb.predict_proba(X_test)

y_pred = [1 if j>=0.5 else 0 for j in y_pred_pa[:,1]]

result = pd.DataFrame()



result['PassengerId'] = X_test_passenger_id

result['Survived'] = y_pred

result.columns = ['PassengerId','Survived']



result.to_csv("submission.csv", index=False)