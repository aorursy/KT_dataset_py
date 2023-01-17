import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score 
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer
from sklearn.metrics import mean_squared_error
from math import sqrt
import warnings
warnings.filterwarnings('ignore')

%matplotlib inline
train = pd.read_csv( "/kaggle/input/eval-lab-1-f464-v2/train.csv")
test = pd.read_csv( "/kaggle/input/eval-lab-1-f464-v2/test.csv")
#Checking for missing values
missing_count = train.isnull().sum()
missing_count[missing_count > 0]
#Replacing null values with mean
train.fillna(value=train.mean(),inplace = True)
test.fillna(value=test.mean(),inplace = True)
train.describe()
#Feature Selection
numerical_features = ['feature1','feature2','feature3','feature5','feature6','feature9','feature10']  #Selecting numerical features with higher variance & range 
category_features = ['type']
X = train[numerical_features+category_features]
y = train['rating']
#Encoding the categorical variable 

type_code = {'old':0,'new':1}
X['type'] = X['type'].map(type_code)
X_train,X_val,y_train,y_val = train_test_split(X,y,test_size=0.33,random_state=42) 
scaler = RobustScaler()
X_train[numerical_features] = scaler.fit_transform(X_train[numerical_features])
X_val[numerical_features] = scaler.transform(X_val[numerical_features])  
clf1 = DecisionTreeClassifier().fit(X_train,y_train)

y_pred_1 = clf1.predict(X_val)
y_pred_1 = [round(val) for val in y_pred_1]

acc1 = accuracy_score(y_pred_1,y_val)*100

print("Accuracy score of clf1: {}".format(acc1))
clf2 = RandomForestClassifier().fit(X_train,y_train)

y_pred_2 = clf2.predict(X_val)
y_pred_2 = [round(val) for val in y_pred_2]

acc2 = accuracy_score(y_pred_2,y_val)*100

print("Accuracy score of clf2: {}".format(acc2))
clf3 = RandomForestClassifier()        #Initialize the classifier object

parameters = {'n_estimators':[10,50,100]}    #Dictionary of parameters

scorer = make_scorer(accuracy_score)         #Initialize the scorer using make_scorer

grid_obj = GridSearchCV(clf3,parameters,scoring=scorer)         #Initialize a GridSearchCV object with above parameters,scorer and classifier

grid_fit = grid_obj.fit(X_train,y_train)        #Fit the gridsearch object with X_train,y_train

best_clf = grid_fit.best_estimator_         #Get the best estimator      
optimized_predictions = best_clf.predict(X_val)        

optimized_predictions = [round(val) for val in optimized_predictions]

acc_op = accuracy_score(y_val, optimized_predictions)*100         #Calculate accuracy for optimized model

print("Accuracy score on optimized model:{}".format(acc_op))
X_test = test[numerical_features + category_features]
X_test['type'] = X_test['type'].map(type_code)  #Encoding categorical data
X_test[numerical_features] = scaler.transform(X_test[numerical_features])
optimized_predictions = best_clf.predict(X_test)
optimized_predictions = [round(val) for val in optimized_predictions]
#Submission 1
submission = pd.DataFrame({'id':test['id'],'rating':optimized_predictions})
submission.to_csv('Lab1_4.csv',index=False)
X1 = train[numerical_features + category_features]   
y1 = train['rating']
#Encoding the categorical variable 

type_code = {'old':0,'new':1}
X1['type'] = X1['type'].map(type_code)
#Traintest Split
X_train1,X_val1,y_train1,y_val1 = train_test_split(X1,y1,test_size=0.33,random_state=42) 
#Scaling Values
scaler = RobustScaler()
X_train1[numerical_features] = scaler.fit_transform(X_train1[numerical_features])
X_val1[numerical_features] = scaler.transform(X_val1[numerical_features])  
#RandomForestClassifier with GridSearchCV with more parameters
clf = RandomForestClassifier()        #Initialize the classifier object

parameters = {'n_estimators':[10,50,100,105,110,115,120,125,130,135,140,145,150]}    #Dictionary of parameters

scorer = make_scorer(accuracy_score)         #Initialize the scorer using make_scorer

grid_obj = GridSearchCV(clf,parameters,scoring=scorer)         #Initialize a GridSearchCV object with above parameters,scorer and classifier

grid_fit = grid_obj.fit(X_train1,y_train1)        #Fit the gridsearch object with X_train,y_train

best_clf1 = grid_fit.best_estimator_                #Get the best estimator      
optimized_predictions = best_clf1.predict(X_val1)        

optimized_predictions = [round(val) for val in optimized_predictions]

acc_op = accuracy_score(y_val, optimized_predictions)*100         #Calculate accuracy for optimized model

print("Accuracy score on optimized model:{}".format(acc_op))
#Scaling test data
X_test1 = test[numerical_features + category_features]
X_test1['type'] = X_test1['type'].map(type_code)
X_test1[numerical_features] = scaler.transform(X_test1[numerical_features])  #Scaling test data
#Prediction
optimized_predictions = best_clf1.predict(X_test1)
optimized_predictions = [round(val) for val in optimized_predictions]
#Submission 2
submission = pd.DataFrame({'id':test['id'],'rating':optimized_predictions})
submission.to_csv('Lab1_7.csv',index=False)