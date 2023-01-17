import numpy as np
import pandas as pd
import os

pima=pd.read_csv("diabetes.csv", na_values=0)
pima.head()
pima.shape
X=pima.drop("Outcome",axis=1)
Y=pima["Outcome"]
X.isnull().sum()
X["Pregnancies"].unique()
col_list=X.columns.tolist()
col_list
for i in col_list:
    X[i].fillna(X[i].mean(),inplace=True)
    
X.isnull().sum()
Y.fillna(0,inplace=True)
Y.unique()
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.30, random_state=200)
print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)
pima.corr()
pima.columns.tolist()
import seaborn as sns
import matplotlib.pyplot as plt
#correlations of each features in dataset
plt.figure(figsize=(20,20))
#plot heat map
g=sns.heatmap(pima[pima.columns.tolist()].corr(),annot=True,cmap="RdYlGn")
g
import statsmodels.formula.api as smf
import statsmodels.api as sm
X_train.columns
glm1=smf.glm('y_train~Pregnancies+Glucose+BloodPressure+SkinThickness+Insulin+BMI+DiabetesPedigreeFunction+Age',data=X_train,family=sm.families.Binomial()).fit()
print(glm1.summary())
glm1.aic  
glm2=smf.glm('y_train~Pregnancies+Glucose+BMI+DiabetesPedigreeFunction',data=X_train,family=sm.families.Binomial()).fit()
print(glm2.summary())
glm2.aic
#Multicollinearity check

from statsmodels.stats.outliers_influence import variance_inflation_factor
idv = glm2.model.exog
vif = [variance_inflation_factor(idv,i) for i in range(idv.shape[1])]

pd.DataFrame({'Features':glm2.model.exog_names,'vif':vif})
X_test['prob']=glm2.predict(X_test)
X_test['prob'].head()
pred_y=X_test['prob'].map(lambda x:1 if x>0.5 else 0)
pred_y.head()
#Confusion matrix
from sklearn import metrics
metrics.confusion_matrix(y_test,pred_y)
from sklearn.metrics import accuracy_score
accuracy_score(y_test,pred_y)
# AUC
metrics.roc_auc_score(y_test,X_test['prob'])
X_test.columns

X_test.drop("prob",axis=1,inplace=True)
from sklearn.ensemble import RandomForestClassifier
clf=RandomForestClassifier(n_estimators=80,oob_score=True,n_jobs=-1,random_state=400)
clf.fit(X_train,y_train)
#Parameter tuning
for w in range(50,300,10):
    clf=RandomForestClassifier(oob_score=True,n_jobs=-1,n_estimators=w,random_state=400,max_depth=3)
    clf.fit(X_train,y_train)
    oob=clf.oob_score_
    print('For n_estimators = '+str(w))
    print('OOB score is '+str(oob))
    print('************************')
clf=RandomForestClassifier(n_estimators=140,oob_score=True,n_jobs=-1,random_state=400,max_depth=3)
clf.fit(X_train,y_train)
pred_y=clf.predict(X_test)
#Confusion matrix
from sklearn import metrics
metrics.confusion_matrix(y_test,pred_y)
from sklearn.metrics import accuracy_score
accuracy_score(y_test,pred_y)
a = [i for i in range(100,200,10)]

import xgboost as xg
clf=xg.XGBClassifier(objective='binary:logistic',reg_alpha=0.2,max_depth=4,random_state=2) #L1 regularization
clf.fit(X_train,y_train)
#Grid Search CV: Parameter tuning
import sklearn.model_selection as model_selection
model=model_selection.GridSearchCV(clf, param_grid={'max_depth':[3,6,9],'n_estimators':a,'reg_alpha':[0.1,0.2]})
model.fit(X_train,y_train)
model.best_params_
model.score(X_test,y_test)
pred_y=model.predict(X_test)
#Confusion matrix
from sklearn import metrics
metrics.confusion_matrix(y_test,pred_y)
from sklearn.metrics import accuracy_score
accuracy_score(y_test,pred_y)
