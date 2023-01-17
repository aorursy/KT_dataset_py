import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

from sklearn.tree import DecisionTreeClassifier

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score,roc_auc_score,roc_curve,classification_report,confusion_matrix
df= pd.read_csv('../input/pima-indians-diabetes-database/diabetes.csv')

df.head()
df.isnull().sum()
df.describe().T
df.shape
cols=list(df.columns)

for i in cols:

    sns.boxplot(df[i])

    plt.show()

x=df.drop('Outcome',axis=1)

y=df['Outcome']
from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3,random_state=1)
from  sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier()

dt.fit(x_train,y_train)
from sklearn.metrics import confusion_matrix,accuracy_score,roc_auc_score,roc_curve,classification_report 
y_pred_train=dt.predict(x_train)

y_proba_train=dt.predict_proba(x_train)[:,1]

y_pred_test=dt.predict(x_test)

y_proba_test=dt.predict_proba(x_test)[:,1]
print('accuracy score for train: ',accuracy_score(y_train,y_pred_train))

print('accuracy score for test: ',accuracy_score(y_test,y_pred_test))

print('AUC score for train: ',roc_auc_score(y_train,y_proba_train))

print('AUC score for train: ',roc_auc_score(y_test,y_pred_test))
from sklearn.model_selection import GridSearchCV,RandomizedSearchCV
from scipy.stats import randint

dt=DecisionTreeClassifier()

params={'max_depth': randint(2,20),

       'criterion':['gini','entropy'],

      'min_samples_split':randint(2,10),

    'min_samples_leaf':randint(1,10),

    'max_features':randint(1,9),

    'max_leaf_nodes':randint(2,20)}

rand= RandomizedSearchCV(estimator=dt,param_distributions=params,cv=4)

rand.fit(x,y)

rand.best_params_

dt=DecisionTreeClassifier(**rand.best_params_)

dt.fit(x_train,y_train)

y_pred_train=dt.predict(x_train)

y_proba_train=dt.predict_proba(x_train)[:,1]

y_pred_test=dt.predict(x_test)

y_proba_test=dt.predict_proba(x_test)[:,1]

print('accuracy score for train: ',accuracy_score(y_train,y_pred_train))

print('accuracy score for test: ',accuracy_score(y_test,y_pred_test))

print('AUC score for train: ',roc_auc_score(y_train,y_proba_train))

print('AUC score for train: ',roc_auc_score(y_test,y_proba_test))
fpr,tpr,thresh= roc_curve(y_test,y_proba_test)

plt.plot(fpr,tpr)

plt.plot(fpr,fpr,'r')

plt.xlabel('fpr')

plt.ylabel('tpr')

plt.title('ROC-AUC Curve')
confusion_matrix(y_test,y_pred_test)
print(classification_report(y_test,y_pred_test))
from sklearn.ensemble import RandomForestClassifier
rf=RandomForestClassifier()

rf.fit(x_train,y_train)

y_pred_train=rf.predict(x_train)

y_proba_train=rf.predict_proba(x_train)[:,1]

y_pred_test=rf.predict(x_test)

y_proba_test=rf.predict_proba(x_test)[:,1]

print('accuracy score for train: ',accuracy_score(y_train,y_pred_train))

print('accuracy score for test: ',accuracy_score(y_test,y_pred_test))

print('AUC score for train: ',roc_auc_score(y_train,y_proba_train))

print('AUC score for train: ',roc_auc_score(y_test,y_proba_test))
## This also seems to be overfittng hence we will go for Hyperparameter Tuning
params={'n_estimators':randint(50,150),

    'criterion':['gini','entropy'],

    'max_depth':randint(2,20),

    'min_samples_split':randint(2,20),

    'min_samples_leaf':randint(1,20),

    'max_leaf_nodes':randint(2,20)}



rand_search=RandomizedSearchCV(rf,params,cv=5)

rand_search.fit(x,y)

rand_search.best_params_
rf=RandomForestClassifier(**rand_search.best_params_)

rf.fit(x_train,y_train)

y_pred_train=rf.predict(x_train)

y_proba_train=rf.predict_proba(x_train)[:,1]

y_pred_test=rf.predict(x_test)

y_proba_test=rf.predict_proba(x_test)[:,1]

print('accuracy score for train: ',accuracy_score(y_train,y_pred_train))

print('accuracy score for test: ',accuracy_score(y_test,y_pred_test))

print('AUC score for train: ',roc_auc_score(y_train,y_proba_train))

print('AUC score for train: ',roc_auc_score(y_test,y_proba_test))
## The accuracy and AUC score seems to be better in Random Forest 
fpr,tpr,thresh= roc_curve(y_test,y_proba_test)

plt.plot(fpr,tpr)

plt.plot(fpr,fpr,'r')

plt.xlabel('fpr')

plt.ylabel('tpr')

plt.title('ROC-AUC Curve')
confusion_matrix(y_test,y_pred_test)
print(classification_report (y_test,y_pred_test))
import lightgbm 
lgbm=lightgbm.LGBMClassifier()

lgbm.fit(x_train,y_train)

y_pred_train=lgbm.predict(x_train)

y_proba_train=lgbm.predict_proba(x_train)[:,1]

y_pred_test=lgbm.predict(x_test)

y_proba_test=lgbm.predict_proba(x_test)[:,1]

print('accuracy score for train: ',accuracy_score(y_train,y_pred_train))

print('accuracy score for test: ',accuracy_score(y_test,y_pred_test))

print('AUC score for train: ',roc_auc_score(y_train,y_proba_train))

print('AUC score for train: ',roc_auc_score(y_test,y_proba_test))
param={'num_leaves': randint(2,50),

    'max_depth':randint(2,30),

    'learning_rate':[0.1,0.01,0.2],

    'n_estimators': randint(50,100),

    'min_child_samples':randint(10,40)}



rand_search=RandomizedSearchCV(lgbm,param,cv=5)

rand_search.fit(x,y)

rand_search.best_params_
lgbm=lightgbm.LGBMClassifier(**rand_search.best_params_)

lgbm.fit(x_train,y_train)

y_pred_train=lgbm.predict(x_train)

y_proba_train=lgbm.predict_proba(x_train)[:,1]

y_pred_test=lgbm.predict(x_test)

y_proba_test=lgbm.predict_proba(x_test)[:,1]

print('accuracy score for train: ',accuracy_score(y_train,y_pred_train))

print('accuracy score for test: ',accuracy_score(y_test,y_pred_test))

print('AUC score for train: ',roc_auc_score(y_train,y_proba_train))

print('AUC score for train: ',roc_auc_score(y_test,y_proba_test))
#### The overfitting has reduced but still their is a huge difference between the accuracy and AUC score of train and test data
fpr,tpr,thresh= roc_curve(y_test,y_proba_test)

plt.plot(fpr,tpr)

plt.plot(fpr,fpr,'r')

plt.xlabel('fpr')

plt.ylabel('tpr')

plt.title('ROC-AUC Curve')
confusion_matrix(y_test,y_pred_test)
print(classification_report (y_test,y_pred_test))
import pandas as pd

diabetes = pd.read_csv("../input/pima-indians-diabetes-database/diabetes.csv")