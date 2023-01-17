import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import sklearn

import os

import matplotlib.pyplot as plt

import seaborn as sns

from pandas_profiling import ProfileReport

sns.set_style('darkgrid')

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
data = pd.read_csv('/kaggle/input/pima-indians-diabetes-database/diabetes.csv')

data.head(3)
profile = ProfileReport(data, title="Pandas Profiling Report")
## Printing the complete Data Analysis Report!

## we have a balanced dataset

profile
## The 'Outcome' column tells us whether the person has Diabetes or not

## 1- Diabetic

## 0- Non Diabetic



X = data.iloc[:,:-1] # Independent variables

y = data['Outcome'] # Dependent Variables
data = data.drop(data[data['Pregnancies']>11].index)

data = data.drop(data[data['Glucose']<30].index)

data = data.drop(data[data['BloodPressure']>110].index)

data = data.drop(data[data['BloodPressure']<20].index)

data = data.drop(data[data['SkinThickness']>80].index)

data = data.drop(data[data['BMI']>55].index)

data = data.drop(data[data['BMI']<10].index)

data = data.drop(data[data['DiabetesPedigreeFunction']>1.6].index)

data = data.drop(data[data['Insulin']>400].index)

data = data.drop(data[data['Age']>80].index)
plt.figure(figsize=(11,10))

correlation = X.corr()

sns.heatmap(correlation,linewidth = 0.7,cmap = 'Blues',annot = True)
X = X.loc[data.index]

y = y.loc[data.index]
from sklearn.model_selection import train_test_split

X_train,X_val,y_train,y_val = train_test_split(X,y,test_size = 0.25,random_state = 100)
from sklearn.metrics import log_loss,accuracy_score,confusion_matrix,f1_score,recall_score

from xgboost import XGBClassifier

from sklearn.model_selection import cross_val_score,StratifiedKFold,RandomizedSearchCV

xgb = XGBClassifier(booster ='gbtree',objective ='binary:logistic')
from sklearn.model_selection import RandomizedSearchCV



param_lst = {

    'learning_rate' : [0.01, 0.1, 0.15, 0.3, 0.5,0.4],

    'n_estimators' : [100, 500, 1000,1500,2000],

    'max_depth' : [2,3,5, 6,8, 9],

    'min_child_weight' : [1, 5, 10],

    'reg_alpha' : [0.001, 0.01, 0.1],

    'reg_lambda' : [0.001, 0.01, 0.1],

    'colsample_bytree' : [0.3,0.4,0.5,0.7],

    'gamma' : [0.0,0.1,0.2,0.3,0.4]

}



xgb_tuning = RandomizedSearchCV(estimator = xgb, param_distributions = param_lst ,

                          n_iter = 5,

                        cv =6)

       

xgb_search = xgb_tuning.fit(X_train,y_train,

                           early_stopping_rounds = 5,

                           eval_set=[(X_val,y_val)],

                           verbose = False)



## checking for the best paramter values that the model took



best_param = xgb_search.best_params_

xgb = XGBClassifier(**best_param)

print(best_param)
## check the best estimators

xgb_search.best_estimator_
y_pred = xgb_search.predict(X_val)

score0 = accuracy_score(y_pred,y_val)

#print(round(score0*100,4))

print('Score: {}%'.format(round(score0*100,4)))
## checking  'Accuracy' value using Cross Validation menthod

acc_scores1_xgb =  cross_val_score(xgb_search,X,y,n_jobs=5,

                                 cv = StratifiedKFold(n_splits=10),

                                 scoring = 'accuracy')

acc_scores1_xgb
from sklearn.metrics import RocCurveDisplay

from sklearn.metrics import roc_auc_score,roc_curve,auc

from sklearn import metrics



fpr, tpr, threshold = metrics.roc_curve(y_val, y_pred)

roc_auc = metrics.auc(fpr, tpr)



plt.title('Receiver Operating Characteristic')

plt.plot(fpr, tpr, 'red', label = 'ROC AUC score = %0.2f' % roc_auc)

plt.legend(loc = 'lower right')

plt.plot([0, 1], [0, 1],'b--')

plt.xlim([0, 1])

plt.ylim([0, 1])

plt.ylabel('True Positive Rate')

plt.xlabel('False Positive Rate')

plt.show()
from sklearn.linear_model import LogisticRegression

lr= LogisticRegression()

lr.fit(X_train,y_train)
log_scores_logi = -1 * cross_val_score(lr, X, y,

                              cv=5,

                              scoring='neg_log_loss')

acc_scores1_logi =  cross_val_score(lr,X,y,

                                 cv = 5,

                                 scoring = 'accuracy')

f_score_logi =  cross_val_score(lr,X,y,

                                 cv = 5,

                                 scoring = 'f1')
print("log_loss scores:\n", log_scores_logi)

print("Accuracy scores:\n", acc_scores1_logi)

print("f1_score scores:\n", f_score_logi)
print(acc_scores1_logi.mean())