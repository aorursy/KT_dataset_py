# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))

#import numpy as np

import matplotlib.pyplot as plt

import seaborn as sn

import random

import time



random.seed(100)



### Data Preprocessing ###



dataset = pd.read_csv('../input/P39-Financial-Data.csv')





# Feature Engineering



dataset = dataset.drop(columns = ['months_employed'])

dataset['personal_account_months'] = (dataset.personal_account_m + (dataset.personal_account_y * 12))

dataset[['personal_account_m', 'personal_account_y', 'personal_account_months']].head()

dataset = dataset.drop(columns = ['personal_account_m', 'personal_account_y'])





# Any results you write to the current directory are saved as output.
dataset=pd.get_dummies(dataset)

dataset.columns

dataset=dataset.drop(columns=['pay_schedule_semi-monthly'])

#removing extra columns

response=dataset["e_signed"]

users=dataset["entry_id"]

dataset=dataset.drop(columns=["e_signed","entry_id"])

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test=train_test_split(dataset,response,test_size=0.2, random_state=0)

from sklearn.preprocessing import StandardScaler

sc_X = StandardScaler()

X_train2 = pd.DataFrame(sc_X.fit_transform(X_train))

X_test2 = pd.DataFrame(sc_X.transform(X_test))

X_train2.columns = X_train.columns.values

X_test2.columns = X_test.columns.values

X_train2.index = X_train.index.values

X_test2.index = X_test.index.values

X_train = X_train2

X_test = X_test2

from sklearn.linear_model import LogisticRegression

classifier=LogisticRegression(random_state=0,penalty='l1')

classifier.fit(X_train,y_train)

y_pred=classifier.predict(X_test)

from sklearn.metrics import confusion_matrix,accuracy_score,f1_score,precision_score,recall_score

acc=accuracy_score(y_test,y_pred)

prec=precision_score(y_test,y_pred)

rec=recall_score(y_test,y_pred)

f1=f1_score(y_test,y_pred)

results=pd.DataFrame([['Linear Regression (Lasso)',acc,prec,rec,f1]], columns=['Model','Accuracy','Precision','Recall','F1 Score'])

#svm

from sklearn.svm import SVC

classifier=SVC(random_state=0,kernel='linear')

classifier.fit(X_train,y_train)

y_pred=classifier.predict(X_test)

acc=accuracy_score(y_test,y_pred)

prec=precision_score(y_test,y_pred)

rec=recall_score(y_test,y_pred)

f1=f1_score(y_test,y_pred)

model_results=pd.DataFrame([['SVM(linear)',acc,prec,rec,f1]],columns=['Model','Accuracy','Precision','Recall','F1 Score'])

results=results.append(model_results,ignore_index=True)

from sklearn.svm import SVC

classifier=SVC(random_state=0,kernel='rbf')

classifier.fit(X_train,y_train)

y_pred=classifier.predict(X_test)

acc=accuracy_score(y_test,y_pred)

prec=precision_score(y_test,y_pred)

rec=recall_score(y_test,y_pred)

f1=f1_score(y_test,y_pred)

model_results=pd.DataFrame([['SVM(RBF)',acc,prec,rec,f1]],columns=['Model','Accuracy','Precision','Recall','F1 Score'])

results=results.append(model_results,ignore_index=True)



results

#randomforest

from sklearn.ensemble import RandomForestClassifier

classifier=RandomForestClassifier(random_state=0, n_estimators=100,criterion='entropy')

classifier.fit(X_train,y_train)

y_pred=classifier.predict(X_test)

acc=accuracy_score(y_test,y_pred)

prec=precision_score(y_test,y_pred)

rec=recall_score(y_test,y_pred)

f1=f1_score(y_test,y_pred)

model_results=pd.DataFrame([['Random Forest (n=100)',acc,prec,rec,f1]],columns=['Model','Accuracy','Precision','Recall','F1 Score'])

results=results.append(model_results,ignore_index=True)

results

 
results
from sklearn.model_selection import cross_val_score
accuracies=cross_val_score(estimator=classifier,X=X_train, y=y_train,cv=10)

print("Random Forest Classifier Accuracy: %0.2f(+/- %0.2f)"%(accuracies.mean(),accuracies.std() *2))

#grid search

parameters={"max_depth":[3,None],

           "max_features":[1,5,10],

           'min_samples_split':[2,5,10],

           'min_samples_leaf':[1,5,10],

           "bootstrap":[True,False],

           "criterion":["entropy"]}



from sklearn.model_selection import GridSearchCV

grid_search=GridSearchCV(estimator=classifier,

                        param_grid=parameters,

                        scoring="accuracy",

                        cv=3,

                        n_jobs=2)

t0=time.time()

grid_search=grid_search.fit(X_train,y_train)

t1=time.time()

print("Took %0.2f seconds"%(t1-t0))

rf_best_accuracy=grid_search.best_score_

rf_best_parameters=grid_search.best_params_

rf_best_accuracy,rf_best_parameters



parameters={"max_depth":[None],

            "max_features":[3,5,7],

            'min_samples_split':[8,10,12],

            'min_samples_leaf':[1,2,3],

            "bootstrap":[True],

            "criterion":["entropy"]}

from sklearn.model_selection import GridSearchCV

grid_search=GridSearchCV(estimator=classifier,

                        param_grid=parameters,

                        scoring="accuracy",

                        cv=3,

                        n_jobs=2)

t0=time.time()

grid_search=grid_search.fit(X_train,y_train)

t1=time.time()

print("Took %0.2f seconds"%(t1-t0))

rf_best_accuracy=grid_search.best_score_

rf_best_parameters=grid_search.best_params_

rf_best_accuracy,rf_best_parameters





y_pred=grid_search.predict(X_test)

acc=accuracy_score(y_test,y_pred)

prec=precision_score(y_test,y_pred)

rec=recall_score(y_test,y_pred)

f1=f1_score(y_test,y_pred)



model_results = pd.DataFrame([['Random Forest (n=100, GSx2 + Entropy)', acc, prec, rec, f1]],

               columns = ['Model', 'Accuracy', 'Precision', 'Recall', 'F1 Score'])



results = results.append(model_results, ignore_index = True)

# Round 1: Gini

parameters = {"max_depth": [3, None],

              "max_features": [1, 5, 10],

              'min_samples_split': [2, 5, 10],

              'min_samples_leaf': [1, 5, 10],

              "bootstrap": [True, False],

              "criterion": ["gini"]}



from sklearn.model_selection import GridSearchCV

grid_search = GridSearchCV(estimator = classifier, # Make sure classifier points to the RF model

                           param_grid = parameters,

                           scoring = "accuracy",

                           cv = 3,

                           n_jobs = 2)



t0 = time.time()

grid_search = grid_search.fit(X_train, y_train)

t1 = time.time()

print("Took %0.2f seconds" % (t1 - t0))



rf_best_accuracy = grid_search.best_score_

rf_best_parameters = grid_search.best_params_

rf_best_accuracy, rf_best_parameters



# Round 2: Gini

parameters = {"max_depth": [None],

              "max_features": [8, 10, 12],

              'min_samples_split': [2, 3, 4],

              'min_samples_leaf': [8, 10, 12],

              "bootstrap": [True],

              "criterion": ["gini"]}



from sklearn.model_selection import GridSearchCV

grid_search = GridSearchCV(estimator = classifier, # Make sure classifier points to the RF model

                           param_grid = parameters,

                           scoring = "accuracy",

                           cv = 3,

                           n_jobs = 2)



t0 = time.time()

grid_search = grid_search.fit(X_train, y_train)

t1 = time.time()

print("Took %0.2f seconds" % (t1 - t0))



rf_best_accuracy = grid_search.best_score_

rf_best_parameters = grid_search.best_params_

rf_best_accuracy, rf_best_parameters





# Predicting Test Set

y_pred = grid_search.predict(X_test)

acc = accuracy_score(y_test, y_pred)

prec = precision_score(y_test, y_pred)

rec = recall_score(y_test, y_pred)

f1 = f1_score(y_test, y_pred)



model_results = pd.DataFrame([['Random Forest (n=100, GSx2 + Gini)', acc, prec, rec, f1]],

               columns = ['Model', 'Accuracy', 'Precision', 'Recall', 'F1 Score'])



results = results.append(model_results, ignore_index = True)



## EXTRA: Confusion Matrix

cm = confusion_matrix(y_test, y_pred) 

df_cm = pd.DataFrame(cm, index = (0, 1), columns = (0, 1))

plt.figure(figsize = (10,7))

sn.set(font_scale=1.4)

sn.heatmap(df_cm, annot=True, fmt='g')

print("Test Data Accuracy: %0.4f" % accuracy_score(y_test, y_pred))





#Final Results



final_results = pd.concat([y_test, users], axis = 1).dropna()

final_results['predictions'] = y_pred

final_results = final_results[['entry_id', 'e_signed', 'predictions']]






