%matplotlib inline
import pandas as pd

import numpy as np

from matplotlib import pyplot as plt

import seaborn as sns
diab = pd.read_csv("../input/pima-indians-diabetes-database/diabetes.csv")
diab.head().T
diab.count()    
preg = (diab.Pregnancies != 0.000)

preg.value_counts()
diab[preg].head()
diab[preg].Outcome.value_counts()
pos = (diab.Outcome == 1)

neg = (diab.Outcome == 0)

preg_pos = preg & pos

preg_neg = preg & neg
prob_pos = diab[preg_pos].Pregnancies.count()/diab[preg].Pregnancies.count()

prob_neg = diab[preg_neg].Pregnancies.count()/diab[preg].Pregnancies.count()

print('Probabilité de positive:',prob_pos,'->',100*prob_pos,'%')

print('Probabilité de negative:',prob_neg,'->',100*prob_neg,'%')
X = diab.drop(['Outcome'], axis=1)

y = diab.Outcome
from sklearn.model_selection import train_test_split



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
print(X_train.shape)

print(X_test.shape)
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()

lr.fit(X_train,y_train)
y_lr = lr.predict(X_test)
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, roc_auc_score,auc, accuracy_score
print(confusion_matrix(y_test,y_lr))
print(accuracy_score(y_test,y_lr))
print(classification_report(y_test, y_lr))
probas = lr.predict_proba(X_test)
print(probas)
dfprobas = pd.DataFrame(probas,columns=['proba_0','proba_1'])

dfprobas['y'] = np.array(y_test)
dfprobas
plt.figure(figsize=(10,10))

sns.distplot(1-dfprobas.proba_0[dfprobas.y==0], bins=50)

sns.distplot(dfprobas.proba_1[dfprobas.y==1], bins=50)
false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test,probas[:, 1])

roc_auc = auc(false_positive_rate, true_positive_rate)

print (roc_auc)
plt.figure(figsize=(12,12))

plt.title('Receiver Operating Characteristic')

plt.plot(false_positive_rate, true_positive_rate, 'b', label='AUC = %0.2f'% roc_auc)

plt.legend(loc='lower right')

plt.plot([0,1],[0,1],'r--')    

plt.plot([0,0,1],[0,1,1],'g:') 

plt.xlim([-0.1,1.2])

plt.ylim([-0.1,1.2])

plt.ylabel('True Positive Rate')

plt.xlabel('False Positive Rate')
from sklearn import ensemble

rf = ensemble.RandomForestClassifier()

rf.fit(X_train, y_train)

y_rf = rf.predict(X_test)
print(classification_report(y_test, y_rf))
cm = confusion_matrix(y_test, y_rf)

print(cm)
rf1 = ensemble.RandomForestClassifier(n_estimators=10, min_samples_leaf=10, max_features=3)

rf1.fit(X_train, y_train)

y_rf1 = rf.predict(X_test)

print(classification_report(y_test, y_rf1))
from sklearn.model_selection import validation_curve

params = np.arange(1, 300,step=30)

train_score, val_score = validation_curve(rf, X, y, 'n_estimators', params, cv=7)

plt.figure(figsize=(12,12))

plt.plot(params, np.median(train_score, 1), color='blue', label='training score')

plt.plot(params, np.median(val_score, 1), color='red', label='validation score')

plt.legend(loc='best')

plt.ylim(0, 1)

plt.xlabel('n_estimators')

plt.ylabel('score');
from sklearn import model_selection
param_grid = {

              'n_estimators': [10, 100, 500],

              'min_samples_leaf': [1, 20, 50]

             }

estimator = ensemble.RandomForestClassifier()

rf_gs = model_selection.GridSearchCV(estimator, param_grid)
rf_gs.fit(X_train, y_train)
print(rf_gs.best_params_)
rf2 = rf_gs.best_estimator_
y_rf2 = rf2.predict(X_test)
print(classification_report(y_test, y_rf2))
importances = rf2.feature_importances_

indices = np.argsort(importances)
plt.figure(figsize=(8,5))

plt.barh(range(len(indices)), importances[indices], color='b', align='center')

plt.yticks(range(len(indices)), X_train.columns[indices])

plt.title('Importance des caracteristiques')