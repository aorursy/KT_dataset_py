import pandas as pd

import numpy as np

from matplotlib import pyplot as plt

import seaborn as sns
df = pd.read_csv("../input/pima-indians-diabetes-database/diabetes.csv")
df.head()
df.columns
df.count()
num_missing = (df[['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin',

       'BMI', 'DiabetesPedigreeFunction', 'Age','Outcome']] == 0).sum()

print(num_missing)
from numpy import nan

df[['Glucose','BloodPressure','SkinThickness','Insulin','BMI']] = df[['Glucose','BloodPressure','SkinThickness','Insulin','BMI']].replace(0, nan)
df.fillna(df.mean(), inplace=True)
df.head(100)
X = df.drop(['Outcome'], axis=1)

y = df.Outcome
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
from sklearn.linear_model import LogisticRegression

lr = LogisticRegression()

lr.fit(X_train,y_train)

y_lr = lr.predict(X_test)
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, roc_auc_score,auc, accuracy_score
print(accuracy_score(y_test, y_lr))
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

plt.plot([0,1],[0,1],'r--')        # plus mauvaise courbe

plt.plot([0,0,1],[0,1,1],'g:')     # meilleure courbe

plt.xlim([-0.1,1.2])

plt.ylim([-0.1,1.2])

plt.ylabel('True Positive Rate')

plt.xlabel('False Positive Rate')
from sklearn import ensemble

rf = ensemble.RandomForestClassifier()

rf.fit(X_train, y_train)

y_rf = rf.predict(X_test)

rf_score = accuracy_score(y_test, y_rf)

print(rf_score)

print(classification_report(y_test, y_rf))
rf1 = ensemble.RandomForestClassifier(n_estimators=20, min_samples_leaf=15, max_features=5)

rf1.fit(X_train, y_train)

y_rf1 = rf.predict(X_test)

rf1_score = accuracy_score(y_test, y_rf1)

print(rf1_score)
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

              'n_estimators': [1,100,500],

              'min_samples_leaf': [1,100,300,500]

             }

estimator = ensemble.RandomForestClassifier()

rf_gs = model_selection.GridSearchCV(estimator, param_grid)
rf_gs.fit(X_train, y_train)

print(rf_gs.best_params_)
rf2 = rf_gs.best_estimator_

y_rf2 = rf2.predict(X_test)

rf2_score = accuracy_score(y_test, y_rf2)

print(rf2_score)
print(classification_report(y_test, y_rf2))
importances = rf2.feature_importances_

indices = np.argsort(importances)

plt.figure(figsize=(8,5))

plt.barh(range(len(indices)), importances[indices], color='b', align='center')

plt.yticks(range(len(indices)), X_train.columns[indices])

plt.title('Importance des caracteristiques')
import xgboost as XGB

xgb  = XGB.XGBClassifier()

xgb.fit(X_train, y_train)

y_xgb = xgb.predict(X_test)

rf2_score = accuracy_score(y_test, y_xgb)

print(rf2_score)
cm = confusion_matrix(y_test, y_xgb)

print(cm)

print(classification_report(y_test, y_xgb))
from sklearn import svm

clf = svm.SVC()

clf.fit(X_train, y_train)

y_clf = clf.predict(X_test)

rf2_score = accuracy_score(y_test, y_clf)

print(rf2_score)
cm = confusion_matrix(y_test, y_clf)

print(cm)

print(classification_report(y_test, y_clf))
from sklearn.neighbors.nearest_centroid import NearestCentroid

import numpy as np

clf = NearestCentroid()

clf.fit(X_train, y_train)

y_clf = clf.predict(X_test)

rf2_score = accuracy_score(y_test, y_clf)

print(rf2_score)
cm = confusion_matrix(y_test, y_clf)

print(cm)

print(classification_report(y_test, y_clf))