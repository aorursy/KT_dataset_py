%matplotlib inline

import pandas as pd

import numpy as np

from matplotlib import pyplot as plt

import seaborn as sns
indiabetes = pd.read_csv("../input/pyms-diabete/diabete.csv")
indiabetes.columns
indiabetes.count()
indiabetes=indiabetes[(indiabetes.glucose!=0) & (indiabetes.tension !=0) & (indiabetes.bmi!=0)]

indiabetes.count()
plt.hist(indiabetes.thickness, bins=100)
indiabetes.thickness=indiabetes.thickness.replace(0,np.nan)
indiabetes=indiabetes.fillna(method="pad")
plt.hist(indiabetes.thickness, bins=100)
plt.hist(indiabetes.insulin, 100)
indiabetes_where_insulin=indiabetes[indiabetes.insulin!=0]

plt.hist(indiabetes_where_insulin.insulin, 100)


g = sns.lmplot(x="glucose", y="insulin", hue="diabete", data=indiabetes_where_insulin,

               palette="Set1")
sns.regplot(x="glucose", y="insulin", data=indiabetes_where_insulin,

                 scatter_kws={"s": 80}, ci=None, order=5)
indiabetes.insulin=indiabetes.insulin.replace(0,np.nan)

indiabetes.insulin=indiabetes['insulin'].interpolate(method='polynomial', order=5)

indiabetes=indiabetes.dropna()

indiabetes.count()
sns.regplot(x="glucose", y="insulin", data=indiabetes,

                 scatter_kws={"s": 80}, ci=None, order=5)
plt.hist(indiabetes.insulin, 100)
X = indiabetes.drop(['diabete'], axis=1)

y = indiabetes.diabete

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

from sklearn import ensemble

rf = ensemble.RandomForestClassifier()

rf.fit(X_train, y_train)

y_rf = rf.predict(X_test)

from sklearn.metrics import classification_report, confusion_matrix, roc_curve, roc_auc_score,auc, accuracy_score
print(classification_report(y_test, y_rf))
print(confusion_matrix(y_test,y_rf))
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

              'n_estimators': [10, 50,250, 500],

              'min_samples_leaf': [1, 20, 50]

             }

estimator = ensemble.RandomForestClassifier()

rf_gs = model_selection.GridSearchCV(estimator, param_grid)

rf_gs.fit(X_train, y_train)

print(rf_gs.best_params_)
rf2=rf_gs.best_estimator_

y_rf2=rf2.predict(X_test)

print(classification_report(y_test, y_rf2))
print(confusion_matrix(y_test,y_rf2))
import xgboost as XGB

xgb  = XGB.XGBClassifier()

xgb.fit(X_train, y_train)

y_xgb = xgb.predict(X_test)

cm = confusion_matrix(y_test, y_xgb)

print(cm)

print(classification_report(y_test, y_xgb))
print(confusion_matrix(y_test,y_xgb))
importances = rf2.feature_importances_

indices = np.argsort(importances)

plt.figure(figsize=(8,5))

plt.barh(range(len(indices)), importances[indices], color='b', align='center')

plt.yticks(range(len(indices)), X_train.columns[indices])

plt.title('Importance des caracteristiques')
sns.pairplot(indiabetes, hue="diabete")
indiabetes_2=indiabetes.drop(["tension","thickness","insulin","pedigree"],axis=1)

sns.pairplot(indiabetes_2, hue="diabete")
X_2 = indiabetes_2.drop(['diabete'], axis=1)

y_2 = indiabetes_2.diabete

X_train_2, X_test_2, y_train_2, y_test_2 = train_test_split(X_2, y_2, test_size=0.2, random_state=1)



xgb_2  = XGB.XGBClassifier()

xgb_2.fit(X_train_2, y_train_2)

y_xgb_2 = xgb_2.predict(X_test_2)

print(confusion_matrix(y_test_2, y_xgb_2))

print(classification_report(y_test_2, y_xgb_2))