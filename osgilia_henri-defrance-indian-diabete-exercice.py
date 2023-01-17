# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
t = pd.read_csv("../input/pima-indians-diabetes-database/diabetes.csv")
# Pandas : librairie de manipulation de données

# NumPy : librairie de calcul scientifique

# MatPlotLib : librairie de visualisation et graphiques

# SeaBorn : librairie de graphiques avancés

import pandas as pd

import numpy as np

from matplotlib import pyplot as plt

import seaborn as sns
t.head().T

t.columns
t.count()
def detect_0(df) :

    columns = list(df)

    str = ''

    for i in columns: 

        res = (df[i] == 0).any();

        str += '{} {}\n'.format(i,res)

    return str

        

strRes = detect_0(t)

print(strRes)
def replace_0(df,col) :

    df1 = df.copy()

    n = df.shape[0]

    m = df[col].mean()

    s = df[col].std()

    for i in range(n) :

        if df.loc[i,col]==0 :

            df1.loc[i,col] = np.random.normal(m,s)

    return df1



t = replace_0(t,'Glucose')

t = replace_0(t,'BloodPressure')

t = replace_0(t,'SkinThickness')

t = replace_0(t,'Insulin')

t = replace_0(t,'BMI')

strRes = detect_0(t)

print(strRes)


plt.hist(t.Glucose, bins=80)
plt.hist(t.BloodPressure, bins=80)
plt.hist(t.SkinThickness, bins=80)
plt.hist(t.Insulin, bins=80)
plt.hist(t.BMI, bins=80)
X = t.drop(['Outcome'], axis=1)

y = t.Outcome
from sklearn.model_selection import train_test_split



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()

lr.fit(X_train,y_train)

y_lr = lr.predict(X_test)

# Importation des méthodes de mesure de performances

from sklearn.metrics import classification_report, confusion_matrix, roc_curve, roc_auc_score,auc, accuracy_score
rf_score = accuracy_score(y_test, y_lr)

print(rf_score)
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

rf1_score = accuracy_score(y_test, y_rf2)

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