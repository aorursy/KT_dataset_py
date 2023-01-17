# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import pandas as pd

import numpy as np

from matplotlib import pyplot as plt

import seaborn as sns
diabete = pd.read_csv("../input/pyms-diabete/diabete.csv")
diabete.head().T
diabete.describe()
diabete.diabete.value_counts()
fig = sns.FacetGrid(diabete, hue="diabete", aspect=3, palette="Set2") # aspect=3 permet d'allonger le graphique

fig.map(sns.kdeplot, "n_pregnant", shade=True)

fig.add_legend()
fig = sns.FacetGrid(diabete, hue="diabete", aspect=3, palette="Set2") # aspect=3 permet d'allonger le graphique

fig.map(sns.kdeplot, "glucose", shade=True)

fig.add_legend()
sns.set(style='ticks')

plt.figure(figsize=(20,20))

sns.pairplot(diabete, hue='diabete')
print(diabete[diabete.tension == 0].shape[0])

print(diabete[diabete.tension == 0].index.tolist()) # tension n'est pas possible d'être equal à0

print(diabete[diabete.tension == 0].groupby('diabete')['age'].count())
print(diabete[diabete.glucose == 0].shape[0])

print(diabete[diabete.glucose == 0].index.tolist()) # Glucose n'est pas possible d'être equal à0

print(diabete[diabete.glucose == 0].groupby('diabete')['age'].count())
print(diabete[diabete.thickness == 0].shape[0])

print(diabete[diabete.thickness == 0].index.tolist())

print(diabete[diabete.thickness == 0].groupby('diabete')['age'].count())
print(diabete[diabete.bmi == 0].shape[0])

print(diabete[diabete.bmi == 0].index.tolist())

print(diabete[diabete.bmi == 0].groupby('diabete')['age'].count())
data = diabete[(diabete.tension != 0) & (diabete.bmi != 0) & (diabete.glucose!= 0)]
fig = sns.FacetGrid(data, hue="diabete", aspect=3, palette="Set2") # aspect=3 permet d'allonger le graphique

fig.map(sns.kdeplot, "n_pregnant", shade=True)

fig.add_legend()
fig = sns.FacetGrid(data, hue="diabete", aspect=3, palette="Set2") # aspect=3 permet d'allonger le graphique

fig.map(sns.kdeplot, "tension", shade=True)

fig.add_legend()
fig = sns.FacetGrid(data, hue="diabete", aspect=3, palette="Set2") # aspect=3 permet d'allonger le graphique

fig.map(sns.kdeplot, "glucose", shade=True)

fig.add_legend()
fig = sns.FacetGrid(data, hue="diabete", aspect=3, palette="Set2") # aspect=3 permet d'allonger le graphique

fig.map(sns.kdeplot, "thickness", shade=True)

fig.add_legend()
fig = sns.FacetGrid(data, hue="diabete", aspect=3, palette="Set2") # aspect=3 permet d'allonger le graphique

fig.map(sns.kdeplot, "bmi", shade=True)

fig.add_legend()
fig = sns.FacetGrid(data, hue="diabete", aspect=3, palette="Set2") # aspect=3 permet d'allonger le graphique

fig.map(sns.kdeplot, "insulin", shade=True)

fig.add_legend()
X = data.drop(['diabete'], axis=1) # sauf cible

y = data.diabete
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
print(X_train.shape)

print(X_test.shape)
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import classification_report, confusion_matrix, roc_curve, roc_auc_score,auc, accuracy_score
lr = LogisticRegression()

lr.fit(X_train,y_train)
y_lr = lr.predict(X_test)
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
import xgboost as XGB

xgb  = XGB.XGBClassifier()

xgb.fit(X_train, y_train)

y_xgb = xgb.predict(X_test)

cm = confusion_matrix(y_test, y_xgb)

print(cm)

print(classification_report(y_test, y_xgb))
from sklearn.svm import SVC, LinearSVC



linear_svc = LinearSVC()

linear_svc.fit(X_train, y_train)

Y_pred = linear_svc.predict(X_test)

print(accuracy_score(y_test,Y_pred))
from sklearn.linear_model import SGDClassifier



sgd = SGDClassifier()

sgd.fit(X_train, y_train)

Y_pred = sgd.predict(X_test)

print(accuracy_score(y_test,Y_pred))
from sklearn.kernel_approximation import RBFSampler



rbf = RBFSampler(random_state=1)

X_features=rbf.fit_transform(X_test)

# clf = SGDClassifier()

sgd.fit(X_features,y_test)

print(sgd.score(X_features,y_test))
from sklearn.neighbors import KNeighborsClassifier



knn = KNeighborsClassifier(n_neighbors = 3)

knn.fit(X_train, y_train)

Y_pred = knn.predict(X_test)

print(accuracy_score(y_test,Y_pred))