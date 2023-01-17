# Directive pour afficher les graphiques dans Jupyter

%matplotlib inline
# Pandas : librairie de manipulation de données

# NumPy : librairie de calcul scientifique

# MatPlotLib : librairie de visualisation et graphiques

# SeaBorn : librairie de graphiques avancés

import pandas as pd

import numpy as np

from matplotlib import pyplot as plt

import seaborn as sns
# Lecture des données d'apprentissage et de test

t = pd.read_csv("../input/titanic/train.csv")
t.head().T
t.Sex.value_counts()      # nombre d'hommes et de femmes
t.Sex.count()              # nombre total hommes+femmes
t.Cabin.count()
t.count()                  # Comptage par colonnes
t[np.isnan(t.Age)].Survived.value_counts()
hommes = (t.Sex=="male")
t[hommes].head()        # t[hommes] est le tableau où on ne retient que lignes pour lesquelles hommes est True
t[hommes].Survived.value_counts()
femmes = t.Sex=="female"

classe1 = t.Pclass == 1

classe2 = t.Pclass == 2

classe3 = t.Pclass == 3

survivant = t.Survived == 1

mort = ~ survivant
jack = hommes & classe3

rose = femmes & classe1
p_jack = t[jack & survivant].Sex.count()/t[jack].Sex.count()

print(p_jack)
p_rose = t[rose & survivant].Sex.count()/t[rose].Sex.count()

print(p_rose)
t.head()
t.columns
t.describe()
fig = sns.FacetGrid(t, hue="Survived", aspect=5, palette="Set2")

fig.map(sns.kdeplot, "Fare", shade=True)

fig.add_legend()
sns.jointplot("Age", "Pclass", t, kind='kde');
sns.lmplot(x="Age", y="Pclass", data=t, fit_reg=False, hue='Survived')
sns.boxplot("Pclass", "Age", data=t)
t.columns
# On élimine les colonnes non pertinentes pour la prédiction

titanic = t.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)
titanic.count()
titanic[np.isnan(titanic.Age)]
titanic1 = titanic.fillna(value = {'Age':titanic.Age.mean()})
plt.hist(titanic1.Age, bins=80)
titanic = titanic.fillna(method='pad')
titanic = titanic.fillna(method='pad')
titanic.count()
plt.hist(titanic.Age, bins=80)
sns.distplot(titanic.Fare, color='blue')
titanic['log_fare'] = np.log(titanic.Fare+1)
sns.kdeplot(titanic.log_fare, color='blue')
titanic = titanic.drop(['Fare'], axis=1)
titanic[['Age','log_fare']].describe()
sns.kdeplot(titanic.log_fare, color='blue')

sns.kdeplot(titanic.Age, color='red')
from sklearn import preprocessing
minmax = preprocessing.MinMaxScaler(feature_range=(0, 1))

titanic[['Age', 'log_fare']] = minmax.fit_transform(titanic[['Age', 'log_fare']])
sns.distplot(titanic.log_fare, color='blue')

sns.distplot(titanic.Age, color='red')
scaler = preprocessing.StandardScaler()

titanic[['Age', 'log_fare']] = scaler.fit_transform(titanic[['Age', 'log_fare']])
sns.kdeplot(titanic.log_fare, color='blue')

sns.kdeplot(titanic.Age, color='red')
titanic.info()
titanic.Sex = titanic.Sex.map({"male":0, "female":1})
titanic.head()
titanic = pd.get_dummies(data=titanic, columns=['Pclass', 'Embarked'])
titanic.head()
X = titanic.drop(['Survived'], axis=1)

y = titanic.Survived
from sklearn.model_selection import train_test_split



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
print(X_train.shape)

print(X_test.shape)
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()

lr.fit(X_train,y_train)
y_lr = lr.predict(X_test)
# Importation des méthodes de mesure de performances

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
from sklearn.model_selection import validation_curve

params = np.arange(1, 300,step=30)

train_score, val_score = validation_curve(rf, X, y, 'min_samples_leaf', params, cv=7)

plt.figure(figsize=(12,12))

plt.plot(params, np.median(train_score, 1), color='blue', label='training score')

plt.plot(params, np.median(val_score, 1), color='red', label='validation score')

plt.legend(loc='best')

plt.ylim(0, 1)

plt.xlabel('min_samples_leaf')

plt.ylabel('score');
from sklearn.model_selection import validation_curve

params = np.arange(1, params/dfprobas.count(),step=30)

train_score, val_score = validation_curve(rf, X, y, 'max_features', params, cv=7)

plt.figure(figsize=(12,12))

plt.plot(params, np.median(train_score, 1), color='blue', label='training score')

plt.plot(params, np.median(val_score, 1), color='red', label='validation score')

plt.legend(loc='best')

plt.ylim(0, 1)

plt.xlabel('max_features')

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
# Sous Jupyter, si xgboost n'est pas déjà installé

!pip install xgboost
import xgboost as XGB

xgb  = XGB.XGBClassifier()

xgb.fit(X_train, y_train)

y_xgb = xgb.predict(X_test)

cm = confusion_matrix(y_test, y_xgb)

print(cm)

print(classification_report(y_test, y_xgb))
# Pandas : librairie de manipulation de données

# NumPy : librairie de calcul scientifique

# MatPlotLib : librairie de visualisation et graphiques

# SeaBorn : librairie de graphiques avancés

import pandas as pd

import numpy as np

from matplotlib import pyplot as plt

import seaborn as sns
# Lecture des données d'apprentissage et de test

d = pd.read_csv("../input/pyms-diabete/diabete.csv")
d.head()
d.count()
d.glucose.value_counts()
d.n_pregnant.value_counts()
d.diabete.value_counts()
#jamaisEnceinte = d.n_pregnant == 0

dejaEnceinte = d.n_pregnant >= 1

#glucosePlusDe100 = d.glucose >= 100



#test_jamaisEnceinte_et_glucocePlusDe100 = jamaisEnceinte & glucosePlusDe100
# p_test_jamaisEnceinte_et_diabetique = d[jamaisEnceinte & d.diabete].age.count()/d.age.count()

# print('Si t\'as jamais été enceinte et diabetique :')

# print(p_test_jamaisEnceinte_et_diabetique)



p_test_dejaEnceinte_et_diabetique = d[dejaEnceinte & d.diabete].age.count()/d.age.count()

print('Si t\'as déjà été enceinte et diabetique :')

print(p_test_dejaEnceinte_et_diabetique)



# p_test_glucocePlusDe100_et_diabetique = d[glucosePlusDe100 & d.diabete].age.count()/d.age.count()

# print('Si tu as un glucose de plus de 100 et diabetique :')

# print(p_test_glucocePlusDe100_et_diabetique)



# p_test_jamaisEnceinte_et_glucocePlusDe100 = d[test_jamaisEnceinte_et_glucocePlusDe100 & d.diabete].age.count()/d.age.count()

# print('Si t\'as jamais été enceinte et tu as un glucose de plus de 100 et diabetique :')

# print(p_test_jamaisEnceinte_et_glucocePlusDe100)

sns.lmplot(x="insulin", y="pedigree", data=d, fit_reg=False, hue='diabete')
sns.jointplot("insulin", "glucose", d, kind='kde');
d.columns
# On élimine les colonnes non pertinentes pour la prédiction

# diabet = d.drop(['age'], axis=1)
sns.distplot(d.n_pregnant, color='blue')
sns.distplot(d.glucose, color='blue')
sns.distplot(d.tension, color='blue')
sns.distplot(d.thickness, color='blue')
sns.distplot(d.insulin, color='blue')
sns.distplot(d.bmi, color='blue')
sns.distplot(d.pedigree, color='blue')
d['log_thickness'] = np.log(d.thickness+1)
sns.distplot(d.log_thickness, color='blue')
d['log_pedigree'] = np.log(d.pedigree+1)
sns.distplot(d.log_pedigree, color='blue')
d = d.drop(['pedigree', 'thickness'], axis=1)
d.count()
sns.kdeplot(d.n_pregnant, color="blue")

sns.kdeplot(d.bmi, color="red")
d.info()
X = d.drop(['diabete'], axis=1)

y = d.diabete
from sklearn.model_selection import train_test_split



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
print(X_train.shape)

print(X_test.shape)
from sklearn.linear_model import LogisticRegression



lr = LogisticRegression()

lr.fit(X_train,y_train)
y_lr = lr.predict(X_test)
# Importation des méthodes de mesure de performances

from sklearn.metrics import classification_report, confusion_matrix, roc_curve, roc_auc_score,auc, accuracy_score

print(confusion_matrix(y_test,y_lr))
print(accuracy_score(y_test,y_lr))
#RandomForest
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