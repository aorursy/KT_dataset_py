# Directive pour afficher les graphiques dans Jupyter

%matplotlib inline
import pandas as pd

import numpy as np

from matplotlib import pyplot as plt

import seaborn as sns
iris = sns.load_dataset("iris")
iris.head(20)
iris.describe()
iris.species.value_counts()
sns.jointplot("sepal_length", "sepal_width", iris, kind='kde');
sns.boxplot(x="sepal_length", y="sepal_width", data=iris)
sns.boxplot(x="petal_length", y="petal_width", data=iris)
fig = sns.FacetGrid(iris, hue="species", aspect=3, palette="Set2") # aspect=3 permet d'allonger le graphique

fig.map(sns.kdeplot, "sepal_length", shade=True)

fig.add_legend()
sns.lmplot(x="sepal_length", y="sepal_width", data=iris, fit_reg=False, hue='species')
sns.lmplot(x="petal_length", y="petal_width", data=iris, fit_reg=False, hue='species')
# Division Area

iris['area_sepal'] = iris.apply(lambda x: x['sepal_length']*x['sepal_width'], axis=1)

iris['area_petal'] = iris.apply(lambda x: x['petal_length']*x['petal_width'], axis=1)

sns.lmplot(x="area_sepal", y="area_petal", data=iris, fit_reg=False, hue='species')
iris['classe'] = iris.species.map({"setosa":0, "versicolor":1, "virginica":2}) # Créer une nouvelle colonne pour déstinguer les spécies en utilisant booléan
iris.describe()
iris = iris.drop(['species'], axis=1)
iris.head(100)
data_train = iris.sample(frac=0.8)          # 80% des données avec frac=0.8

data_test = iris.drop(data_train.index)
X_train = data_train.drop(['classe'], axis=1)

y_train = data_train.classe

X_test = data_test.drop(['classe'], axis=1)

y_test = data_test.classe
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()

lr.fit(X_train,y_train)
y_lr = lr.predict(X_test)
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
lr_score = accuracy_score(y_test, y_lr)

print(lr_score)
cm = confusion_matrix(y_test, y_lr)

print(cm)
print(classification_report(y_test, y_lr))
from sklearn.linear_model import Perceptron
per = Perceptron()

per.fit(X_train,y_train)

y_per = per.predict(X_test)
per_score = accuracy_score(y_test, y_per)

print("Pertinence : ")

print(per_score)

print()

print("Matrice de confusion :")

print(confusion_matrix(y_test, y_per))

print()

print("Rapport de classification :")

print(classification_report(y_test, y_per))
from sklearn import tree

dtc = tree.DecisionTreeClassifier()

dtc.fit(X_train,y_train)

y_dtc = dtc.predict(X_test)

print(accuracy_score(y_test, y_dtc))
dtc1 = tree.DecisionTreeClassifier(max_depth = 4, min_samples_leaf = 30)

dtc1.fit(X_train,y_train)
y_dtc1 = dtc1.predict(X_test)

print(accuracy_score(y_test, y_dtc1))
from sklearn import ensemble

rf = ensemble.RandomForestClassifier()

rf.fit(X_train, y_train)

y_rf = rf.predict(X_test)
rf_score = accuracy_score(y_test, y_rf)

print(rf_score)
pd.crosstab(y_test, y_rf, rownames=['Reel'], colnames=['Prediction'], margins=True)
importances = rf.feature_importances_

indices = np.argsort(importances)
plt.figure(figsize=(12,8))

plt.barh(range(len(indices)), importances[indices], color='b', align='center')

plt.yticks(range(len(indices)), iris.columns[indices])

plt.title('Importance des caracteristiques')
from xgboost import XGBClassifier

from xgboost import plot_importance
xg = XGBClassifier(learning_rate=0.01,

                      n_estimators=10,         

                      max_depth=4,              

                      min_child_weight = 1,      

                      gamma=0.,                  

                      subsample=1,           

                      colsample_btree=1,      

                      scale_pos_weight=1,     

                      random_state=27,           

                      slient = 0

                      )
xg.fit(X_train,y_train)
y_xg=xg.predict(X_train)
xg_score = accuracy_score(y_train, y_xg)

print(xg_score)
cm = confusion_matrix(y_train, y_xg)

print(cm)
y_xg=xg.predict(X_test)
xg_score = accuracy_score(y_test, y_xg)

print(xg_score)
cm = confusion_matrix(y_test, y_xg)

print(cm)