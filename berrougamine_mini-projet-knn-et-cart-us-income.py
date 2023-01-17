import numpy as np

import pandas as pd

import matplotlib as mpl

import matplotlib.pyplot as plt 

import seaborn as sns

from pandas.plotting import scatter_matrix

from sklearn import model_selection

from sklearn.metrics import classification_report

from sklearn.metrics import confusion_matrix

from sklearn.metrics import accuracy_score

from sklearn.tree import DecisionTreeClassifier

from sklearn.neighbors import KNeighborsClassifier

from collections import Counter

import warnings#ignorer les alertes

warnings.filterwarnings('ignore')
#Base donnée de l'ensemble d'apprentissage 

train = pd.read_csv("../input/train.csv")



#Base donnée de l'ensemble de test

test = pd.read_csv("../input/test.csv")
#Description des données d'apprentissage

train.describe()
#Informations sur les données

train.info()
#Affichage d'un apercu des 10 instances de l'ensemble d'apprentissage

train.head(10)
#Informations sur l'ensemble de test

test.info()
#Description des données par attributs

test.describe()
#Affichage d'un apercu des 10 instances de l'ensemble de test

test.head(10)
#Pour voir la corrélation qui existe entre variable classe Income et les autres attributs il faut distinguer les 2 types <50k et >50k

#Donner la valeurs 1 si revenue>50k et 0 si revenue<50k

train['income'] = train['income'].apply(lambda x: 1 if x==' >50K' else 0)

test['income'] = test['income'].apply(lambda x: 1 if x==' >50K' else 0)
#Détection d'éventuelles corrélations 

g = sns.heatmap(train[["income","age","fnlwgt","educational-num","capital-loss","capital-gain","hours-per-week"]].corr(),annot=True,fmt = ".2f", cmap = "coolwarm")
#Histogramme du revenue

plt.hist(train['income']);
#Histogramme de l'age

plt.hist(train['age']);
#Distribution de l'age par la class Income

g = sns.FacetGrid(train, col='income')

g = g.map(sns.distplot, "age")
g = sns.factorplot(x="income", y="age", data=train,size=6, kind="bar", palette="muted")

g.despine(left=True)

g = g.set_ylabels("Distribution de l'age par revenue")
g = sns.factorplot(x="income", y="capital-gain", data=train,size=6, kind="bar", palette="muted")

g.despine(left=True)

g = g.set_ylabels("Distribution du gain par revenue")
g = sns.factorplot(x="income", y="hours-per-week", data=train,size=6, kind="bar", palette="muted")

g.despine(left=True)

g = g.set_ylabels("Distribution des heures du travail par revenue")
sns.boxplot(x="income",y="age", data=train, palette="Set3")
sns.boxplot(x="income",y="educational-num", data=train, palette="Set3")
sns.boxplot(x="income",y="hours-per-week", data=train, palette="Set3")
ax = sns.scatterplot(x="income", y="age", data=train)
ax = sns.scatterplot(x="income", y="hours-per-week", data=train)
#il faut tout d'abord remplacer les valeurs manquantes '?' par 'Nan'

train.replace(' ?', np.nan, inplace=True)

test.replace(' ?', np.nan, inplace=True)
#Affichage des valeurs manquantes pour l'ensemble d'apprentissage

train.isnull().sum()
#affichage des valeurs manquantes pour l'ensemble de test

test.isnull().sum()
print('Procentage des Valeurs manquantes dans "workclass" est %.2f%%' %((train['workclass'].isnull().sum()/train.shape[0])*100))

train["workclass"].isnull().sum()
#remplacer les valeurs manquantes par '0' dans données test et train

train['workclass'].fillna(' 0', inplace=True)

test['workclass'].fillna(' 0', inplace=True)
#on voit si ces valeurs vont affecter notre prédiction de salaire 'income'

sns.factorplot(x="workclass", y="income", data=train, kind="bar", size = 6, palette = "muted")

plt.xticks(rotation=45);
train['workclass'].value_counts()
#Comme on peut voir que without pay et never worked se voient similaire on peut les remplacer 

train['workclass'].replace(' Without-pay', ' Never-worked', inplace=True)

test['workclass'].replace(' Without-pay', ' Never-worked', inplace=True)
#remplacer les valeurs manquantes par 0

train['occupation'].fillna(' 0', inplace=True)

test['occupation'].fillna(' 0', inplace=True)
#visualtion des valeurs manquantes et leur impact sur la prédiction de la class 'income'

sns.factorplot(x="occupation",y="income",data=train,kind="bar", size = 8, palette = "muted")

plt.xticks(rotation=60);
train['occupation'].value_counts()
train['occupation'].replace(' Armed-Forces', ' 0', inplace=True)

test['occupation'].replace(' Armed-Forces', ' 0', inplace=True)
#visualtion des valeurs manquantes et leur impact sur la prédiction de la class 'income'

sns.factorplot(x="occupation",y="income",data=train,kind="bar", size = 8, palette = "muted")

plt.xticks(rotation=60);
#Meme traitement remplacer les valeurs manquantes par '0'

train['native-country'].fillna(' 0', inplace=True)

test['native-country'].fillna(' 0', inplace=True)
sns.factorplot(x="native-country",y="income",data=train,kind="bar", size = 10, 

palette = "muted")

plt.xticks(rotation=80);
#def d'une fonction de rassemblage des pays 

def native(country):

    if country in [' United-States', ' Cuba', ' 0']:

        return 'Amerique'

    elif country in [' England', ' Germany', ' Canada', ' Italy', ' France', ' Greece', ' Philippines']:

        return 'Ouest'

    elif country in [' Mexico', ' Puerto-Rico', ' Honduras', ' Jamaica', ' Columbia', ' Laos', ' Portugal', ' Haiti',

                     ' Dominican-Republic', ' El-Salvador', ' Guatemala', ' Peru', 

                     ' Trinadad&Tobago', ' Outlying-US(Guam-USVI-etc)', ' Nicaragua', ' Vietnam', ' Holand-Netherlands' ]:

        return 'Sud-Amerique' 

    elif country in [' India', ' Iran', ' Cambodia', ' Taiwan', ' Japan', ' Yugoslavia', ' China', ' Hong']:

        return 'Est'

    elif country in [' South', ' Poland', ' Ireland', ' Hungary', ' Scotland', ' Thailand', ' Ecuador']:

        return 'Scandinavie'

    

    else: 

        return country  
#Application de la fonction 

train['native-country'] = train['native-country'].apply(native)

test['native-country'] = test['native-country'].apply(native)
#Affichage du résultat après application de fonction

train['native-country'].value_counts()
sns.factorplot(x="native-country",y="income",data=train,kind="bar", size = 5, 

palette = "muted")

plt.xticks(rotation=60);
train['marital-status'].value_counts()
#Maried spouse absent et af spouse similaire on peut les rassembler

train['marital-status'].replace(' Married-AF-spouse', ' Married-civ-spouse', inplace=True)

test['marital-status'].replace(' Married-AF-spouse', ' Married-civ-spouse', inplace=True)
sns.factorplot(x="marital-status",y="income",data=train,kind="bar", size = 6, 

palette = "muted")

plt.xticks(rotation=60);
sns.factorplot(x="education",y="income",data=train,kind="bar", size = 7, 

palette = "muted")

plt.xticks(rotation=60);
#Def de la fonction de fusionnement

def primary(x):

    if x in [' 1st-4th', ' 5th-6th', ' 7th-8th', ' 9th', ' 10th', ' 11th', ' 12th']:

        return ' Primary'

    else:

        return x
#Application de la fonction 

train['education'] = train['education'].apply(primary)

test['education'] = test['education'].apply(primary)
sns.factorplot(x="education",y="income",data=train,kind="bar", size = 6, 

palette = "muted")

plt.xticks(rotation=60);
#On peut voir que la distrivution des valeurs est trop grande 

g = sns.factorplot(y="fnlwgt",x="income",data=train,kind="box")
train['fnlwgt'].describe()
#Application du Logarithm des valeurs

train['fnlwgt'] = train['fnlwgt'].apply(lambda x: np.log1p(x))

test['fnlwgt'] = test['fnlwgt'].apply(lambda x: np.log1p(x))
train['fnlwgt'].describe()
#résultat de l'application

g = sns.factorplot(y="fnlwgt",x="income",data=train,kind="box")
#Utilisation de la méthode get.dummies(), pour cela il faut fusionner train et test en une seul dataset 

dataset = pd.concat([train, test], axis=0)
#Types des attributs pour lister les attributs catégorique dont dtypes=Object

dataset.dtypes
#Liste des attributs catégoriques

categorical_features = dataset.select_dtypes(include=['object']).axes[1]



for col in categorical_features:

    print (col, dataset[col].nunique())
#Application de la fonction get.dummies()

for col in categorical_features:

    dataset = pd.concat([dataset, pd.get_dummies(dataset[col], prefix=col, prefix_sep=':')], axis=1)

    dataset.drop(col, axis=1, inplace=True)
#Résultat de l'application 

dataset.head()
train = dataset.head(train.shape[0])

test = dataset.tail(test.shape[0])
#X_all: toutes les entités sauf la valeur que nous voulons prédire (income).

#y_all: Seule la valeur que nous voulons prédire.

#on va utiliser train-test-split de la biblio sklearn

#Dans ce cas, on forme 80% des données, puis on teste les 20% restants.

from sklearn.model_selection import train_test_split

X_all = train.drop(['income'],axis=1)

Y_all = train['income']

num_test = 0.20 #20% pour test

X_train, X_test, Y_train, Y_test = train_test_split(X_all, Y_all, test_size=num_test, random_state=23)
#1ième modèle CART



from sklearn.metrics import make_scorer, accuracy_score

from sklearn.model_selection import GridSearchCV

# Choose the type of classifier.

CART = DecisionTreeClassifier()

# Choose some parameter combinations to try

parameters = {'max_depth': [1, 2, 3, 4, 5],

'max_features': [1, 2, 3, 4]}



# Type of scoring used to compare parameter combinations

acc_scorer = make_scorer(accuracy_score)

# Run the grid search

grid_obj = GridSearchCV(CART, parameters, scoring=acc_scorer)

grid_obj = grid_obj.fit(X_train, Y_train)

# Set the clf to the best combination of parameters

clf = grid_obj.best_estimator_

# Fit the best algorithm to the data.

clf.fit(X_train, Y_train)
#calcul mesure de performance sur l'ensemble de validation 2-Cart



predictions = clf.predict(X_test)

print(accuracy_score(Y_test, predictions))
#2ier modèle KNN

#La performance du modèle est estimée en calculant son exactitude (Accuracy).

from sklearn.metrics import make_scorer, accuracy_score

from sklearn.model_selection import GridSearchCV

# Choose the type of classifier.

knn = KNeighborsClassifier()

# Choose some parameter combinations to try

parameters = {'n_neighbors':[1,3, 5, 7],

'weights':['uniform'],

'algorithm':['auto'],

'leaf_size':[30]

}



# Type of scoring used to compare parameter combinations

acc_scorer = make_scorer(accuracy_score)

# Run the grid search

grid_obj = GridSearchCV(knn, parameters, scoring=acc_scorer)

grid_obj = grid_obj.fit(X_train, Y_train)

# Set the clf to the best combination of parameters

clf = grid_obj.best_estimator_

# Fit the best algorithm to the data.

clf.fit(X_train, Y_train)
#calcul mesure de performance sur l'ensemble de validation Knn

predictions = clf.predict(X_test)

print(accuracy_score(Y_test, predictions))
#Exactitude



predictions = clf.predict(X_test)

print(accuracy_score(Y_test, predictions))
#Matrice de Confusion



predictions = clf.predict(X_test)

print(confusion_matrix(Y_test, predictions))
#Précision+Rappel+F-score=classification_report



predictions = clf.predict(X_test)

print(classification_report(Y_test, predictions))