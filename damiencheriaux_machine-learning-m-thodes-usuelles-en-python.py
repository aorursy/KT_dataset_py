## Initialisation ##

# Import des librairies

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

from math import *

import statsmodels.api as sm

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC

from sklearn.ensemble import RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.linear_model import Perceptron

from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import roc_auc_score, roc_curve



# Import des données

train = pd.read_csv("../input/titanic/train.csv", sep=",")

test = pd.read_csv("../input/titanic/test.csv", sep=",")



# Définition des fonctions

def plot_ROC(y_train,proba_train, legende = '', color = 'blue'):

    fpr_train, tpr_train, _ = roc_curve(y_train, proba_train,drop_intermediate=False)

    auc_train = round(roc_auc_score(y_train, proba_train),2)

    plt.plot(fpr_train, tpr_train, label=legende + " AUC = " + str(auc_train),color=color)

    plt.xlim(0,1)

    plt.ylim(0,1)

    plt.legend(loc=0)



# Modification des paramètres d'affichage de Python

desired_width=320

pd.set_option('display.width', desired_width)

np.set_printoptions(linewidth=desired_width)

pd.set_option('display.max_columns',20)
print(train.columns)

    # PassengerID : Num : Identifiant du passager

    # Survival    : Num : Le passager à survécu ou non : 0 = Non, 1 = Oui

    # Pclass      : Num : Classe du ticket :	1 = 1ere classe, 2 = Seconde classe, 3 = 3ieme classe

    # Name        : Txt : Nom du passager

    # Sex	      : Txt : Sexe

    # Age	      : Num : Age en année

    # Sibsp       : Num : Nombre de relations horizontales (frères, soeurs, époux...)

    # Parch       : Num : Nombre de relations verticales (parents, enfants...)

    # Ticket      : Txt : Numéro du billet

    # Fare        : Num : Tarif du billet

    # Cabin       : Txt : Numéro de cabine

    # Embarked    : Txt : Port d'embarcation :	C = Cherbourg, Q = Queenstown, S = Southampton



print("\n ------------------------------------------------------------ \n")

print(train.info())

    # On remarque des valeurs manquantes pour plusieurs variables :

        # Age      : 177 NA

        # Cabin    : 687 NA

        # Embarked : 2 NA



print("\n ------------------------------------------------------------ \n")

print(test.info())

    # On remarque des valeurs manquantes pour plusieurs variables :

        # Age   : 86 NA

        # Cabin : 327 NA

        # Fare  : 1 NA

        

print("\n ------------------------------------------------------------ \n")

print(train.describe())

    # L'échantillon de train contient les informations de 891 passagers, ce qui représente 40% du nombre total de passagers à bord du Titanic. Le taux de survie est de 38%, ce qui se rapproche du taux de survie reel à bord du Titanic. L'échantillon semble représentatif. L'âge moyen est de 29 ans, ce qui est plutôt jeune. Au moins 75% des passagers ont moins de 38 ans et 25% ont moins de 20 ans. Il y a donc une forte concentration de passagers dans la tranche d'age 20-38 ans (la moitié)Plus de la moitié des passagers étaient en 3ième classe. Au moins 75% des passagers voyageaient sans enfants et/ou parents. Il y a une grosse disparité dans les prix des billets. Au moins 75% des passagers ont payé leur ticket 31€ malgré un prix maximum de 512€.



print("\n ------------------------------------------------------------ \n")

print(train.describe(include=['O']))

    # Il n'y a pas de doublons sur les noms des passagers. Il y avait en majorité des hommes à bord dans notre échantillon (577 sur 891 passagers, soit 65%) Il y a des doublons dans les tickets, les cabines et les quais d'embarquements. Il y a donc des passagers qui partageaient la même cabine.
g = sns.FacetGrid(train, col='Survived')

g.map(plt.hist, 'Age', bins=20)

    # On constate que les enfants (de moins de 4 ans) ont un fort taux de survie. De la même manière, les personnes agées ont survécue. Il semblerait que les passagers ont favorisé l'évacuation des personnes plus fragiles en priorité. En revanche, les 15-35 ans ont un faible  taux de survie par rapport au nombre de passagers de cette tranche.



grid = sns.FacetGrid(train, col='Survived', row='Pclass', size=2.2, aspect=1.6)

grid.map(plt.hist, 'Age', alpha=.5, bins=20)

grid.add_legend()

    # La 3ième classe est la plus représentée mais également celle qui a le moins survécu. Les enfants de la 2ième classe sont très peu nombreux mais ont un fort taux de survie. Les passagers de la 1ère classe sont les moins représentés mais également ceux qui ont le plus survécu. Les classes contiennent une distribution bien répartie sur l'âge.



grid = sns.FacetGrid(train, row='Embarked', size=2.2, aspect=1.6)

grid.map(sns.pointplot, 'Pclass', 'Survived', 'Sex', palette='deep')

grid.add_legend()

    # En général les femmes ont un meilleur taux de survie que les hommes (à l'exception du port d'embarquement C) Les 1ère et 2ième classes ont un meilleur taux de survie que la 3ième classe.



grid = sns.FacetGrid(train, row='Embarked', col='Survived', size=2.2, aspect=1.6)

grid.map(sns.barplot, 'Sex', 'Fare', alpha=.5, ci=None)

grid.add_legend()

    # Les passagers ayant payé leur ticket plus chère ont un plus fort taux de survie. On remarque des différences de prix des billets entre les différents ports d'embarquement.
## Transformation des données ##



# Suppression des variables Ticket et Cabin

train = train.drop(['Ticket', 'Cabin'], axis=1)

test = test.drop(['Ticket', 'Cabin'], axis=1)

    # Dans un premier temps on va supprimer les variables Ticket et Cabin car elles contiennent trop de valeurs manquantes et de doublons.



DATA = [train, test]

    # On rassemble les deux échantillons de données dans une liste pour appliquer les transformations une seule fois dans une boucle.
# Extraction du Titre 

for echantillon in DATA:

    echantillon['Titre'] = echantillon.Name.str.extract(' ([A-Za-z]+)\.', expand=False)

    # Extraction du titre dans le nom complet des passagers.

    

pd.crosstab(train['Titre'], train['Sex'])

    # On regarde la distribution des sexes pour chacun des titres.



for echantillon in DATA:

    echantillon['Titre'] = echantillon['Titre'].replace(['Lady', 'Countess', 'Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer'

                                                         , 'Dona'], 'Autre')

    echantillon['Titre'] = echantillon['Titre'].replace('Mlle', 'Miss')

    echantillon['Titre'] = echantillon['Titre'].replace('Ms', 'Miss')

    echantillon['Titre'] = echantillon['Titre'].replace('Mme', 'Mrs')

    # On rassembles certains titres pour déduire le nombre de classes. Certains titres comptaient que très peu de personnes. Il n'est donc

    # pas pertinent de les conserver ainsi.

    

train[['Titre', 'Survived']].groupby(['Titre'], as_index=False).mean().sort_values(['Survived'], ascending = False)

    # On retrouve ici encore un taux de survie plus fort chez les femme. S'en suit ensuite les hommes avec des titres plus importants.



Titre_Num = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}

for echantillon in DATA:

    echantillon['Titre'] = echantillon['Titre'].map(Titre_Num)

    echantillon['Titre'] = echantillon['Titre'].fillna(0)

    echantillon['Titre'] = echantillon['Titre'].astype(int)

    # On remplace ensuite les données textes par des données numérique pour que les modèles aient une meilleur interprétation. La classe 0 est une classe spécifique pour les passagers n'ayant pas de Titre.



print(train.head())



train = train.drop(['Name', 'PassengerId'], axis=1)

test = test.drop(['Name'], axis=1)

DATA = [train, test]

    # On peut à présent supprimer la variable Name. On en profite pour supprimer la variable PassengerID de l'échantillon de train car elle n'est pas utile. Il faut toutefois conserver cette variable dans le test pour les soumission au Kaggle.
# Transformation en numérique



for echantillon in DATA:

    echantillon['Sex'] = echantillon['Sex'].map( {'female': 0, 'male': 1} ).astype(int)

    # Conversion de la variable Sex en catégories numériques



# La variable Embarked contient 2 valeurs manquantes. Nous allons les compléter par la modalité la plus présente dans l'échantillon train.

freq_Embarked = train.Embarked.dropna().mode()[0]

freq_Embarked

    # Le port d'embarquement le plus présent dans l'échantillon train est S : Southampton

for echantillon in DATA:

    echantillon['Embarked'] = echantillon['Embarked'].fillna(freq_Embarked)

    echantillon['Embarked'] = echantillon['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)

    # On convertie ensuite la variable Embarked en variable catégorielle numérique.

print(train.head())
# Etude des corrélations 



    # A présent que toutes les variables sont au format numérique, nous pouvons regarder de plus près les corrélations.

colormap = plt.cm.RdBu

plt.figure(figsize=(14,12))

plt.title('orrélations de Pearson', y=1.05, size=15)

sns.heatmap(train.astype(float).corr(),linewidths=0.1,vmax=1.5, square=True, cmap=colormap, linecolor='white', annot=True)

    # On observe ainsi une forte corrélation entre le sexe et le Titre. Ce qui est cohérent avec l'hypothèse posée précédemment. De la même manière, le prix du billet est fortement corrélé à la classe de celui-ci. Encore une fois, ce résultat est cohérent. On observe également une forte corrélation entre le sexe et la survie ou non du passager. Ceci vérifie notre conjecture établie au début de l'analyse. On remarque une corrélation entre les variables Age, Pclass et Titre. 
# Nous allons donc utiliser ces corrélations pour completer les valeurs manquantes de la variable Age. Toutefois, afin de simplifier les choses, nous allons utiliser la variable Sexe qui contient moins de modalités que la variable Titre. Ces deux variables étant dortement corrélées.



Mat_age = np.zeros((2,3))

    # Matrice des ages médians par croisement de Pclass et Sex

for echantillon in DATA:

    for sexe in range(0, 2):

        for classe in range(0, 3):

            df = echantillon[(echantillon['Sex'] == sexe) & (echantillon['Pclass'] == classe + 1)]['Age'].dropna()

            Mat_age[sexe, classe] = df.median()

    for sexe in range(0, 2):

        for classe in range(0, 3):

            echantillon.loc[(echantillon.Age.isnull()) & (echantillon.Sex == sexe) & (echantillon.Pclass == classe + 1), 'Age'] = Mat_age[sexe, classe]

    echantillon['Age'] = echantillon['Age'].astype(int)



    # A présent on va convertire la variable Age en variable catégorielle numérique en créant des classes d'age. Les modèles seront plus performant de cette manière.

plt.hist(train.Age, density = True)

for echantillon in DATA:

    echantillon.loc[ echantillon['Age'] <= 18, 'Age'] = 0

    echantillon.loc[(echantillon['Age'] > 18) & (echantillon['Age'] <= 25), 'Age'] = 1

    echantillon.loc[(echantillon['Age'] > 25) & (echantillon['Age'] <= 32), 'Age'] = 2

    echantillon.loc[(echantillon['Age'] > 32) & (echantillon['Age'] <= 48), 'Age'] = 3

    echantillon.loc[echantillon['Age'] > 48 , 'Age'] = 4

train[['Age','Survived']].groupby('Age').count()

    # La répartition des passagers par tranche d'âge n'est pas déséquilibrée. Nous pouvons donc faire un premier test avec 5 tranches d'âge.



# On remplace la valeur manquante du prix du billet dans l'échantillon test par la médiane de la variable.

test['Fare'].fillna(test['Fare'].dropna().median(), inplace=True)

print(test.info())



# On va a convertire la variable Fare en variable catégorielle en créant des classes de prix.

print("\n ------------------------------------------------------------ \n")

print(train.Fare.describe())

    # Nous allons créer des classes au niveau des quartiles. Nous allons également isoler les 10% des billets les plus chère pour voir si il y a une différence de survie entre les billets les plus chères et les autres.

for echantillon in DATA:

    echantillon.loc[echantillon['Fare'] <= 7.91, 'Fare'] = 0

    echantillon.loc[(echantillon['Fare'] > 7.91) & (echantillon['Fare'] <= 14.454), 'Fare'] = 1

    echantillon.loc[(echantillon['Fare'] > 14.454) & (echantillon['Fare'] <= 31), 'Fare'] = 2

    echantillon.loc[(echantillon['Fare'] > 31) & (echantillon['Fare'] <= 77.9583), 'Fare'] = 3

    echantillon.loc[echantillon['Fare'] > 77.9583, 'Fare'] = 4

    echantillon['Fare'] = echantillon['Fare'].astype(int)

        

print("\n ------------------------------------------------------------ \n")

print(train[['Survived']].groupby(train['Fare']).mean())

    # On remarque que le taux de survie est plus important pour les passagers appartenant à la classe de prix la plus chère. On peut conjecturer que cette classe sera discriminante dans les modèles.

    

            

print("\n ------------------------------------------------------------ \n")

print(train.head())
# Création de nouvelles variables

for echantillon in DATA:

    echantillon['Famille'] = echantillon['SibSp'] + echantillon['Parch']

    # On rassemble les deux variables SibSP et Parch pour connaître la taille de la famille du passager. Si Famille = 0 alors la personne a embarqué seule sur le Titanic.



train[['Famille', 'Survived']].groupby(['Famille'], as_index=False).mean().sort_values(by='Survived', ascending=False)

    # On remarque que les passager ayant embarqué avec 3 personnes ou moins un meilleur taux de survie.

train[['Famille', 'Survived']].groupby(['Famille'], as_index=False).count()

    # On remarque que la répartition par classe est très inégale. Afin de palier à ce problème, nous allons rassembler des classes.



for echantillon in DATA:

    echantillon.loc[echantillon['Famille'] == 0, 'Famille'] = 0

    echantillon.loc[(echantillon['Famille'] > 0) & (echantillon['Famille'] <= 2), 'Famille'] = 1

    echantillon.loc[(echantillon['Famille'] >= 3) , 'Famille'] = 2

    echantillon['Fare'] = echantillon['Fare'].astype(int)

print(train.head())

    # On peut à présent supprimer les variables SibSP et Parch.



train = train.drop(['Parch', 'SibSp'], axis=1)

test = test.drop(['Parch', 'SibSp'], axis=1)

DATA = [train, test]
# Vérifications des échantillons 

    # Maintenant que les données ont été préalablement traitées, nous faisons les dernières vérifications avant la phase de modélisation.

print(train.info())

print("\n ------------------------------------------------------------ \n")



print(train.describe())

print("\n ------------------------------------------------------------ \n")



print(train.head())

print("\n ------------------------------------------------------------ \n")



print(test.info())

print("\n ------------------------------------------------------------ \n")

print(test.describe())

print("\n ------------------------------------------------------------ \n")

print(test.head())

    # Toutes les variables sont catégorielles et numériques. Il n'y a plus de valeurs manquantes
X_train = train.drop("Survived", axis=1)

Y_train = train["Survived"]

X_test  = test.drop("PassengerId", axis=1).copy()

X_train.shape, Y_train.shape, X_test.shape
## Régression Logistique ##

    # Deux possibilités s'offrent à nous pour la regression logistique. La première est la fonction LogisticRegression du package Scikit-Learn. La seconde est la fonction Logit du package Statsmodels.



# Package Scikit Learn : 

    # Lien de l'explication de la fonction sur le package Sk-learn : https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html

reg_sk = LogisticRegression()

reg_sk.fit(X_train, Y_train)

print("Accuracy de la Régression Logistique via Scikit Learn : %s"%(round(reg_sk.score(X_train, Y_train)*100,2)))

print("\n ------------------------------------------------------------ \n")

plot_ROC(Y_train, [x[1] for x in reg_sk.predict_proba(X_train)], 'Logistique Sk')

    # Ce modèle obtient un AUC de 0.87 sur l'échantillon train.

submission = pd.DataFrame({"PassengerId": test["PassengerId"], "Survived": reg_sk.predict(X_test)})

submission.to_csv('Soumission_Logistic_sk.csv', index=False)

    # Ce modèle donne un score que l'échantillon test de 77.99%.



# Avec le Package Statsmodels :

    # Lien de l'explication de la fonction sur le package Statsmodels : https://www.statsmodels.org/stable/generated/statsmodels.discrete.discrete_model.Logit.html

reg_sm = sm.Logit(Y_train, X_train)

reg_sm = reg_sm.fit()

Y_pred_sm = round(reg_sm.predict(X_train), 0)

print("\n ------------------------------------------------------------ \n")

print("Accuracy de la Régression Logistique via Statsmodels : %s"%(round(sum(Y_pred_sm == Y_train)/len(Y_train)*100,2)))

plot_ROC(Y_train, [x for x in reg_sm.predict(X_train)], 'Logistique SM', 'red')

    # Ce modèle obtient un AUC de 0.87 sur l'échantillon train.

print("\n ------------------------------------------------------------ \n")

print(reg_sm.summary2())

    # La variable Age n'est pas significative. Nous allons essayer de la retirer dans le modèle.

    

reg_sm_cust = sm.Logit(Y_train, X_train.drop(['Age'], axis = 1))

reg_sm_cust = reg_sm_cust.fit()

Y_pred_sm_cust = round(reg_sm_cust.predict(X_train.drop(['Age'], axis = 1)), 0)

print("\n ------------------------------------------------------------ \n")

print("Accuracy de la Régression Logistique via Statsmodels sans la variable Age : %s"%(round(sum(Y_pred_sm_cust == Y_train)/len(Y_train)*100,2)))

print("\n ------------------------------------------------------------ \n")

print(reg_sm_cust.summary2())

    # A présent toutes les variables son significatives.

plot_ROC(Y_train, [x for x in reg_sm_cust.predict(X_train.drop(['Age'], axis = 1))], 'Custome SM', 'green')

    # Ce modèle obtient un AUC de 0.87 sur l'échantillon train. L'AUC ne change pas en supprimant la variable Age On remarque toutefois que le modèle perd en performance sur l'échantillon de train via le critère Accuracy. La variable Age contribue donc au modèle malgré sa non significativité. Nous faisons donc le choix de la conserver.



submission = pd.DataFrame({"PassengerId": test["PassengerId"], "Survived": round(reg_sm.predict(X_test),0).astype(int)})

submission.to_csv('Soumission_Logistic_sm.csv', index=False)

    # Ce modèle donne un score que l'échantillon test de 77.99%.
## Arbre de décision ##

    # Lien de l'explication de la fonction sur le package Sk-learn : https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html

arbre = DecisionTreeClassifier()

arbre.fit(X_train, Y_train)

print("Accuracy de l'Arbre de décision : %s"%(round(arbre.score(X_train, Y_train)*100,2)))

plot_ROC(Y_train, [x[1] for x in arbre.predict_proba(X_train)])

    # Ce modèle obtient un AUC de 0.96 sur l'échantillon train.

submission = pd.DataFrame({"PassengerId": test["PassengerId"], "Survived": arbre.predict(X_test)})

submission.to_csv('Soumission_Arbre.csv', index=False)
## Random Forest ##

    # Lien de l'explication de la fonction sur le package Sk-learn : https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html

random_forest = RandomForestClassifier()

random_forest.fit(X_train, Y_train)

print("Accuracy du Random Forest : %s"%(round(random_forest.score(X_train, Y_train)*100,2)))

print("\n ------------------------------------------------------------ \n")

plot_ROC(Y_train, [x[1] for x in random_forest.predict_proba(X_train)], 'Random Forest')

    # Ce modèle obtient un AUC de 0.95 sur l'échantillon train. A première vue, le modèle de random forest semble faire un moins bon score que l'arbre de décision. Toutefois, n'oublions pas que l'arbre de décision était en sur-apprentissage sur nos données. Nous allons vérifier le score du modèle random forest avec les paramètres par défaut sur l'échantillon test.

submission = pd.DataFrame({"PassengerId": test["PassengerId"], "Survived": random_forest.predict(X_test)})

submission.to_csv('Soumission_RF.csv', index=False)

    # Ce modèle donne un score que l'échantillon test de 77.03%. Le modèle obtient donc de meilleures performances sur l'échantillon test. Il est moins en sur-apprentissage que l'arbre de décision. On constate toutefois que le random forest avec les paramètres par défaut ne performe pas la régression logistique sur l'échantillon test. Essayons d'optimiser ses paramètres via un grid-search.



# Mise en place du GRID-SEARCH :

params = {"parametres" : None, "AUC" : 0, "Acc" : 0, "Soumission" : None}

for n_est in [10, 100, 500, 1000] :

    for max_feat in [int(x) for x in np.r_[int(sqrt(X_train.shape[1])):X_train.shape[1]:3j]] :

        for max_dep in [int(x) for x in np.r_[1:int(sqrt(X_train.shape[0])):5j]] :

            RF = RandomForestClassifier(n_estimators=n_est, max_depth=max_dep, max_features=max_feat)

            RF.fit(X_train, Y_train)

            if round(RF.score(X_train, Y_train) * 100, 2) > params['Acc'] and round(roc_auc_score(Y_train, [x[1] for x in RF.predict_proba(X_train)]),2) > params['AUC']:

                params['parametres'] = [n_est, max_feat, max_dep]

                params['AUC'] = round(roc_auc_score(Y_train, [x[1] for x in RF.predict_proba(X_train)]),2)

                params['Acc'] = round(RF.score(X_train, Y_train) * 100, 2)

                params['Soumission'] = pd.DataFrame({"PassengerId": test["PassengerId"], "Survived": RF.predict(X_test)})

                print('Accuracy améliorée : %s'%params['Acc'])



print("\n ------------------------------------------------------------ \n")

print("Le meilleur modèle à l'issue du grid search utilise les paramètres suivants : %s et obtient en score un AUC de %s et une accuracy de %s"%(params['parametres'], params['AUC'], params['Acc']))

params['Soumission'].to_csv('Soumission_RF_cust.csv', index=False)

    # On fait la soumission pour connaître le score sur l'échantillon test. On obtient un score de 77.55% ce qui améliore par rapport au modèle avec les paramètres par défaut. On constate toutefois que le modèle ne performe pas la régression logistique.



## KPPV - Plus proches voisins ##

    # Lien de l'explication de la fonction sur le package Sk-learn : https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html

knn = KNeighborsClassifier()

knn.fit(X_train, Y_train)

print("Accuracy des 5 plus proches voisins : %s"%(round(knn.score(X_train, Y_train)*100,2)))

print("\n ------------------------------------------------------------ \n")

plot_ROC(Y_train, [x[1] for x in knn.predict_proba(X_train)], 'KNN')

    # Ce modèle obtient un AUC de 0.92 sur l'échantillon train.

submission = pd.DataFrame({"PassengerId": test["PassengerId"], "Survived": knn.predict(X_test)})

submission.to_csv('Soumission_Knn1.csv', index=False)

    # Ce modèle donne un score que l'échantillon test de 77.03%.



# Mise en place du GRID-SEARCH :

params = {"n" : None, "AUC" : 0, "Acc" : 0, "Soumission" : None}

for n in [2, 3, 4, 5, 7, 10, 15] :

    knn = KNeighborsClassifier(n_neighbors=n)

    knn.fit(X_train, Y_train)

    if round(knn.score(X_train, Y_train) * 100, 2) >= params['Acc'] and round(roc_auc_score(Y_train, [x[1] for x in knn.predict_proba(X_train)]),2) >= params['AUC']:

        params['n'] = n

        params['AUC'] = round(roc_auc_score(Y_train, [x[1] for x in knn.predict_proba(X_train)]),2)

        params['Acc'] = round(knn.score(X_train, Y_train) * 100, 2)

        params['Soumission'] = pd.DataFrame({"PassengerId": test["PassengerId"], "Survived": knn.predict(X_test)})

        print('Accuracy améliorée : %s'%params['Acc'])



print("\n ------------------------------------------------------------ \n")

print("Le meilleur modèle à l'issue du grid search utilise n = %s et obtient en score un AUC de %s et une Accuracy de %s"%(params['n'], params['AUC'], params['Acc']))

params['Soumission'].to_csv('Soumission_Knn_cust.csv', index=False)

    # Le meilleur modèle de Knn obtient un score de 75.11% sur l'échantillon test, ce qui est inférieur au modèle par défaut qui utilise en paramètre n = 5 voisins. On constat donc que notre modèle est en sur-apprentissage.
## Gaussian Naive Bayes ##

    # Lien de l'explication de la fonction sur le package Sk-learn : https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.GaussianNB.html

gaussian = GaussianNB()

gaussian.fit(X_train, Y_train)

print("Accuracy de la Gaussian Naive Bayes : %s"%(round(gaussian.score(X_train, Y_train)*100,2)))

plot_ROC(Y_train, [x[1] for x in gaussian.predict_proba(X_train)], 'Gaussian Naive Bayes')

    # Ce modèle obtient un AUC de 0.84 sur l'échantillon train.

submission = pd.DataFrame({"PassengerId": test["PassengerId"], "Survived": gaussian.predict(X_test)})

submission.to_csv('Soumission_Gaussian.csv', index=False)

    # Ce modèle donne un score que l'échantillon test de 73.20%. Il ne permet donc pas de performer la régression logistique.
## Support Vector Machine ##

    # Lien de l'explication de la fonction sur le package Sk-learn : https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html

svc = SVC(probability=True)

svc.fit(X_train, Y_train)

print("Accuracy du SVM : %s"%(round(svc.score(X_train, Y_train)*100,2)))

print("\n ------------------------------------------------------------ \n")

plot_ROC(Y_train, [x[1] for x in svc.predict_proba(X_train)], 'SVM')

    # Ce modèle obtient un AUC de 0.89 sur l'échantillon train.

submission = pd.DataFrame({"PassengerId": test["PassengerId"], "Survived": svc.predict(X_test)})

submission.to_csv('Soumission_SVM.csv', index=False)

    # Ce modèle obtient un score de 78.94 % sur l'échantillon test. Il performe donc la regression malgré les paramètres par défaut. Nous allons donc essayer d'améliorer ce modèle en le customisant.



# Mise en place du GRID-SEARCH :

params = {"n" : None, "AUC" : 0, "Acc" : 0, "Soumission" : None}

for kernel in ['linear', 'poly', 'rbf', 'sigmoid']:

    for gamma in ['auto', 'scale']:

        for c in [0.5, 0.8, 1, 1.2, 1.5]:

            svc_cust = SVC(probability=True, C = c, gamma = gamma, kernel = kernel)

            svc_cust.fit(X_train, Y_train)

        if round(svc_cust.score(X_train, Y_train) * 100, 2) > params['Acc'] and round(roc_auc_score(Y_train, [x[1] for x in svc_cust.predict_proba(X_train)]),2) > params['AUC']:

            params['parametres'] = [kernel, gamma, c]

            params['AUC'] = round(roc_auc_score(Y_train, [x[1] for x in svc_cust.predict_proba(X_train)]),2)

            params['Acc'] = round(svc_cust.score(X_train, Y_train) * 100, 2)

            params['Soumission'] = pd.DataFrame({"PassengerId": test["PassengerId"], "Survived": svc_cust.predict(X_test)})

            print('Accuracy améliorée : %s'%params['Acc'])

            

print("\n ------------------------------------------------------------ \n")

print("Le meilleur modèle à l'issue du grid search utilise n = %s et obtient en score un AUC de %s et une Accuracy de %s"%(params['parametres'], params['AUC'], params['Acc']))

params['Soumission'].to_csv('Soumission_SVM_cust.csv', index=False)

    # Le score sur l'échantillon test du meilleur modèle SVM est de 79.42%. On améliore donc le score de notre modèle après optimisation des paramètre.
## Perceptron ##

    # Lien de l'explication de la fonction sur le package Sk-learn : https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Perceptron.html

perceptron = Perceptron()

perceptron.fit(X_train, Y_train)

print("Accuracy du Perceptron : %s"%(round(perceptron.score(X_train, Y_train)*100,2)))

print("\n ------------------------------------------------------------ \n")

    # La fonction predict_proba n'est pas disponible pour ce modèle.

submission = pd.DataFrame({"PassengerId": test["PassengerId"], "Survived": perceptron.predict(X_test)})

submission.to_csv('Soumission_Perceptron.csv', index=False)

    # Nous obtenons un score de 79.42%. sur l'échantillon de test avec les paramètres par défaut.



# Mise en place du GRID-SEARCH :

params = {"parametres" : None, "Acc" : 0, "Soumission" : None}

for penality in ['l2','l1','elasticnet']:

    for validation_frac in [0.1, 0.2, 0.3, 0.4]:

        for alpha in [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1]:

            for max_iter in [10,100,1000,5000]:

                for tol in [0.0001, 0.0005, 0.001, 0.005, 0.01]:

                    perceptron = Perceptron(early_stopping= True, penalty=penality,

                                            validation_fraction=validation_frac, max_iter=max_iter, tol=tol)

                    perceptron.fit(X_train, Y_train)

                    if round(perceptron.score(X_train, Y_train) * 100, 2) > params['Acc'] :

                        params['parametres'] = [penality, validation_frac, alpha, max_iter, tol]

                        params['Acc'] = round(perceptron.score(X_train, Y_train) * 100, 2)

                        params['Soumission'] = pd.DataFrame({"PassengerId": test["PassengerId"], "Survived": perceptron.predict(X_test)})

                        print('Accuracy améliorée : %s'%params['Acc'])

                        

print("\n ------------------------------------------------------------ \n")

print("Le meilleur modèle à l'issue du grid search utilise les paramètres : %s et obtient en score et une Accuracy de %s"%(params['parametres'], params['Acc']))

params['Soumission'].to_csv('Soumission_SVM_cust.csv', index=False)

    # Le meilleur modèle après grid-search obtient un score de 75.11% sur l'échantillon test malgré une meilleure accuracy sur l'échantillon train. Il semblerait donc que le modèle soit en surapprentissage.

    
def cut_seuil (proba, seuil):

    if proba > seuil :

        return 1

    else :

        return 0



# Modèle SVM

Proba_SVM = [x[1] for x in svc.predict_proba(X_train)]

liste_acc = []

for seuil in [x * 0.01 for x in range(100)]:

    Pred = [cut_seuil(x, seuil) for x in Proba_SVM]

    liste_acc.append(round(sum(Pred == Y_train)/len(Y_train)*100,2))

df_acc = pd.DataFrame({"seuil" : [x * 0.01 for x in range(100)], 'Acc_SVM' : liste_acc})



plt.plot(df_acc.seuil, df_acc.Acc_SVM , label='SVM',color='blue')

plt.legend(loc=0)

print(df_acc.loc[df_acc.Acc_SVM == max(df_acc.Acc_SVM)])



    # 5 seuil nous permettent de maximiser l'accuracy sur l'échantillon de train. Nous pouvons faire une soumission en utilisant chacun de ces seuils pour constater la différence sur le test.

pd.DataFrame({"PassengerId": test["PassengerId"], "Survived": [cut_seuil(x[1], 0.57) for x in svc.predict_proba(X_test)]}).to_csv('Soumission_SVM_057.csv', index=False)

pd.DataFrame({"PassengerId": test["PassengerId"], "Survived": [cut_seuil(x[1], 0.58) for x in svc.predict_proba(X_test)]}).to_csv('Soumission_SVM_058.csv', index=False)

pd.DataFrame({"PassengerId": test["PassengerId"], "Survived": [cut_seuil(x[1], 0.59) for x in svc.predict_proba(X_test)]}).to_csv('Soumission_SVM_059.csv', index=False)

pd.DataFrame({"PassengerId": test["PassengerId"], "Survived": [cut_seuil(x[1], 0.60) for x in svc.predict_proba(X_test)]}).to_csv('Soumission_SVM_060.csv', index=False)

pd.DataFrame({"PassengerId": test["PassengerId"], "Survived": [cut_seuil(x[1], 0.62) for x in svc.predict_proba(X_test)]}).to_csv('Soumission_SVM_062.csv', index=False)





# Modèle régression logistique

Proba_Reg = [x[1] for x in reg_sk.predict_proba(X_train)]

liste_acc = []

for seuil in [x * 0.01 for x in range(100)]:

    Pred = [cut_seuil(x, seuil) for x in Proba_Reg]

    liste_acc.append(round(sum(Pred == Y_train)/len(Y_train)*100,2))    

df_acc = pd.DataFrame({"seuil" : [x * 0.01 for x in range(100)], 'Acc_Reg' : liste_acc})



print("\n ------------------------------------------------------------ \n")

plt.plot(df_acc.seuil, df_acc.Acc_Reg ,color='red', label='Regression Logistique')

plt.legend(loc=0)

print(df_acc.loc[df_acc.Acc_Reg == max(df_acc.Acc_Reg)])

    # 4 seuils nous permettent de maximiser l'accuracy sur l'échantillon de train. Nous pouvons faire une soumission en utilisant chacun de ces seul pour constater la différence sur le test.



pd.DataFrame({"PassengerId": test["PassengerId"], "Survived": [cut_seuil(x[1], 0.49) for x in svc.predict_proba(X_test)]}).to_csv('Soumission_reg_049.csv', index=False)

pd.DataFrame({"PassengerId": test["PassengerId"], "Survived": [cut_seuil(x[1], 0.52) for x in svc.predict_proba(X_test)]}).to_csv('Soumission_reg_052.csv', index=False)

pd.DataFrame({"PassengerId": test["PassengerId"], "Survived": [cut_seuil(x[1], 0.53) for x in svc.predict_proba(X_test)]}).to_csv('Soumission_reg_053.csv', index=False)

pd.DataFrame({"PassengerId": test["PassengerId"], "Survived": [cut_seuil(x[1], 0.65) for x in svc.predict_proba(X_test)]}).to_csv('Soumission_reg_065.csv', index=False)
from lime.lime_tabular import LimeTabularExplainer



df_poids = pd.DataFrame({'variable': X_train.columns})

explainer = LimeTabularExplainer(

    training_data=X_train

    , mode='classification'

    , feature_names=X_train.columns

    , discretize_continuous=False

)

# Boucle sur les ligne du dataframe pour appliquer LIME par ligne

poids = pd.DataFrame(data=[],columns = X_train.columns)

for idx, line in X_train.iterrows():

    exp = explainer.explain_instance(

        data_row=line

        , predict_fn=svc.predict_proba 

            # Fonction de prédiction du modèle à analyser.

    )

    temp = {i[0]: i[1] for i in exp.as_map()[1]}

    new_line = [temp[x] if x in temp.keys() else None for x in range(len(X_train.columns))]

    poids = pd.concat([poids, pd.DataFrame(data=[new_line],columns= X_train.columns)])

poids = poids.mean().reset_index()

poids.columns = ['variable','poids']

df_poids = df_poids.merge(poids, on='variable',how='left')



print(df_poids)