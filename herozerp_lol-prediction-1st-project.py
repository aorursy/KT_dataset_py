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
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, Binarizer, KBinsDiscretizer, LabelEncoder, OneHotEncoder

from sklearn.model_selection import train_test_split, cross_val_score, learning_curve



url = '../input/league-of-legends/games.csv'



data = pd.read_csv(url, index_col=0, encoding = "ISO-8859-1")
#On vire les variables qui ne me serviront à rien, connaissant le jeu (et pour un premier projet, surtout)

df = data.copy()

df = df.drop(['creationTime', 'seasonId'], axis=1)

df = df.drop(df.columns[8: df.shape[1]], axis=1)

df.head()
#Modifier notre index.

df['Index'] = range(0, df.shape[0])

df = df.set_index('Index')

df.head()
#Modifier les valeurs 1 et 2 en 0 et 1

df['winner'] = df['winner'].replace(1, 0)

df['winner'] = df['winner'].replace(2, 1)



df['winner'].unique()
df.shape
#Compter le nombre de fois où la team 1 a gagné.

df['winner'].value_counts(normalize=True)
#Choisir notre target et notre data

X = df.drop(['winner'], axis=1)

y = df['winner']
#Créer des sous-ensembles quand l'équipe 1 gagne et quand elle perd

team1_win = df[df['winner'] == 0]

team1_lose = df[df['winner'] == 1]
#Tracer les histogrammes des colonnes

for col in df.select_dtypes('int'):

    plt.figure()

    sns.distplot(df[col])
#Création petits groupes pour partager les gameDuration par index 

df['1st_group_gd'] = df.index < 10000

df['2nd_group_gd'] = (df.index >= 10000) & (df.index < 20000)

df['3rd_group_gd'] = (df.index >= 20000) & (df.index < 30000)

df['4th_group_gd'] = (df.index >= 30000) & (df.index < 40000)

df['5th_group_gd'] = (df.index >= 40000) & (df.index < df.shape[0])



#Remplacer les true & false par 1 & 0 -> Méthode 1

df = df.applymap(lambda x: 1 if x == True else x)

df = df.applymap(lambda x: 0 if x == False else x)



df.head()
#Afficher les variables corellées.

sns.clustermap(df.corr())
#Vérifier si y'a une corrélation avec des %

df.corr()['winner'].sort_values()
#Créer les set

trainset, testset = train_test_split(df, test_size=0.2, random_state=0) 
#Vérifier le nombre dans les set

trainset['winner'].value_counts()
#Les proportions sont bien gardées

testset['winner'].value_counts()
#Autre méthode pour les true/false en 1/0 -> Encodage

def encodage(df):

#    code = {

#        'True':1,

#        'False':0

#    }

#    

#    for col in df.select_dtypes('bool').columns:

#        df.loc[col] = df[col].map(code)

#        

    return df
#Plus tard, si nous voulons supprimer quelques colonnes pour tester l'efficatité de notre model, on le fera ici

def imputation(df):

    df = df.drop(df.columns[8: df.shape[1]], axis=1)

    df = df.drop(['gameDuration'], axis=1) #D'ailleurs, après plusieurs tentatives, en supprimant la colonne gameDuration, les résultats sont meilleurs

    return df
def preprocessing(df):

    

    df = encodage(df)

    df = imputation(df)

    

    X = df.drop('winner', axis=1)

    y = df['winner']

    

    print(y.value_counts())

    

    return X, y
#Créer nos X_train et y_train automatisés.

X_train, y_train = preprocessing(trainset)
#Créer nos variables de test automatisées

X_test, y_test = preprocessing(testset)
from sklearn.metrics import f1_score, confusion_matrix, classification_report, recall_score

from sklearn.model_selection import learning_curve

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn import svm

from sklearn.svm import SVC

from sklearn.pipeline import make_pipeline

from sklearn.feature_selection import SelectKBest, f_classif

from sklearn.preprocessing import PolynomialFeatures, StandardScaler
def evaluation(model):

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    

    print(confusion_matrix(y_test, y_pred))

    print(classification_report(y_test, y_pred))

    

    # Savoir si il est en overfitting ou en underfitting

    #N, train_score, val_score = learning_curve(model, X_train, y_train, 

    #                                           cv=4, scoring='f1',

    #                                    train_sizes=np.linspace(0.1, 1, 10))

    

    #plt.figure(figsize=(12, 8))

    #plt.plot(N, train_score.mean(axis=1), label='Train_score')

    #plt.plot(N, val_score.mean(axis=1), label='Val_score')

    #plt.legend()
#Création d'un premier model basique mais qui donne des résultats finalement assez intéressants.

model = KNeighborsClassifier(n_neighbors=5)

evaluation(model)
#Vérifier les variables les plus importantes

#pd.DataFrame(model.feature_importances_, index=X_train.columns).plot.bar(figsize=(12, 8))
#Il faut d'abord créer une pipeline

preprocessor = make_pipeline(SelectKBest(f_classif, k='all'))
#Définir les modèles qu'on utilisera. Liste non exhaustive, on choisit 4 algo connus

RandomForest = make_pipeline(preprocessor, 

                             RandomForestClassifier(random_state=0))

AdaBoost = make_pipeline(preprocessor, 

                             AdaBoostClassifier(random_state=0))

SVM = make_pipeline(preprocessor, StandardScaler(),

                             SVC(random_state=0))

KNN = make_pipeline(preprocessor, StandardScaler(),

                             KNeighborsClassifier())



list_of_models = {'RandomForest':RandomForest, 

                  'AdaBoost':AdaBoost,

                  'SVM': SVM,

                  'KNN': KNN

                 }
#Automatiser chaque essai

for name, model in list_of_models.items():

    print(name)

    evaluation(model)
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
#Création d'une liste d'hyper_paramètres

hyper_params = {'svc__C': [1, 2],  

                'svc__kernel': ['linear', 'poly', 'rbf']

}
#Trouver les meilleurs paramètres.

grid = GridSearchCV(SVM, hyper_params, scoring='recall', cv=4)



#Entraîner la grille

grid.fit(X_train, y_train)



#Vérifier les meilleurs paramètres

print(grid.best_params_)



#Prédire y

y_pred = grid.predict(X_test)



print(classification_report(y_test, y_pred))
#Puis on entraîne enfin notre model avec les meilleurs paramètres possibles

evaluation(grid.best_estimator_)
from sklearn.metrics import precision_recall_curve
#Utilisation du model qui fonctionne bien sans toucher aux hyper_paramètres

model = SVM
from sklearn.metrics import precision_recall_curve
#Récupérer les données des variables

precision, recall, threshold = precision_recall_curve(y_test, grid.best_estimator_.decision_function(X_test))
#Afficher les courbes de precision et de recall

plt.plot(threshold, precision[:-1], label='precision')

plt.plot(threshold, recall[:-1], label='recall')

plt.legend()
#Décider du point de décision

def final_model(model, X, threshold=0):

    return model.decision_function(X) > threshold



y_pred = final_model(grid.best_estimator_, X_test, threshold=0)
#Vérifier le score F1

f1_score(y_test, y_pred)
#Vérifier le recall score

recall_score(y_test, y_pred)