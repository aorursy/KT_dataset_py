## Importar las librerías requeridas

import sqlite3

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

from matplotlib.colors import ListedColormap

from sklearn.model_selection import train_test_split

from sklearn import preprocessing

from sklearn.preprocessing import StandardScaler

from sklearn.neural_network import MLPClassifier

from sklearn.gaussian_process import GaussianProcessClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import ExtraTreesClassifier

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis



import warnings

warnings.filterwarnings('ignore')
## Obtención de datos

#Conectar a la base de datos

path = "../input/"

database = path + 'database.sqlite'

conn = sqlite3.connect(database)



#Listamos todas las tablas que la componen.

query = "SELECT name as TablasBD FROM sqlite_master WHERE type='table';"

table_soccer = pd.read_sql(query, conn)
table_soccer
#Obtención de tablas de datos requeridas

player_data = pd.read_sql("SELECT * FROM Player;", conn)

player_atts = pd.read_sql("SELECT * FROM Player_Attributes;", conn)

team_data = pd.read_sql("SELECT * FROM Team;", conn)

team_atts = pd.read_sql_query("SELECT * FROM Team_Attributes", conn)

match_data = pd.read_sql("SELECT * FROM Match;", conn)
player_data.head()
player_atts.head()
team_data.head()
team_atts.head()
match_data.head()
#Reducir los datos de coincidencia

#Selección de columnas más representativas

columns = ['id','country_id', 'league_id', 'stage', 'match_api_id', 'home_team_api_id', 'away_team_api_id', 

           'home_team_goal', 'away_team_goal', 'B365H', 'B365D', 'B365A', 'BWH', 'BWD', 'BWA', 'IWH', 

           'IWD', 'IWA', 'LBH', 'LBD', 'LBA', 'PSH', 'PSD', 'PSA', 'WHH', 'WHD', 'WHA', 'SJH', 'SJD', 

           'SJA', 'VCH', 'VCD', 'VCA', 'GBH', 'GBD', 'GBA', 'BSH', 'BSD', 'BSA']



X_data = match_data[columns]

X_data.dropna(subset = columns, inplace = True)

X = X_data
len(X)
#Definir función para obtener quien ganó el partido. Local: 1 Visitante: 2 Empate: 0

def matchresult(homeScore, awayScore):

    if(homeScore > awayScore):

        return 1

    elif(homeScore < awayScore):

        return 2

    else:

        return 0
#Determinar si gano el equipo local o visitante (apoyándose en la función matchresult)

X['result'] = match_data.apply(lambda r: matchresult(r['home_team_goal'], r['away_team_goal']), axis=1)

y = X['result']

#Descartar las características, para no perjudicar la precisión del modelo

X = X.drop('result',1)

X = X.drop('home_team_goal',1)

X = X.drop('away_team_goal',1)
X.head()
y.head()
#Obtención de datos de entrenamiento y de test

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=0)
X_train.shape, y_train.shape
X_test.shape, y_test.shape
model = ExtraTreesClassifier()

model.fit(X_train,y_train)
model.score(X_test, y_test)
importances = model.feature_importances_

std = np.std([tree.feature_importances_ for tree in model.estimators_],

             axis=0)

indices = np.argsort(importances)[::-1]



# Mostrar el ranking de caracteríaticas ordenadas por relevancia

print("Feature ranking:")



for f in range(X.shape[1]):

    print("%d. feature %d %s \n    (%f)" % (f + 1, indices[f], columns[indices[f]], importances[indices[f]]))



# Representar las características más representativas para el modelo

plt.figure()

plt.title("Feature importances")

plt.bar(range(X.shape[1]), importances[indices],

       color="r", yerr=std[indices], align="center")

plt.xlabel('features')

plt.ylabel('importance')

plt.title('Feature importance')

plt.xticks(range(X.shape[1]) , indices)

plt.xlim([-1, X.shape[1]])

plt.show()
def Evaluacion_Modelos(X, y):

   

    # Nombre de los diferentes algoritmos de clasificación utilizados en las pruebas

    names = ["Extra Tree", "Decision Tree", "Random Forest", "Neural Net", "AdaBoost", "Naive Bayes", "QDA"]



    classifiers = [

        ExtraTreesClassifier(),

        DecisionTreeClassifier(),

        RandomForestClassifier(),

        MLPClassifier(alpha=1),

        AdaBoostClassifier(),

        GaussianNB(),

        QuadraticDiscriminantAnalysis()]

    

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.4, random_state=0)

    

    # Iterar sobre clasificadores

    for name, clf in zip(names, classifiers):

        clf.fit(X_train, y_train)

        score = clf.score(X_test, y_test)

   

        print("%s: \n   %f" % (name, score))
Evaluacion_Modelos(X, y)
match_team = pd.merge(match_data, team_atts, left_on='home_team_api_id', right_on='team_api_id')

match_team.head()
team_atts_date_sort = team_atts.groupby('team_api_id').apply(pd.DataFrame.sort_values, 'date')

team_atts_nodups = team_atts_date_sort.drop_duplicates(subset= 'team_api_id', keep = 'last', inplace=False)
match_home = pd.merge(X_data, team_atts_nodups, left_on='home_team_api_id', right_on='team_api_id', how='inner')

match_home_away = pd.merge(match_home, team_atts_nodups, left_on='away_team_api_id', right_on='team_api_id', how='inner')
#Contiene información de partidos y las características de los equipos (tanto del equipo local (_x) como visitante (_y))

match_home_away.head()
#Establecer valores númericos a las columnas

le = preprocessing.LabelEncoder()

match_home_away_le = match_home_away.apply(le.fit_transform)
match_home_away_le.head()
X = match_home_away_le
#Determinar el resultado del partido, ganó el equipo local, el visitante o fue empate (apoyandose en la función matchresult)

X['result'] = match_home_away_le.apply(lambda r: matchresult(r['home_team_goal'], r['away_team_goal']), axis=1)

y = X['result']

#Descartar las características, para no perjudicar la precisión del modelo

X = X.drop('result',1)

X = X.drop('home_team_goal',1)

X = X.drop('away_team_goal',1)
X.head()
Evaluacion_Modelos(X, y)
player_atts_group_sort = player_atts.groupby('player_api_id').apply(pd.DataFrame.sort_values, 'date')

player_atts_nodups = player_atts_group_sort.drop_duplicates(subset= 'player_api_id', keep = 'last', inplace=False)
player_atts_nodups.shape
player_atts_nodups.dropna(axis=0, how='any', inplace=True)
player_atts_nodups.shape
player_atts_nodups
player_atts_nodups_le = player_atts_nodups.apply(le.fit_transform)
y = player_atts_nodups_le['overall_rating']

X = player_atts_nodups_le



#Descartar las características, para no perjudicar la precisión del modelo

X = X.drop('id', axis=1, inplace=False)

X = X.drop('player_fifa_api_id', axis=1, inplace=False)

X = X.drop('player_api_id', axis=1, inplace=False)

X = X.drop('date', axis=1, inplace=False)

X = X.drop('overall_rating', axis=1, inplace=False)
#Obtención de datos de entrenamiento y de test

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=0)



model = ExtraTreesClassifier()

model.fit(X_train,y_train)
columns = list(X.columns.values)

importances = model.feature_importances_

std = np.std([tree.feature_importances_ for tree in model.estimators_],

             axis=0)

indices = np.argsort(importances)[::-1]



# Mostrar el ranking de caracteríaticas ordenadas por relevancia

print("Feature ranking:")



for f in range(X.shape[1]):

    print("%d. feature %d %s \n    (%f)" % (f + 1, indices[f], columns[indices[f]], importances[indices[f]]))



# Representar las características más representativas para el modelo

plt.figure()

plt.title("Feature importances")

plt.bar(range(X.shape[1]), importances[indices],

       color="r", yerr=std[indices], align="center")

plt.xlabel('features')

plt.ylabel('importance')

plt.title('Feature importance')

plt.xticks(range(X.shape[1]) , indices)

plt.xlim([-1, X.shape[1]])

plt.show()