#Acquisition des bibliotheques
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
from sklearn import preprocessing, model_selection, metrics
import warnings
from sklearn.model_selection import train_test_split
warnings.filterwarnings("ignore")
import seaborn as sns

import plotly as pl
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode(connected=True)
import plotly.graph_objs as go
%matplotlib inline
#Lecture des donnees
dtst = pd.read_csv("../input/energydata_complete.csv")
dtst.head()
#Visualisation des informations sur les donnees
dtst.info()
#Description des donnees
dtst.describe()
#Visualisation du nombre de lignes et colonnes
print('Nombre de lignes : ' , dtst.shape[0])
print('Nombre de colonnes : ' , dtst.shape[1])
#Verification des valeurs nulles
dtst.isnull().sum().sort_values(ascending = True)
#Division des donnes en donnees donnees test et entrainements
# 75% des donnees pour la formation de models et 25 test
train, test = train_test_split(dtst,test_size=0.25,random_state=40)
train.describe()
#Description des donnees d'entrainement
train.describe()
#Division des colonnes de leur type

col_temp = ["T1","T2","T3","T4","T5","T6","T7","T8","T9"]

col_hum = ["RH_1","RH_2","RH_3","RH_4","RH_5","RH_6","RH_7","RH_8","RH_9"]

col_weather = ["T_out", "Tdewpoint","RH_out","Press_mm_hg",
                "Windspeed","Visibility"] 
col_light = ["lights"]

col_randoms = ["rv1", "rv2"]

col_target = ["Appliances"]
#Separation variables dependantes et variables independantes
feature = train[col_temp + col_hum + col_weather + col_light + col_randoms ]
target = train[col_target]
feature.describe()
#Verification des valeurs dans lights
feature.lights.value_counts()
target.describe()
_ = feature.drop(['lights'], axis=1 , inplace= True) ;
feature.head(2)
#Comprehension de la variation chronologique de la consommation d'énergie de l'appareil
visData = go.Scatter( x= dtst.date  ,  mode = "lines", y = dtst.Appliances )
layout = go.Layout(title = 'Mesure de la consommation d_énergie des appareils' , xaxis=dict(title='Date'), yaxis=dict(title='(Wh)'))
fig = go.Figure(data=[visData],layout=layout)
iplot(fig)
dtst['WEEKDAY'] = ((pd.to_datetime(dtst['date']).dt.dayofweek)// 5 == 1).astype(float)
dtst['WEEKDAY'].value_counts()
#Jour de la semaine
temp_weekday =  dtst[dtst['WEEKDAY'] == 0]
#Comprehension de la variation chronologique
visData = go.Scatter( x= temp_weekday.date  ,  mode = "lines", y = temp_weekday.Appliances )
layout = go.Layout(title = 'Mesure de la consommation d_énergie des appareils en semaine' , xaxis=dict(title='Date'), yaxis=dict(title='(Wh)'))
fig = go.Figure(data=[visData],layout=layout)

iplot(fig)
#Jour de la semaine
temp_weekday =  dtst[dtst['WEEKDAY'] == 0]
#Comprehension de la variation chronologique
visData = go.Scatter( x= temp_weekday.date  ,  mode = "lines", y = temp_weekday.Appliances )
layout = go.Layout(title = 'Mesure de la consommation d_énergie des appareils en semaine' , xaxis=dict(title='Date'), yaxis=dict(title='(Wh)'))
fig = go.Figure(data=[visData],layout=layout)

iplot(fig)
#Histogramme des fonctionnalites 
feature.hist(bins = 20, figsize= (12,16)) ;
f, ax = plt.subplots(2,2,figsize=(12,8))
v1 = sns.distplot(feature["RH_6"],bins=10, ax= ax[0][0])
v2 = sns.distplot(feature["RH_out"],bins=10, ax=ax[0][1])
v3 = sns.distplot(feature["Visibility"],bins=10, ax=ax[1][0])
v4 = sns.distplot(feature["Windspeed"],bins=10, ax=ax[1][1])
#Distribution des valeurs de la colonne Appliances en fonction de la frequence
f = plt.figure(figsize=(12,5))
plt.xlabel('Consommation d_appareils en Wh')
plt.ylabel('Frequence')
sns.distplot(target , bins=10 ) ;
print('Le pourcentage de la consommation de l_appareil est inférieur à 200 Wh')
print(((target[target <= 200].count()) / (len(target)))*100 )
# Correlation entre les colonnes météo, température, appareils
train_corr = train[col_temp + col_hum + col_weather +col_target+col_randoms]
corr = train_corr.corr()
# Masquage des valeurs repetees
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True
  
f, ax = plt.subplots(figsize=(16, 14))
#Génération d'un une carte de chaleur 
sns.heatmap(corr, annot=True, fmt=".2f" , mask=mask,)
    #Application de xticks
plt.xticks(range(len(corr.columns)), corr.columns);
    #Application de yticks
plt.yticks(range(len(corr.columns)), corr.columns)
    #Affichage
plt.show()
def get_redundant_pairs(df):
    '''Get diagonal and lower triangular pairs of correlation matrix'''
    pairs_to_drop = set()
    cols = df.columns
    for i in range(0, df.shape[1]):
        for j in range(0, i+1):
            pairs_to_drop.add((cols[i], cols[j]))
    return pairs_to_drop

# Fonction d'obtention de meilleures corrélations 

def get_top_abs_correlations(df, n=5):
    au_corr = df.corr().abs().unstack()
    labels_to_drop = get_redundant_pairs(df)
    au_corr = au_corr.drop(labels=labels_to_drop).sort_values(ascending=False)
    return au_corr[0:n]

print("Top Absolute Correlations")
print(get_top_abs_correlations(train_corr, 40))
#Division l'ensemble de données d'apprentissage en variables indépendantes et dépendantes
train_X = train[feature.columns]
train_y = train[target.columns]
#Pour eviter confusion faite ci-dessus, ces colonnes sont supprimées
train_X.drop(["rv1","rv2","Visibility","T6","T9"],axis=1 , inplace=True)
#Division l'ensemble de données de test en variables indépendantes et dépendantes
test_X = test[feature.columns]
test_y = test[target.columns]
#Pour eviter confusion faite ci-dessus, ces colonnes sont supprimées
test_X.drop(["rv1","rv2","Visibility","T6","T9"], axis=1, inplace=True)
train_X.columns
test_X.columns
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()

# Creation d'un ensemble de tests et de formation en incluant la colonne Appliances

train = train[list(train_X.columns.values) + col_target ]

test = test[list(test_X.columns.values) + col_target ]

# Creation d'un test factice et un ensemble de formation pour contenir des valeurs mises à l'échelle

sc_train = pd.DataFrame(columns=train.columns , index=train.index)

sc_train[sc_train.columns] = sc.fit_transform(train)

sc_test= pd.DataFrame(columns=test.columns , index=test.index)

sc_test[sc_test.columns] = sc.fit_transform(test)
sc_train.head()
sc_test.head()
sc_test.head()
#Suppression de la colonne Appliances de l'ensemble de formation
train_X =  sc_train.drop(['Appliances'] , axis=1)
train_y = sc_train['Appliances']

test_X =  sc_test.drop(['Appliances'] , axis=1)
test_y = sc_test['Appliances']
train_X.head()
train_y.head()
from sklearn.linear_model import Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.neural_network import MLPRegressor
from sklearn import neighbors
from sklearn.svm import SVR
models = [
           ['Lasso: ', Lasso()],
           ['Ridge: ', Ridge()],
           ['KNeighborsRegressor: ',  neighbors.KNeighborsRegressor()],
           ['SVR:' , SVR(kernel='rbf')],
           ['RandomForest ',RandomForestRegressor()],
           ['ExtraTreeRegressor :',ExtraTreesRegressor()],
           ['GradientBoostingClassifier: ', GradientBoostingRegressor()] ,
           ['MLPRegressor: ', MLPRegressor(  activation='relu', solver='adam',learning_rate='adaptive',max_iter=1000,learning_rate_init=0.01,alpha=0.01)]
         ]
#Exécution des modèles et mettre à jour des informations dans une liste model_data

import time
from math import sqrt
from sklearn.metrics import mean_squared_error

model_data = []
for name,curr_model in models :
    curr_model_data = {}
    curr_model.random_state = 78
    curr_model_data["Name"] = name
    start = time.time()
    curr_model.fit(train_X,train_y)
    end = time.time()
    curr_model_data["Train_Time"] = end - start
    curr_model_data["Train_R2_Score"] = metrics.r2_score(train_y,curr_model.predict(train_X))
    curr_model_data["Test_R2_Score"] = metrics.r2_score(test_y,curr_model.predict(test_X))
    curr_model_data["Test_RMSE_Score"] = sqrt(mean_squared_error(test_y,curr_model.predict(test_X)))
    model_data.append(curr_model_data)

model_data
dtst = pd.DataFrame(model_data)
dtst
#Affichage des resultat de R2
dtst.plot(x="Name", y=['Test_R2_Score' , 'Train_R2_Score' , 'Test_RMSE_Score'], kind="bar" , 
        title = 'Resultats Score R2' , figsize= (10,8)) ;
from sklearn.model_selection import GridSearchCV
param_grid = [{
              'max_depth': [80, 150, 200,250],
              'n_estimators' : [100,150,200,250],
              'max_features': ["auto", "sqrt", "log2"]
            }]
reg = ExtraTreesRegressor(random_state=40)
# Instantiate the grid search model
grid_search = GridSearchCV(estimator = reg, param_grid = param_grid, cv = 5, n_jobs = -1 , scoring='r2' , verbose=2)
grid_search.fit(train_X, train_y)
#Jeu de paramètres accordé
grid_search.best_params_
#Meilleurs paramètres possibles pour ExtraTreesRegressor
grid_search.best_estimator_
#Score R2 sur l'ensemble d'entraînement avec paramètres réglés
grid_search.best_estimator_.score(train_X,train_y)
#Score R2 sur l'ensemble de test avec paramètres réglés
grid_search.best_estimator_.score(test_X,test_y)
## Score RMSE sur l'ensemble de test avec paramètres réglés
np.sqrt(mean_squared_error(test_y, grid_search.best_estimator_.predict(test_X)))
#Obtention une liste triée des fonctionnalités par ordre d'importance
feature_indices = np.argsort(grid_search.best_estimator_.feature_importances_)
importances = grid_search.best_estimator_.feature_importances_
indices = np.argsort(importances)[::-1]
names = [train_X.columns[i] for i in indices]
# Create plot
plt.figure(figsize=(10,6))

# Create plot title
plt.title("Importance des Fonctionnalites")

# Add bars
plt.bar(range(train_X.shape[1]), importances[indices])

# Add feature names as x-axis labels
plt.xticks(range(train_X.shape[1]), names, rotation=90)

# Show plot
plt.show()
#Les 5 fonctionnalités les plus importantes sont :
names[0:5]
#Les 5 fonctionnalités les moins importantes sont : 'T7', 'Tdewpoint', 'Windspeed', 'T1', 'T5'
names[-5:]
train_important_feature = train_X[names[0:5]]
test_important_feature = test_X[names[0:5]]
from sklearn.base import clone
cloned_model = clone(grid_search.best_estimator_)
cloned_model.fit(train_important_feature , train_y)
print('Training set R2 Score - ', metrics.r2_score(train_y,cloned_model.predict(train_important_feature)))
print('Testing set R2 Score - ', metrics.r2_score(test_y,cloned_model.predict(test_important_feature)))
print('Testing set RMSE Score - ', np.sqrt(mean_squared_error(test_y, cloned_model.predict(test_important_feature))))
