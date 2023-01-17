'''

Librairies nécessaires

'''

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import numpy as np

from datetime import datetime

from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_squared_error

from math import radians, cos, sin, asin, sqrt

from sklearn.tree import DecisionTreeRegressor

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import GridSearchCV

from sklearn.feature_selection import RFECV

from haversine import haversine, Unit 

import csv as csv
'''

Chargement dataset

'''

#Chargement du dataset complet

dataset_original = pd.read_csv('/kaggle/input/train.csv')



# Création d'un copie

dataset = dataset_original.copy()
'''

Traitement des données

'''

# Vecteurs Coords pour calculer la Distance directe

pick_coords = dataset.loc[:, ["pickup_latitude","pickup_longitude"]] 

drop_coords = dataset.loc[:, ["dropoff_latitude","dropoff_longitude"]] 



#Conversion de coords à tuples pour nourrir la fonction haversine

pick_coords_tup = [tuple(x) for x in pick_coords.values]

drop_coords_tup = [tuple(x) for x in drop_coords.values]



# Calcul de la distance directe en KMS et conversion à MÉTRES

trip_dist = [haversine(pick_coords_tup[i], drop_coords_tup[i]) for i in range(len(pick_coords_tup))]

trip_dist = pd.Series(trip_dist)*1000



# Conversion de types (object -> datetime) 

dataset['pickup_datetime'] = pd.to_datetime(dataset['pickup_datetime'])

dataset['dropoff_datetime'] = pd.to_datetime(dataset['dropoff_datetime'])



# Création de nouvelles variables à partir de 'pickup_datime' 

week_day = dataset['pickup_datetime'].dt.weekday   # day of the week (0=Monday ... 6=Sunday)

month = dataset['pickup_datetime'].dt.month   # month (0=Jan ... 11=Dec)

year = dataset['pickup_datetime'].dt.year   # year

time = dataset['pickup_datetime'].dt.time   # time

date = dataset['pickup_datetime'].dt.date   # date 



# Création du dataset avec les variables pertinentes

dataset = dataset[['vendor_id','passenger_count','pickup_longitude','pickup_latitude','dropoff_longitude','dropoff_latitude','trip_duration']]

dataset.insert(2, 'week_day', week_day, True)

dataset.insert(3, 'month', month, True)

dataset.insert(4, 'trip_dist', trip_dist, True)
'''

Nettoyage de données

''' 

# Constraintes à appliquer:

#   -trip_duration < 30000  ( 8 heures)

#   -trip_duration < 45)

#   -trip_dist < 300 kms



 

#Mise en place du netoyage selon constraints

dataset = dataset.drop(dataset[dataset.trip_duration > 30000].index)

dataset = dataset.drop(dataset[dataset.trip_duration < 45].index)

dataset = dataset.drop(dataset[dataset.trip_dist >= 300000].index)



dataset = dataset.drop(dataset[dataset.pickup_latitude < 40.1].index)

dataset = dataset.drop(dataset[dataset.pickup_latitude > 41.4].index)

dataset = dataset.drop(dataset[dataset.pickup_longitude < -86].index)

dataset = dataset.drop(dataset[dataset.pickup_longitude > -73].index)



dataset = dataset.drop(dataset[dataset.dropoff_latitude < 39].index)

dataset = dataset.drop(dataset[dataset.dropoff_latitude > 42].index)

dataset = dataset.drop(dataset[dataset.dropoff_longitude < -75].index)

dataset = dataset.drop(dataset[dataset.dropoff_longitude > -30].index)

'''

Creation du Train Set et du Test Set

'''

#Fonction pour diviser le dataset complet en Train set et Test set

def split_train_test(data, test_ratio):

    np.random.seed(42)

    shuffled_indices = np.random.permutation(len(data))

    test_set_size = int(len(data) * test_ratio)

    test_indices = shuffled_indices[:test_set_size]

    train_indices = shuffled_indices[test_set_size:]

    return data.iloc[train_indices], data.iloc[test_indices]



#Division du dataset

new_train, new_test = split_train_test(dataset, 0.2)



# Variables objetives (trip_duration)

y = new_train[['trip_duration']]

y_test = new_test[['trip_duration']]



# Suppression de la valeur objetive du train et test set

new_train.drop('trip_duration', axis=1, inplace=True)

new_test.drop('trip_duration', axis=1, inplace=True)

'''

Creation de l'arbre Régresseur sans régulariser

'''

# Création de l'arbre régresseur sans régularisation

tree_reg_nr = DecisionTreeRegressor(random_state=42)   



# Entrainement 

tree_reg_nr.fit(new_train, y)



# Afficher les hyperparamètres de l'arbre

tree_reg_nr.get_params()
'''

Evaluation de l'arbre entrainé sur le training set

'''

# Modèle sans Régularisation

predictions_nr = tree_reg_nr.predict(new_train)

tree_mse = mean_squared_error(y, predictions_nr)

tree_rmse_nr = np.sqrt(tree_mse)

print("")

print("Error in Train set = " + str(tree_rmse_nr))
'''

Afficher la rélation entre valeurs réelles et calculées sans régularisation

'''

def plot_predictions_vs_val_reelles(tree_reg, y, title, ylabel="$variableObjetive$"):

    y_pred = tree_reg.predict(new_train)[:100]

    y_pred = np.asarray(y_pred, dtype='float64')

    y_orig = y.iloc[:100]

    y_orig = np.asarray(y_orig, dtype='float64')

    x1 = np.linspace(0, 100, 100)#.reshape(-1, 1)

    plt.axis([0, 100, 0, 4000])

    plt.xlabel("$Instances (train set)$", fontsize=12)

    if ylabel:

        plt.ylabel(ylabel, fontsize=12)

    plt.plot(x1, y_orig, "bo", label='valeur_réelle')

    plt.plot(x1, y_pred, "r.-", label='valeur_calculée')

    plt.legend(loc='upper right')

    plt.title(title, fontsize=14)

    plt.show()



plot_predictions_vs_val_reelles(tree_reg_nr, y,"Prédiction modèle libre")    
'''

Evaluation avec K-fold Cross-Validation (Pas de régularisation)

'''

# fonction auxiliaire pour afficher les résultats de K-fold Cross-Validation

def display_scores(scores):

    print("")

    print("Scores:", scores)

    print("Mean:", scores.mean())

    print("Standard deviation:", scores.std())

    print("")

    

# K-fold Cross-Validation

scores_nr = cross_val_score(tree_reg_nr, new_train, y, scoring="neg_mean_squared_error", cv=5)

rmse_scores_nr = np.sqrt(-scores_nr)



display_scores(rmse_scores_nr) 
'''

Optimisation du modèle

'''

# Différentes ensembles de paramètres à essayer

param_grid_1 = [{'max_depth': [2, 4, 6], 'max_leaf_nodes': [2, 20, 200]}]   #best_params = max_depth: 6, max_leaf_nodes: 200

param_grid_2 = [{'max_depth': [6, 7, 8], 'max_leaf_nodes': [20, 200, 500]}]     #best_params = max_depth: 8, max_leaf_nodes: 500

param_grid_3 = [{'max_depth': [8, 9, 10], 'max_leaf_nodes': [450, 500, 550]}]     #best_params =  max_depth: 9, max_leaf_nodes: 450



# Bucle pour afficher les meilleurs résultats de chaque essai  

score_list = [] 

for param in list([param_grid_1, param_grid_2, param_grid_3]) :

    

    grid_search = GridSearchCV(tree_reg_nr, param, cv=5, n_jobs=-1, 

                           scoring='neg_mean_squared_error', 

                           return_train_score=True)

    grid_search.fit(new_train, y)

    cv_results = grid_search.cv_results_

    score_list.append(np.sqrt(-cv_results["mean_test_score"]))

    

    for mean_score, params in zip(cv_results["mean_test_score"], cv_results["params"]):

        print(np.sqrt(-mean_score), params)



    print("")

    print("Best paramateres: ", grid_search.best_params_)

    print("")



'''

Afficher les performances de différents sets de paramètres

'''

def plot_erreur_par_config(score_list,):

    x1 = np.linspace(0, 9, 9)

    plt.ylabel("$Erreur (RMSE)$", fontsize=12)

    plt.title("Influence Hyper-paramètres", fontsize=14)

    for resultat, style, label_ in ((score_list[0],"b.-", "config_1"),

                            (score_list[1],"r.-", "config_2"),

                            (score_list[2],"g.-", "config_3")):

        plt.plot(x1, resultat, style, label=label_)

    plt.legend(loc='upper right')

    plt.show()



plot_erreur_par_config(score_list)
'''

Création Arbre Régressor avec Régularisation

'''

# Meilleurs paramètres: max_depth= 9, max_leaf_nodes= 450



# Création de l'arbre régresseur avec régularisation

tree_reg_r = DecisionTreeRegressor(max_depth = 9, max_leaf_nodes = 450, random_state=42)   



# Entrainement 

tree_reg_r.fit(new_train, y)



# Montrer hyperparamètres de l'arbre

tree_reg_r.get_params()
'''

Evaluation de l'arbre entrainé sur training set

'''

# Modèle avec Régularisation

predictions_r = tree_reg_r.predict(new_train)

tree_mse = mean_squared_error(y, predictions_r)

tree_rmse_r = np.sqrt(tree_mse)

print("")

print("Error in Train set (Regularized) = " + str(tree_rmse_r))
'''

Afficher la rélation entre valeurs réelles et calculées avec régularisation

'''

plot_predictions_vs_val_reelles(tree_reg_r, y, "Prédiction modèle limité")
'''

Evaluation avec K-fold Cross-Validation (Régularisation)

'''

#  K-fold Cross-Validation

scores_r = cross_val_score(tree_reg_r, new_train, y, scoring="neg_mean_squared_error", cv=5)

rmse_scores_r = np.sqrt(-scores_r)



display_scores(rmse_scores_r) 
'''

Evaluation de l'arbre sur TESTING set

'''

# Modèle sans Régularisation

predictions_nr = tree_reg_nr.predict(new_test)

tree_mse = mean_squared_error(y_test, predictions_nr)

tree_rmse_nr_test = np.sqrt(tree_mse)

print("")

print("Error in Test set = " + str(tree_rmse_nr_test))



# Modele avec Régularisation

predictions_r = tree_reg_r.predict(new_test)

tree_mse = mean_squared_error(y_test, predictions_r)

tree_rmse_r_test = np.sqrt(tree_mse)

print("")

print("Error in Test set (Regularized) = " + str(tree_rmse_r_test))



#%%

#'''

#Affichage des erreurs finales

#'''

err_train = (tree_rmse_nr, tree_rmse_r)

err_test = (tree_rmse_nr_test, tree_rmse_r_test)

x1 = np.linspace(0, 2, 2)#.reshape(-1, 1)

plt.axis([0, 3, 0, 600])

#plt.xlabel("$Instances (train set)$", fontsize=12)

plt.ylabel("$Erreur(RMSE)$", fontsize=12)

plt.plot(x1, err_train, "bo-", label='erreur_train_set')

plt.plot(x1, err_test, "ro-", label='erreur_test_set')

plt.legend(loc='upper right')

plt.title("$Evolution-de-l'erreur-total$", fontsize=14)

plt.show()
'''

Affichage de l'arbre final

'''

from sklearn.tree import export_graphviz

import graphviz



# Création d'une représentation visuelle de l'arbre

export_graphviz(tree_reg_r,out_file='duration_tree.dot',

                feature_names=new_train.columns,

                class_names=y.columns,

                rounded=True,

                filled=True)    



#Conversion du fichier (.dot -> png)

!dot -Tpng duration_tree.dot -o duration_tree.png -Gdpi=50



# Afficher graphique

from IPython.display import Image 

Image(filename = 'duration_tree.png')
'''

Creation d'un arbre avec régularisation NON OPTIMALE juste pour visualisation

'''

# Création de l'arbre régresseur sans régularisation

tree_reg_r_ale = DecisionTreeRegressor(max_depth = 3, random_state=42)   



# Entrainement 

tree_reg_r_ale.fit(new_train, y)