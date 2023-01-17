import numpy as np

import pandas as pd
data = pd.read_csv("../input/housedata/data.csv")

print("Taille du jeu de données :")

print(data.shape)

print("\nType des colonnes :")

print(data.info(verbose=True))

print("\nOn remarque que la colonne 'date' n'est pas considérée comme un type de date, mais comme 'object'.")

print("Il faudra changer le type par la suite.")

print("\nRegarder à quoi ressemblent les données :")

data.head()
# Pour les colonnes numériques :

data.describe()
# On remarque le minimum du prix est 0, on va retirer les maisons vendues pour 0$ :

data = data[data.price > 0]
# Pour les colonnes qualitatives: 

print("Nombre de valeurs pour les 10 villes contenant le plus de valeurs :")

data.city.value_counts().head(10)
# Construction de la variable cible Y (prix) et suppression de cette variable de data :

Y = data.price

data = data.loc[:,data.columns.difference(['price'])]

# On construit une cible en trois tranches de prix :

Y_cut = pd.cut(Y,Y.quantile([0,.33,.66,1]), include_lowest=True) # On inclue pour éviter de créer des NAs

print("Quantiles à 33% et 66% :")

print(Y.quantile([.33,.66]))



# On modifie le type de la colonne date :

# Ici, nous allons utiliser le Timestamp sans chercher à retrouver le mois où l'année.

data.date = pd.to_numeric(pd.to_datetime(data.date))

print("\nLe type de la colonne 'date' a bien changé en integer :")

print(data[['date']].info())
# On visualise la boxplot des prix :

Y.plot.box()



# Quelques rares valeurs sont supérieures à 50 000 000$, on supprime donc ces outliers :

data = data.loc[Y<=5000000]

Y_cut = Y_cut.loc[Y<=5000000]

Y = Y.loc[Y<=5000000]
ix_train = data.sample(frac=.7, replace=False, random_state=0).index.values # On tire 70% des individus parmis les n_individus présents, sans remplacement

ix_test = data.index.difference(ix_train).values # Les autres iandexes forment le test set

print("Nombre d'éléments dans le jeu d'entraîntement : {}".format(len(ix_train)))

print("Nombre d'éléments dans le jeu de test : {}".format(len(ix_test)))
print("Moyenne des prix sur le jeu d'entraînement :")

print(round(Y.loc[ix_train].mean()))

print("\nMoyenne des prix sur le jeu de test :")

print(round(Y.loc[ix_test].mean()))
from sklearn.model_selection import KFold



# Création de 5 jeux d'indexes :

kf = KFold(n_splits=5)

folds = [(train, test) for train, test in kf.split(data.index.values)]



print("Nombre de tupple crées :")

print(len(folds))

print("\nNombre d'éléments dans le train et le test set du premier fold :")

print(len(folds[0][0]), len(folds[0][1]))
data = data[data.columns.difference(['country', 'street', 'statezip'])]



print("Comme on peut le voir, supprimer les colonnes dans le jeu de données complet impact les jeux d'entrainement et de test :")

data.loc[ix_train].head()
### 1. Standardisation des données numériques :

from sklearn import preprocessing



# Initialisation du standard scaler :

std_scaler = preprocessing.StandardScaler()

# Récupérer la liste des colonnes numériques :

num_cols = data.select_dtypes(exclude=["object"]).columns



# Pour ne pas "tricher" en appliquant la standardisation à partir des données du test set,

# on paramètre le scaler avec les données du jeu d'entrainement exclusivement :

std_scaler.fit(data.loc[:,num_cols].loc[ix_train])

# On applique ensuite sur tout le jeu de données :

data.loc[:,num_cols] = std_scaler.transform(data.loc[:,num_cols])
### 2. Transformation les modalités de la variable city :

dict_replace = {city_name:'Other' for city_name,counter in data.city.value_counts().iteritems() if counter<50}

data.city = data.city.replace(dict_replace)



### 3. Encodage en dummy variables de city :

# Construction des colonnes contenant les dummies :

data_with_dummies = pd.get_dummies(data, drop_first=True) # On retire la première colonne pour éviter la multi-colinéarité dans les modèles linéaires.

data_with_dummies.head()
from sklearn.cluster import KMeans

# Initialisation des paramètres de l'algorithme :

Kmeans_model = KMeans(n_clusters=6, init = 'k-means++', random_state=123)



# On applique le clustering sur le jeu d'entraînement (train) :

# On enregistre dans la variable clusters le numéro du cluster auquel chaque maison est identifiée

clusters = Kmeans_model.fit_predict(data.loc[ix_train, data.columns.difference(['city'])])



# On enregistre les clusters et les colonnes à comparer dans un dataframe pandas :

train_clusters = pd.DataFrame([*zip(clusters,Y.loc[ix_train],Y_cut.loc[ix_train])], columns=['cluster','price','price_cut'])



train_clusters.boxplot(by='cluster')

print('On affiche les boxplot de prix par clusters :')
print("On affiche la table des profils ligne par tranche de prix, pour chaque cluster :")

cross_table = pd.crosstab(train_clusters.cluster, train_clusters.price_cut, normalize='index')

cross_table.style.background_gradient(cmap='Reds', axis=1, low=0, high=1)
# Pour chaque maison du jeu test, on retrouve le cluster auquel elle appartient :

clusters_test = Kmeans_model.predict(data.loc[ix_test, data.columns.difference(['city'])])



# On construit un dictionnaire pour associer à chaque cluster la classe dominante :

dict_cluster_pred = {

    0:'(370000.0, 575000.0]' ,

    1:'(575000.0, 26590000.0]' ,

    2:'(7799.999, 370000.0]' ,

    3:'(575000.0, 26590000.0]' ,

    4:'(575000.0, 26590000.0]' ,

    5:'(7799.999, 370000.0]'

}

y_test_clusters = [dict_cluster_pred[c] for c in clusters_test]



# Calcul de l'accuracy et du recall :

from sklearn.metrics import accuracy_score, recall_score

print("Classe dominante du cluster :\n- Accuracy : {}\n- Recall   : {}".format(

    round(accuracy_score(y_test_clusters, Y_cut.astype(str)[ix_test]),4),

    round(recall_score(y_test_clusters, Y_cut.astype(str)[ix_test], average='macro'),4)

))
from sklearn.neighbors import KNeighborsClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.neural_network import MLPClassifier



# Initialisation des modèles :

KNN_model = KNeighborsClassifier(n_neighbors=11)

RFC_model = RandomForestClassifier(n_estimators=100, max_depth=8, min_samples_leaf=7, random_state=0)

MLPC_model = MLPClassifier(hidden_layer_sizes=(50,), activation='relu', max_iter=1000, random_state=0)
# Pour la prédiction, le type Categorical (catégories ordonnées de pandas)

# n'est pas compatible avec les fonctions de sklearn.metrics, on 

# le convertit dans le type string :

Y_cut_str = Y_cut.astype(str)



# Entraînement :

%time KNN_model.fit(data_with_dummies.loc[ix_train], Y_cut_str.loc[ix_train])

%time RFC_model.fit(data_with_dummies.loc[ix_train], Y_cut_str.loc[ix_train])

%time MLPC_model.fit(data_with_dummies.loc[ix_train], Y_cut_str.loc[ix_train])



# Prédiction sur le jeu de test :

y_test_knn = KNN_model.predict(data_with_dummies.loc[ix_test])

y_test_rfc = RFC_model.predict(data_with_dummies.loc[ix_test])

y_test_mlpc = MLPC_model.predict(data_with_dummies.loc[ix_test])



# On affiche la précision (accuracy) et le rappel (recall) pour chaque méthode :

from sklearn.metrics import accuracy_score, recall_score

print("\n11 plus proches voisins :\n- Accuracy : {}\n- Recall   : {}".format(

    round(accuracy_score(y_test_knn, Y_cut_str[ix_test]),4),

    round(recall_score(y_test_knn, Y_cut_str[ix_test], average='macro'),4)

))

print("\nRandom Forest Classifier (100 arbres) :\n- Accuracy : {}\n- Recall   : {}".format(

    round(accuracy_score(y_test_rfc, Y_cut_str[ix_test]),4),

    round(recall_score(y_test_rfc, Y_cut_str[ix_test], average='macro'),4)

))

print("\nMulti Layer Perceptron Classifier (50 couches) :\n- Accuracy : {}\n- Recall   : {}".format(

    round(accuracy_score(y_test_mlpc, Y_cut_str[ix_test]),4),

    round(recall_score(y_test_mlpc, Y_cut_str[ix_test], average='macro'),4)

))
from sklearn.model_selection import cross_val_score



# Cross validation avec 5 folds : 

scores_knn = cross_val_score(KNN_model, data_with_dummies, Y_cut_str, cv=5)

scores_rfc = cross_val_score(RFC_model, data_with_dummies, Y_cut_str, cv=5)

scores_mlpc = cross_val_score(MLPC_model, data_with_dummies, Y_cut_str, cv=5)



# Afficher la moyenne et l'écart type des accuracy :

print("Accuracy KNN  : %0.2f (+/- %0.2f)" % (scores_knn.mean(), scores_knn.std() * 2))

print("Accuracy RFC  : %0.2f (+/- %0.2f)" % (scores_rfc.mean(), scores_rfc.std() * 2))

print("Accuracy MLPC : %0.2f (+/- %0.2f)" % (scores_mlpc.mean(), scores_mlpc.std() * 2))
from sklearn.model_selection import KFold



# Création de 5 jeux d'indexes :

kf = KFold(n_splits=5)

folds = [(train, test) for train, test in kf.split(data_with_dummies.index)]

# KFold renvoie les index numérotés et pas le nom de la ligne

# on les utilisera avec .iloc et pas .loc comme précédemment.



# Application dans une boucle :

accuracy_knn, accuracy_rfc, accuracy_mlpc = [], [], []

for ix_train_cv, ix_test_cv in folds :

    # Entraînement :

    KNN_model.fit(data_with_dummies.iloc[ix_train_cv], Y_cut_str.iloc[ix_train_cv])

    RFC_model.fit(data_with_dummies.iloc[ix_train_cv], Y_cut_str.iloc[ix_train_cv])

    MLPC_model.fit(data_with_dummies.iloc[ix_train_cv], Y_cut_str.iloc[ix_train_cv])



    # Prédiction sur le jeu de test :

    y_test_knn = KNN_model.predict(data_with_dummies.iloc[ix_test_cv])

    y_test_rfc = RFC_model.predict(data_with_dummies.iloc[ix_test_cv])

    y_test_mlpc = MLPC_model.predict(data_with_dummies.iloc[ix_test_cv])



    # Calcul de l'accuracy :

    accuracy_knn.append(accuracy_score(y_test_knn, Y_cut_str.iloc[ix_test_cv]))

    accuracy_rfc.append(accuracy_score(y_test_rfc, Y_cut_str.iloc[ix_test_cv]))

    accuracy_mlpc.append(accuracy_score(y_test_mlpc, Y_cut_str.iloc[ix_test_cv]))

    

# Afficher la moyenne et l'écart type des accuracy :

print("Accuracy KNN  : %0.2f (+/- %0.2f)" % (np.mean(accuracy_knn), np.std(accuracy_knn) * 2))

print("Accuracy RFC  : %0.2f (+/- %0.2f)" % (np.mean(accuracy_rfc), np.std(accuracy_rfc) * 2))

print("Accuracy MLPC : %0.2f (+/- %0.2f)" % (np.mean(accuracy_mlpc), np.std(accuracy_mlpc) * 2))
# Entrainement et prédiction :

RFC_model.fit(data_with_dummies.loc[ix_train], Y_cut_str.loc[ix_train])

y_test_rfc = RFC_model.predict(data_with_dummies.loc[ix_test])

y_test_rfc = pd.Series(y_test_rfc, name='Predictions') # Transformation en série pour donner un nom à la colonne



print("On affiche la matrice de confusion sur le jeu de test :")

cross_table = pd.crosstab(y_test_rfc, Y_cut_str.loc[ix_test].reindex(ix_test).reset_index(drop=True))

cross_table = cross_table.iloc[[2,0,1],[2,0,1]] # On réordonne les colonnes pour qu'elles soient dans le 'bon' ordre

cross_table.style.background_gradient(cmap='Reds', axis=1, low=0, high=1)
from sklearn.linear_model import Lasso

from sklearn.ensemble import RandomForestRegressor

from sklearn.neural_network import MLPRegressor



# Initialisation des modèles :

LASSO_model = Lasso(alpha=.1, random_state=0)

RFR_model = RandomForestRegressor(n_estimators=500, max_depth=20, min_samples_leaf=7, random_state=0)

MLPR_model = MLPRegressor(hidden_layer_sizes=(500,), activation='relu', max_iter=2000, random_state=0)
# Entraînement :

%time LASSO_model.fit(data_with_dummies.loc[ix_train], Y.loc[ix_train])

%time RFR_model.fit(data_with_dummies.loc[ix_train], Y.loc[ix_train])

%time MLPR_model.fit(data_with_dummies.loc[ix_train], Y.loc[ix_train])



# Prédiction sur le jeu de test :

y_test_lasso = LASSO_model.predict(data_with_dummies.loc[ix_test])

y_test_rfr = RFR_model.predict(data_with_dummies.loc[ix_test])

y_test_mlpr = MLPR_model.predict(data_with_dummies.loc[ix_test])



# On affiche la précision (accuracy) et le rappel (recall) pour chaque méthode :

from sklearn.metrics import r2_score, mean_squared_error

print("\nLASSO :\n- R2   : {}\n- RMSE : {}".format(

    round(r2_score(y_test_lasso, Y[ix_test]),4),

    round(np.sqrt(mean_squared_error(y_test_lasso, Y[ix_test])),4)

))

print("\nRandom Forest Regression (500 arbres) :\n- R2   : {}\n- RMSE : {}".format(

    round(r2_score(y_test_rfr, Y[ix_test]),4),

    round(np.sqrt(mean_squared_error(y_test_rfr, Y[ix_test])),4)

))

print("\nMulti Layer Perceptron Regression (500 couches) :\n- R2   : {}\n- RMSE : {}".format(

    round(r2_score(y_test_mlpr, Y[ix_test]),4),

    round(np.sqrt(mean_squared_error(y_test_mlpr, Y[ix_test])),4)

))
from sklearn.model_selection import cross_validate



# Cross validation avec 5 folds : 

scores_lasso = cross_validate(LASSO_model, data_with_dummies, Y, cv=5, scoring=["r2","neg_mean_squared_error"])

scores_rfr = cross_validate(RFR_model, data_with_dummies, Y, cv=5, scoring=["r2","neg_mean_squared_error"])

scores_mlpr = cross_validate(MLPR_model, data_with_dummies, Y, cv=5, scoring=["r2","neg_mean_squared_error"])



print("Résultats pour la méthode LASSO :")

print(scores_lasso)

print('-'*30)



# Afficher la moyenne et l'écart type des accuracy :

print("\nR2 LASSO   : %0.2f (+/- %0.2f)" % (scores_lasso["test_r2"].mean(), scores_lasso["test_r2"].std() * 2))

print("RMSE LASSO : %0.2f (+/- %0.2f)" % (np.sqrt(scores_lasso["test_neg_mean_squared_error"]*-1).mean(), np.sqrt(scores_lasso["test_neg_mean_squared_error"]*-1).std() * 2))



print("\nR2 RFR     : %0.2f (+/- %0.2f)" % (scores_rfr["test_r2"].mean(), scores_rfr["test_r2"].std() * 2))

print("RMSE RFR   : %0.2f (+/- %0.2f)" % (np.sqrt(scores_rfr["test_neg_mean_squared_error"]*-1).mean(), np.sqrt(scores_rfr["test_neg_mean_squared_error"]*-1).std() * 2))



print("\nR2 MLPR    : %0.2f (+/- %0.2f)" % (scores_mlpr["test_r2"].mean(), scores_mlpr["test_r2"].std() * 2))

print("RMSE MLPR  : %0.2f (+/- %0.2f)" % (np.sqrt(scores_mlpr["test_neg_mean_squared_error"]*-1).mean(), np.sqrt(scores_mlpr["test_neg_mean_squared_error"]*-1).std() * 2))