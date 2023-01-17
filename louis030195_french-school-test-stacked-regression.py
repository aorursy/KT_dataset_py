# Manipulation des vecteurs
import numpy as np

# Manipulation des matrices, données non numeriques ...
import pandas as pd

# Algorithmes
from sklearn.linear_model import Lasso, ElasticNet
import lightgbm as lgb
from sklearn.kernel_ridge import KernelRidge
from xgboost import XGBRegressor
import xgboost as xgb
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone

# Arbres
from sklearn.tree import DecisionTreeRegressor

# Ensembles
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.ensemble import BaggingRegressor
from sklearn.ensemble import AdaBoostRegressor

# Superposition de modèles
from sklearn.pipeline import make_pipeline

# Préparation
from sklearn.preprocessing import RobustScaler

# Statistiques
from scipy import stats
from scipy.stats import norm, skew
from scipy.special import boxcox1p

# Réduction de dimensions
from sklearn.decomposition import PCA

# Clustering
from sklearn.cluster import KMeans

# Réseau de neuronnes
import tensorflow as tf

# Metriques
from sklearn.metrics import mean_absolute_error, make_scorer, r2_score, roc_auc_score

# Séparation des données, validation croisée
from sklearn.model_selection import KFold, cross_val_score, train_test_split, RandomizedSearchCV, GridSearchCV, StratifiedKFold

# Utilitaires
from datetime import datetime


# Visualisation
import matplotlib.pyplot as plt
from pandas.tools.plotting import scatter_matrix
from xgboost import plot_importance

# Gestion des fichiers
import os
# Lecture du fichier contenant le jeu d'entrainement
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
# Stockage de la colonne Id
train_ID = train['Id']
test_ID = test['Id']

# Nous retirons la colonne Id qui n'est pas utile pour l'entrainement
train.drop("Id", axis = 1, inplace = True)
test.drop("Id", axis = 1, inplace = True)
train.head()
test.head()
# Voyons voir le prix des maisons ...

# Fonction utile pour afficher la mediane, moyenne, écart-type ...
print(train.SalePrice.describe())
plt.hist(train.SalePrice)
plt.title('A quel point le prix des maisons est-il variable ?')
plt.xlabel('Prix maison')
plt.xlim(0, np.mean(train.SalePrice) * 2.5)
plt.show()
plt.scatter(train.SalePrice, train.GrLivArea)
plt.title('Distribution des prix en fonction de la surface')
plt.xlabel('Prix')
plt.ylabel('Surface')
plt.show()
# Nous voyons clairement des données abbérantes
train = train.drop(train[(train['GrLivArea']>4000) & (train['SalePrice']<300000)].index)
plt.scatter(train.OverallQual, train.SalePrice)
plt.title('Distribution des prix en fonction de la qualité des matériaux')
plt.ylabel('Prix')
plt.xlabel('Qualité des matériaux')
plt.show()
# Pareil ici
train = train.drop(train[(train['OverallQual']<5) & (train['SalePrice']>200000)].index)
train.info()
def plot_corr(df,size=10):
    '''
    Fonction pour afficher une matrice de corrélation 
    pour chaques pair de colonnes dans le dataframe.

    Paramètres:
        df: pandas DataFrame
        size: Taille horizontale et vertical du graphique
    '''

    corr = df.corr()
    fig, ax = plt.subplots(figsize=(size, size))
    ax.matshow(corr)
    plt.xticks(range(len(corr.columns)), corr.columns);
    plt.yticks(range(len(corr.columns)), corr.columns);
# Je me permet de retirer quelques colonnes pour l'affichage, ces colonnes ont des données manquantes donc sont moins intéréssante à visualiser,
plot_corr(train.drop(['FireplaceQu', 'PoolQC', 'Fence', 'MiscFeature'], axis=1), 20)
train.drop(['FireplaceQu', 'PoolQC', 'Fence', 'MiscFeature'], axis=1).corr().sort_values('SalePrice', ascending=True, axis=1)
# Stockage des shapes des jeux de données avant de les assembler en une seule matrice
# Cela nous permettra d'appliquer les transformations des données sur les deux jeux de données en même temps
train_shape = train.shape[0]
test_shape = test.shape[0]
# Stockage de la colonne à prédire dans la variable y avec une transformation pour avoir une distribution logarithmique
y_train = np.log1p(train["SalePrice"])
total_data = pd.concat((train, test), sort=True).reset_index(drop=True)
total_data.drop(['SalePrice'], axis=1, inplace=True)
# Affichage des colonnes ayant des valeurs nulles et le pourcentage de null par rapport au nombre total de lignes
total = total_data.isnull().sum().sort_values(ascending=False)
pourcentage = (total_data.isnull().sum()/total_data.isnull().count()).sort_values(ascending=False)
donnees_manquantes = pd.concat([total, pourcentage], axis=1, keys=['Total', 'Pourcentage'])
donnees_manquantes.head(20)
# Utilities semble particulière
total_data.Utilities.describe()
# Optimisation des lignes ayant des valeurs nulles
total_data.loc[2418, 'PoolQC'] = 'Fa'
total_data.loc[2501, 'PoolQC'] = 'Gd'
total_data.loc[2597, 'PoolQC'] = 'Fa'
total_data.loc[332, 'BsmtFinType2'] = 'ALQ'
total_data.loc[947, 'BsmtExposure'] = 'No' 
total_data.loc[1485, 'BsmtExposure'] = 'No'
total_data.loc[2038, 'BsmtCond'] = 'TA'
total_data.loc[2183, 'BsmtCond'] = 'TA'
total_data.loc[2215, 'BsmtQual'] = 'Po' 
total_data.loc[2216, 'BsmtQual'] = 'Fa'
total_data.loc[2346, 'BsmtExposure'] = 'No'
total_data.loc[2522, 'BsmtCond'] = 'Gd' 
total_data['LotFrontage'] = total_data.groupby('Neighborhood')['LotFrontage'].transform(lambda x: x.fillna(x.median()))
# Ces colonnes a valeurs numériques sont en fait des données catégorique.
# Les transformer en données catégorique pourrait nous faire gagner de la performance
total_data['MSSubClass'] = total_data['MSSubClass'].apply(str)
total_data['OverallCond'] = total_data['OverallCond'].astype(str)
total_data['YrSold'] = total_data['YrSold'].astype(str)
total_data['MoSold'] = total_data['MoSold'].astype(str)
# Nous remplissions les valeurs nulles avec la mediane de la colonne
total_data['LotFrontage'] = total_data['LotFrontage'].fillna(total_data['LotFrontage'].median())
total_data['MasVnrArea'] = total_data['MasVnrArea'].fillna(total_data['MasVnrArea'].median())

# Les valeurs catégoriques nulles sont remplies avec "mode()", le mode c'est comme la mediane mais pour les valeurs catégorique
for col in ('GarageType', 'GarageFinish', 'GarageQual','GarageCond'):
    total_data[col] = total_data[col].fillna(total_data[col].mode()[0])
    
for col in ('BsmtFinType2', 'BsmtExposure', 'BsmtCond','BsmtFinType1','BsmtQual'):
    total_data[col] = total_data[col].fillna(total_data[col].mode()[0])

total_data['Electrical'] = total_data['Electrical'].fillna(total_data['Electrical'].mode()[0])
total_data['MSZoning'] = total_data['MSZoning'].fillna(total_data['MSZoning'].mode()[0])
total_data['KitchenQual'] = total_data['KitchenQual'].fillna(total_data['KitchenQual'].mode()[0])
total_data['Exterior1st'] = total_data['Exterior1st'].fillna(total_data['Exterior1st'].mode()[0])
total_data['Exterior2nd'] = total_data['Exterior2nd'].fillna(total_data['Exterior2nd'].mode()[0])
total_data['SaleType'] = total_data['SaleType'].fillna(total_data['SaleType'].mode()[0])

# Ces informations sont importante, nous ne pouvons pas les remplir avec une valeur par défaut
# Pour le cas de PoolQC par exemple si une valeur est nulle, cela veut dire que la maison n'a pas de piscine
# Donc nous remplissons avec None
total_data['PoolQC'] = total_data['PoolQC'].fillna('None')
total_data['MiscFeature'] = total_data['MiscFeature'].fillna('None')
total_data['Alley'] = total_data['Alley'].fillna('None')
total_data['Fence'] = total_data['Fence'].fillna('None')
total_data['FireplaceQu'] = total_data['FireplaceQu'].fillna('None')
total_data["MasVnrType"] = total_data["MasVnrType"].fillna('None')

# Pareil ici, mais là c'est déjà une valeur numérique donc 0 au lieu de None
for col in ('BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF','TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath'):
    total_data[col] = total_data[col].fillna(0)
    
for col in ('GarageYrBlt', 'GarageArea', 'GarageCars'):
    total_data[col] = total_data[col].fillna(0)

# Venant de la description du jeu de données "(Assume typical unless deductions are warranted)" donc NaN = Typical
total_data["Functional"] = total_data["Functional"].fillna('Typ')
total_data['TotalBsmtSF'] = total_data['TotalBsmtSF'].apply(lambda x: np.exp(6) if x <= 0.0 else x)
total_data['2ndFlrSF'] = total_data['2ndFlrSF'].apply(lambda x: np.exp(6.5) if x <= 0.0 else x)
total_data['GarageArea'] = total_data['GarageArea'].apply(lambda x: np.exp(6) if x <= 0.0 else x)
total_data['GarageCars'] = total_data['GarageCars'].apply(lambda x: 0 if x <= 0.0 else x)
total_data['LotFrontage'] = total_data['LotFrontage'].apply(lambda x: np.exp(4.2) if x <= 0.0 else x)
total_data['MasVnrArea'] = total_data['MasVnrArea'].apply(lambda x: np.exp(4) if x <= 0.0 else x)
total_data['BsmtFinSF1'] = total_data['BsmtFinSF1'].apply(lambda x: np.exp(6.5) if x <= 0.0 else x)
# Utilities contient 99% de AllPub, cela n'apportera aucune informations utiles à l'algorithme
total_data = total_data.drop(['Utilities', 'Street'], axis=1)

# La taille d'une maison semble influencer beaucoup le prix, donc, on peut imaginer
# que la création de nouvelles colonnes liées à la taille pourrait améliorer les performances du modèle
# ainsi, on créer une colonne contenant la surface totale de la maison
total_data['Total_sqr_footage'] = (total_data['BsmtFinSF1'] + total_data['BsmtFinSF2'] +
                                 total_data['1stFlrSF'] + total_data['2ndFlrSF'])

total_data['Total_Bathrooms'] = (total_data['FullBath'] + (0.5*total_data['HalfBath']) + 
                               total_data['BsmtFullBath'] + (0.5*total_data['BsmtHalfBath']))

total_data['Total_porch_sf'] = (total_data['OpenPorchSF'] + total_data['3SsnPorch'] +
                              total_data['EnclosedPorch'] + total_data['ScreenPorch'] +
                             total_data['WoodDeckSF'])
# On vérifie que notre jeu de données de test n'a plus de valeurs nulles
total = total_data.isnull().sum().sort_values(ascending=False)
pourcentage = (total_data.isnull().sum()/total_data.isnull().count()).sort_values(ascending=False)
donnees_manquantes = pd.concat([total, pourcentage], axis=1, keys=['Total', 'Pourcentage'])
donnees_manquantes.head(20)
# Gestion des données asymétriques
colonnes_numeriques = total_data.dtypes[total_data.dtypes != "object"].index

colonnes_asymetriques = total_data[colonnes_numeriques].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)
asymetrie = pd.DataFrame({'Asymetrie' :colonnes_asymetriques})
asymetrie.head(10)
asymetrie = asymetrie[abs(asymetrie) > 0.75]
print("Il y a " + str(asymetrie.shape[0]) + " données numériques asymétriques")
colonnes_asymetriques = asymetrie.index
lam = 0.15
for col in colonnes_asymetriques:
    total_data[col] = boxcox1p(total_data[col], lam)
# Encodage des valeurs catégoriques en valeurs numériques car un algorithme ne traite seulement les valeurs numériques
total_data = pd.get_dummies(total_data)
# On réassigne chaques jeux de données après avoir fait le pre-processing
train = total_data[:train_shape]
test = total_data[train_shape:]
# Après avoir esssayé PCA avec plusieurs différents hyperparamètres (2,3,50,100,150,200 ...), conclusion le PCA n'améliore pas les performances
pca = PCA(n_components=200)
train_pca = pca.fit_transform(train)
# Fonction de validation
def rmsle_cv(model):
    kf = KFold(5, shuffle=True, random_state=42).get_n_splits(train)
    rmse= np.sqrt(-cross_val_score(model, train, y_train, scoring="neg_mean_squared_error", cv = kf))
    return(rmse)
# Superposition des modèles les plus performant pour la régression
lasso = make_pipeline(RobustScaler(), Lasso(alpha =0.0005, random_state=1))
ENet = make_pipeline(RobustScaler(), ElasticNet(alpha=0.0005, l1_ratio=.9, random_state=3))
KRR = make_pipeline(RobustScaler(), KernelRidge(alpha=0.6, kernel='polynomial', degree=2, coef0=2.5))
GBoost = make_pipeline(RobustScaler(), GradientBoostingRegressor(n_estimators=3000, learning_rate=0.05,
                                   max_depth=4, max_features='sqrt',
                                   min_samples_leaf=15, min_samples_split=10, 
                                   loss='huber', random_state =5))
model_xgb = make_pipeline(RobustScaler(), xgb.XGBRegressor(colsample_bytree=0.4603, gamma=0.0468, 
                             learning_rate=0.05, max_depth=3, 
                             min_child_weight=1.7817, n_estimators=2200,
                             reg_alpha=0.4640, reg_lambda=0.8571,
                             subsample=0.5213, silent=1,
                             random_state =7, nthread = -1))
model_lgb = make_pipeline(RobustScaler(), lgb.LGBMRegressor(objective='regression',num_leaves=5,
                              learning_rate=0.05, n_estimators=720,
                              max_bin = 55, bagging_fraction = 0.8,
                              bagging_freq = 5, feature_fraction = 0.2319,
                              feature_fraction_seed=9, bagging_seed=9,
                              min_data_in_leaf =6, min_sum_hessian_in_leaf = 11))
# Voyons voir les performances individuelles
scores = [rmsle_cv(lasso).mean(), rmsle_cv(ENet).mean(), rmsle_cv(GBoost).mean(), rmsle_cv(model_xgb).mean(), rmsle_cv(model_lgb).mean(), rmsle_cv(KRR).mean()]
plt.plot(['lasso','ENet','GBoost','xgb','lgb','KRR'], scores)
# Classe pour calculer la moyenne des modèles
class AveragingModels(BaseEstimator, RegressorMixin, TransformerMixin):
    def __init__(self, models):
        self.models = models
        
    # we define clones of the original models to fit the data in
    def fit(self, X, y):
        self.models_ = [clone(x) for x in self.models]
        
        # Train cloned base models
        for model in self.models_:
            model.fit(X, y)

        return self
    
    #Now we do the predictions for cloned models and average them
    def predict(self, X):
        predictions = np.column_stack([
            model.predict(X) for model in self.models_
        ])
        return np.mean(predictions, axis=1)   
averaged_models = AveragingModels(models = (ENet, GBoost, KRR, lasso, model_xgb, model_lgb))

score = rmsle_cv(averaged_models)
print(" Averaged base models score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
averaged_models.fit(train, y_train)
predictions = np.expm1(averaged_models.predict(test))
soumission = pd.DataFrame({'Id': test_ID, 'SalePrice': predictions})
soumission.to_csv('soumission.csv', index=False)
# Nous pouvons essayer de former des groupes grâce au clustering KMeans
cluster = X_train.copy()
cluster['SalePrice'] = y_train
# Reprenons notre algorithme de gradient boost
model = XGBRegressor()
model.fit(X_train, y_train)

# La librairie de XGB propose une fonction pour afficher l'importance des différentes colonnes
plot_importance(model, max_num_features=10)
plt.show()
for i in range(2, 4):
  print(str(i) + " clusters")
  kmeans1 = KMeans(n_clusters=i, random_state=42).fit(cluster)
  y_kmeans1 = kmeans1.predict(cluster)
  plt.figure(1)
  plt.subplot(211)
  plt.scatter(cluster.BsmtFinSF1, cluster.SalePrice, c=y_kmeans1, s=50, cmap='viridis')
  plt.xlabel('Type 1 finished square feet')
  plt.ylabel('SalePrice')

  plt.subplot(212)
  kmeans2 = KMeans(n_clusters=i, random_state=42).fit(cluster)
  y_kmeans2 = kmeans2.predict(cluster)
  plt.scatter(cluster.GrLivArea, cluster.SalePrice, c=y_kmeans2, s=50, cmap='viridis')
  plt.xlabel('Above ground living area square feet')
  plt.ylabel('SalePrice')
  
  plt.show()
for i in range(2, 4):
  print(str(i) + " clusters")
  kmeans1 = KMeans(n_clusters=i, random_state=42).fit(cluster)
  y_kmeans1 = kmeans1.predict(cluster)
  plt.figure(1)
  plt.subplot(211)
  plt.scatter(cluster.TotalBsmtSF, cluster.SalePrice, c=y_kmeans1, s=50, cmap='viridis')
  plt.xlabel('Total square feet of basement area')
  plt.ylabel('SalePrice')

  plt.subplot(212)
  kmeans2 = KMeans(n_clusters=i, random_state=42).fit(cluster)
  y_kmeans2 = kmeans2.predict(cluster)
  plt.scatter(cluster.OverallQual, cluster.SalePrice, c=y_kmeans2, s=50, cmap='viridis')
  plt.xlabel('Rates the overall material and finish of the house')
  plt.ylabel('SalePrice')
  
  plt.show()
plt.hist(cluster.SalePrice)
plt.xlabel('Prix')
plt.ylabel('Nombre de maisons')


plt.show()

# Reprenons notre algorithme de gradient boost
model = XGBRegressor()
model.fit(train, y_train)

# La librairie de XGB propose une fonction pour afficher l'importance des différentes colonnes
plot_importance(model, max_num_features=15)
plt.show()
# On peut identifier quelques types de maisons particulieres
print('Nombre de maisons avec cheminée', 100 - np.mean(cluster.FireplaceQu_None) * 100, "%")
print('Nombre de maisons avec piscine', 100 - np.mean(cluster.PoolQC_None) * 100, "%")
print('Nombre de maisons avec grillage', 100 - np.mean(cluster.Fence_None) * 100, "%")
print('Nombre de maisons avec allée', 100 - np.mean(cluster.Alley_None) * 100, "%")
print('Nombre de maisons avec terrain de tennis', np.mean(cluster.MiscFeature_TenC) * 100, "%")