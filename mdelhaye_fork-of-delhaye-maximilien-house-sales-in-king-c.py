# Directive pour afficher les graphiques dans Jupyter
%matplotlib inline
# Pandas : librairie de manipulation des données
# NumPy : librairie de calcul scientifique
# MatPlotLib : librairie de visualisation et graphiques
# SeaBorn : librairie de graphiques avancés
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# Lecture des données d'apprentissage et de test
dataset = pd.read_csv("../input/housesalesprediction/kc_house_data.csv")
# Récupération des premières lignes du dataset
dataset.head(10)
# Récupération des informations sur le dataset
dataset.info
# Affichage des maisons en fonction de la longitude et de la latitude
dataset.plot(kind="scatter", x="long", y="lat", c="price", cmap="rainbow", s=3, figsize=(12,12))
# Vue des statistiques sur le dataset
dataset.describe()
plt.scatter(dataset["sqft_living"], dataset["price"])
plt.xlabel("Taille de la propriété")
plt.ylabel("Prix de vente")
plt.show()
plot = dataset.groupby("grade")["price"].mean()
plt.scatter(np.unique(dataset["grade"]), plot)
plt.xlabel("Condition de la propriété (de 1 à 13)")
plt.ylabel("prix de vente moyen")
plt.show()
plot = dataset.groupby("yr_built")["price"].mean()
plt.scatter(np.unique(dataset["yr_built"]), plot)
plt.xlabel("Année")
plt.ylabel("prix de vente moyen")
plt.show()
plot = dataset.groupby("yr_built")["price"].count()
plt.scatter(np.unique(dataset["yr_built"]), plot)
condition = dataset['condition'].value_counts()

print("Nombre de logement par condition : ")
print(condition)

fig, ax = plt.subplots(ncols=2, figsize=(14,5))
sns.countplot(x='condition', data=dataset, ax=ax[0])
sns.boxplot(x='condition', y= 'price',
            data=dataset, ax=ax[1])
plt.show()
dataset.columns
# On retire les colonnes id, lat, long, et code postal ce qui permettra d'éviter des erreurs
houseSales = dataset.drop(['id','date','lat','long','zipcode'], axis=1)
tabcorr = houseSales.corr()
plt.figure(figsize=(12,12))
sns.heatmap(abs(tabcorr), cmap="coolwarm")
correlations = tabcorr.price
print(correlations)
correlations = correlations.drop(['price'], axis=0)
print(abs(correlations).sort_values(ascending=False))
continuous_features = ['sqft_living', 'sqft_above', 'sqft_lot', 'sqft_basement', 'sqft_living15', 'sqft_lot15']
discrete_features = ['grade', 'bathrooms', 'view', 'bedrooms', 'waterfront', 'floors', 'yr_renovated', 'yr_built', 'condition']
# Pour la régression linéaire, on se limite aux maisons de moins de 1M$, et on élimine les features discrètes :
houseSales2 = houseSales[houseSales.price<1000000].drop(discrete_features, axis=1)
from sklearn.model_selection import train_test_split

X = houseSales2.drop(['price'], axis=1)
Y = houseSales2.price
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1, random_state=1)
# On utilise la régression linéaire de sklearn
from sklearn.linear_model import LinearRegression

lr = LinearRegression()
lr.fit(X_train, Y_train)                     # Apprentissage
Y_pred = lr.predict(X_test)                  # Prédiction sur l'ensemble de test
# On trace le nuage de points pour comparer la prédiction et les résultats attendus :
plt.figure(figsize=(12,12))
plt.scatter(Y_test, Y_pred)
plt.plot([Y_test.min(),Y_test.max()],[Y_test.min(),Y_test.max()], color='red', linewidth=3)
plt.xlabel("Prix")
plt.ylabel("Prediction de prix")
plt.title("Prix reels vs predictions")
# On peut visualiser la distribution de l'erreur avec seaborn :
sns.distplot(Y_test-Y_pred)
# On calcule l'erreur sur les moindres carrés :
from sklearn.metrics import mean_squared_error, r2_score

print(np.sqrt(mean_squared_error(Y_test, Y_pred)))
# On calcule le score R2 (rapport des variances estimée/réelle) :
scoreR2 = r2_score(Y_test, Y_pred)
print(scoreR2)
X = houseSales2.drop(['price'], axis=1)
Y = houseSales2.price
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1, random_state=1)
from sklearn import ensemble

randomForest = ensemble.RandomForestRegressor()
randomForest.fit(X_train, Y_train)
Y_randomForest = randomForest.predict(X_test)
print(randomForest.score(X_test,Y_test))
# On trace le nuage de points pour comparer la prédiction et les résultats attendus :
plt.figure(figsize=(12,12))
plt.scatter(Y_test, Y_pred)
plt.plot([Y_test.min(),Y_test.max()],[Y_test.min(),Y_test.max()], color='red', linewidth=3)
plt.xlabel("Prix")
plt.ylabel("Prediction de prix")
plt.title("Prix reels vs predictions")
sns.distplot(Y_test-Y_randomForest)
print(np.sqrt(mean_squared_error(Y_test, Y_randomForest)))