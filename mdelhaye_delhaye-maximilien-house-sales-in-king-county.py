# Directive pour afficher les graphiques dans Jupyter
%matplotlib inline
# Pandas : librairie de manipulation des données
# NumPy : librairie de calcul scientifique
# MatPlotLib : librairie de visualisation et graphiques
# SeaBorn : librairie de graphiques avancés
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
# Lecture des données d'apprentissage et de test
df = pd.read_csv("../input/housesalesprediction/kc_house_data.csv")
df.head().T
df.count()    # Comptage des valeurs par colonnes
plt.figure(figsize = (8, 5))
sns.jointplot(df.sqft_living, df.price, 
              alpha = 0.5)
plt.xlabel('Sqft Living')
plt.ylabel('Sale Price')
plt.show()
condition = df['condition'].value_counts()

print("Nombre de logement par condition : ")
print(condition)

fig, ax = plt.subplots(ncols=2, figsize=(14,5))
sns.countplot(x='condition', data=df, ax=ax[0])
sns.boxplot(x='condition', y= 'price',
            data=df, ax=ax[1])
plt.show()
fig = sns.FacetGrid(df, hue="condition", aspect=3, palette="Set1")
fig.map(sns.kdeplot, "yr_built", shade=True)
fig.add_legend()
fig = sns.FacetGrid(df, hue="grade", aspect=3, palette="Set1")
fig.map(sns.kdeplot, "yr_built", shade=True)
fig.add_legend()
df.columns
# On élimine les colonnes non pertinentes pour la prédiction
HouseSales = df.drop(['id','date','lat','long'], axis=1)
HouseSales.count()
print("Prix Min :", HouseSales['price'].min())
print("Prix Moyen :", HouseSales['price'].mean())
print("Prix Médiant :",HouseSales['price'].median())
print("Prix Max :", HouseSales['price'].max())
sns.distplot(HouseSales['price'], color='blue')
HouseSales['log_price'] = np.log(HouseSales['price']+1)
sns.distplot(HouseSales['log_price'], color="blue")
# HouseSales = HouseSales.drop(['price'], axis=1)
HouseSales[['yr_built', 'log_price']].describe()
sns.distplot(HouseSales.log_price, color='blue')
sns.distplot(HouseSales.yr_built, color='red')
# import de la librairie sklearn qui est une librairie de prétraitement des données
from sklearn import preprocessing
# normalisation des données pour ramener les valeurs min et max à 0 et 1
minmax = preprocessing.MinMaxScaler(feature_range=(0,1))
HouseSales[['yr_built', 'log_price']] = minmax.fit_transform(HouseSales[['yr_built', 'log_price']])
sns.distplot(HouseSales.log_price, color='blue')
sns.distplot(HouseSales.yr_built, color='red')
# normalisation des données pour ramener les valeurs ecart type et moyenne à 0 et 1
scaler = preprocessing.StandardScaler()
HouseSales[['yr_built', 'log_price']] = scaler.fit_transform(HouseSales[['yr_built', 'log_price']])
sns.distplot(HouseSales.log_price, color='blue')
sns.distplot(HouseSales.yr_built, color='red')
HouseSales.info()
X = HouseSales.drop(['price'], axis = 1)
Y = HouseSales['price']
# Séparation du dataset en deux parties :
from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=1)
print(X_train.shape)
print(Y_test.shape)
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(X_train,Y_train)
Y_lr = lr.predict(X_test)
# Importation des méthodes de mesure de performances
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, roc_auc_score,auc, accuracy_score
# Affichage de la matrice de confusion :
print(confusion_matrix(Y_test, Y_lr))
# Affichage du score de performance
print(accuracy_score(Y_test, Y_lr))
# Affichage du rapport de classification
print(classification_report(Y_test, Y_lr))



