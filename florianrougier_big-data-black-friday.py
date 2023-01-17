# Importation de la librairie warnings et désactivation des avertissements (pour rendre les sorties code plus lisible et éviter les avertissements de version de librairies)
import warnings
warnings.filterwarnings('ignore')

# Importation des librairies pandas et numpy permettant la manipulation des données et des tableaux multidimensionnels 
import pandas as pd
import numpy as np

# Librairies de visualisation permettant de tracer différents graphes.
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline

# Paramètrage des visualisations
sns.set(style='darkgrid')
plt.rcParams["patch.force_edgecolor"] = True
# Nous devons dans un premier temps charger les données depuis notre dataset. Le dataset contient 2 fichiers 'train.csv'
# et 'test'.csv' contenant respectivement plus de 550 000 entrées et plus de 234 000 entrées.

# Nous chargons donc les données dans un tableau de données.

# Train set
train = pd.read_csv('../input/train.csv')
# Test set
test = pd.read_csv('../input/test.csv')
# Regardons les 5 premières entrées ainsi que les 5 dernières pour se donner une idée de la forme que prenne les entrées.

# Ensemble d'entrainement:

# 5 premières lignes pour l'ensemble d'entrainement:
train.head(5)
# 5 dernières lignes pour l'ensemble d'entrainement:
train.tail(5)
# Visualisons également les types des données d'entrée ainsi que la taille exacte de l'ensemble de données.
print(train.info())
print('Shape: ',train.shape)
# Explication ici: tailles des ensembles de données.
# Nombre de colonnes pour l'ensemble train: 12 => target(price) est dans l'ensemble; nous nous en servirons pour entrainer l'algorithme
# A l'inverse, 11 colomnes pour l'ensemble train (pas la target). Nous nous en servirons pour tester les performances de l'algorithme sur plusieurs modèles de régression
# Parler également du type des variables et de la signification de chacune des colomnes
# Ensemble de test:

# 5 premières lignes pour l'ensemble de test:
test.head(5)
# 5 dernières lignes pour l'ensemble de test:
test.tail(5)
# Visualisons également les types des données d'entrée ainsi que la taille exacte.
print(test.info())
print('Shape: ',test.shape)
# Examinons de plus près les valeurs manquantes des ensemble des données

# Ensemble d'apprentissage

total_miss = train.isnull().sum()
perc_miss = total_miss/train.isnull().count()*100

missing_data = pd.DataFrame({'Total missing':total_miss,
                            '% missing':perc_miss})

missing_data.sort_values(by='Total missing',
                         ascending=False).head(3)
# Ensemble de test

total_miss = test.isnull().sum()
perc_miss = total_miss/test.isnull().count()*100

missing_data = pd.DataFrame({'Total missing':total_miss,
                            '% missing':perc_miss})

missing_data.sort_values(by='Total missing',
                         ascending=False).head(3)
# Nous devons maintenant nous occuper de ces données manquantes.
# Plusieurs choix s'offrent à nous. Nous pourrions simplement supprimer les lignes possèdant une valeur manquante.
# Cependant, ce choix apparait ici très mauvais car près de 70% des donnnées sont manquantes pour la colomne 'Product_Category_3'.
# Compte tenu du fort pourcentage de données manquante pour cette colomne, nous choisissons de la supprimer de l'ensemble d'apprentissage et de l'ensemble de test.
half_count_train = len(train) / 2 # Cette ligne est un seuil
half_count_test = len(test) / 2 # Cette ligne est un seuil

train = train.dropna(thresh=half_count_train,axis=1) # Cette ligne supprime les colomne comportant plus de 50% de données manquantes
test = test.dropna(thresh=half_count_test,axis=1) # Cette ligne supprime les colomne comportant plus de 50% de données manquantes
# Regardons de nouveau un échantillon de nos ensembles.
train.head()
test.head()
print(train.Product_Category_2)
print(train.Product_Category_2.min())
print(train.Product_Category_2.max())
print(test.Product_Category_2.min())
print(test.Product_Category_2.max())
# Commencons par calculer cette moyenne
avg_train = train.Product_Category_2.mean()
avg_test = test.Product_Category_2.mean()

print(avg_train)
print(avg_test)
# Remplacons les valeurs manquantes de l'ensemble d'apprentissage et de l'ensemble de test par les moyennes repectives 
train['Product_Category_2'].fillna(avg_train, inplace=True)
test['Product_Category_2'].fillna(avg_test, inplace=True)
print(train.Product_Category_2)
print(test.Product_Category_2)
# Train set
unique_users = len(train.User_ID.unique())
unique_products = len(train.Product_ID.unique())
print('There are {} unique users and {} unique products in the train set'.format(unique_users, unique_products))
# Test set
unique_users = len(test.User_ID.unique())
unique_products = len(test.Product_ID.unique())
print('There are {} unique users and {} unique products in the test set'.format(unique_users, unique_products))
# Drop the columns Product_Id and User_Id in the train and test set
train = train.drop(columns="User_ID")
train = train.drop(columns="Product_ID")
test = test.drop(columns="User_ID")
test = test.drop(columns="Product_ID")
for col_name in ['Gender', 'Age', 'Occupation', 'City_Category']:
    print(sorted(train[col_name].unique()))
    print(sorted(test[col_name].unique()))
train['Marital_Status'].unique()
train['Stay_In_Current_City_Years'].unique()
# Train set
# We first need to convert the '4+' value to simply '4'
train['Stay_In_Current_City_Years'] = [x.strip().replace('4+', '4') for x in train['Stay_In_Current_City_Years']]
# Then we need to replace the object type values by integers                                                                                       
train['Stay_In_Current_City_Years'] = train['Stay_In_Current_City_Years'].astype(str).astype(int)
print(train['Stay_In_Current_City_Years'])

# Test set
test['Stay_In_Current_City_Years'] = [x.strip().replace('4+', '4') for x in test['Stay_In_Current_City_Years']]
# Then we need to replace the object type values by integers                                                                                       
test['Stay_In_Current_City_Years'] = test['Stay_In_Current_City_Years'].astype(str).astype(int)
print(test['Stay_In_Current_City_Years'])
# Drop the columns Product_Category_1 and Product_Category_2 in the train and test set
train = train.drop(columns="Product_Category_1")
train = train.drop(columns="Product_Category_2")
test = test.drop(columns="Product_Category_1")
test = test.drop(columns="Product_Category_2")
for col_name in train.columns:
    print(col_name, len(train[col_name].unique()))
fig, axes = plt.subplots(nrows=3, ncols=2, figsize=[15, 15])

train['Gender'].value_counts().plot(kind='barh', ax=axes[0,0], title='Gender')
train['Age'].value_counts().plot(kind='barh', ax=axes[0,1], title='Age')
train['City_Category'].value_counts().plot(kind='barh', ax=axes[1,0], title='City_Category')
train['Marital_Status'].value_counts().plot(kind='barh', ax=axes[1,1], title='Marital_Status')
train['Occupation'].value_counts().plot(kind='barh', ax=axes[2,0], title='Occupation')
train['Stay_In_Current_City_Years'].value_counts().plot(kind='barh', ax=axes[2,1], title='Stay_In_Current_City_Years')
for col_name in test.columns:
    print(col_name, len(test[col_name].unique()))
fig, axes = plt.subplots(nrows=3, ncols=2, figsize=[15, 15])

test['Gender'].value_counts().plot(kind='barh', ax=axes[0,0], title='Gender')
test['Age'].value_counts().plot(kind='barh', ax=axes[0,1], title='Age')
test['City_Category'].value_counts().plot(kind='barh', ax=axes[1,0], title='City_Category')
test['Marital_Status'].value_counts().plot(kind='barh', ax=axes[1,1], title='Marital_Status')
test['Occupation'].value_counts().plot(kind='barh', ax=axes[2,0], title='Occupation')
test['Stay_In_Current_City_Years'].value_counts().plot(kind='barh', ax=axes[2,1], title='Stay_In_Current_City_Years')
# Rappelons que les variables catégoriques sont les suivantes: Age, Gender, Occupation, Ciy_Category, Marital_Status, Product Category 1, ¨Product Category 2
train.dtypes
# Train set
# Nous remplacons tout d'abord les valeurs de la colonne 'Gender' initialement égales à 'F' par 0 et 'M' par 1
# Cette opération n'est pas nécessaire pour la colonne 'Marital_Status' puisque ces valeurs sont déjà sous cette forme
train.Gender = np.where(str(train.Gender)=='M',1,0) # Femelle: 0, Male: 1
test.Gender = np.where(str(test.Gender)=='M',1,0) # Femelle: 0, Male: 1

# Nous encodons ensuites les variables catégoriques (Age, City_Category, Occupation)
train_Age = pd.get_dummies(train.Age)
train_CC = pd.get_dummies(train.City_Category)
train_Occup = pd.get_dummies(train.Occupation)
test_Age = pd.get_dummies(test.Age)
test_CC = pd.get_dummies(test.City_Category)
test_Occup = pd.get_dummies(test.Occupation)

# Et nous remplacons les variables catégoriques initiales par les variables encodées
train_encoded = pd.concat([train_Age,train_CC,train_Occup,train],axis=1)
train_encoded.drop(['Age','City_Category','Occupation'],axis=1,inplace=True)
test_encoded = pd.concat([test_Age,test_CC,test_Occup,test],axis=1)
test_encoded.drop(['Age','City_Category','Occupation'],axis=1,inplace=True)
print(train_encoded)
print(test_encoded)
X_train = train_encoded.iloc[:, :-1].values # Prend toutes les colonnes sauf la dernière (variable cible)
y_train = train_encoded.iloc[:,-1].values # Prend la dernière colonne (variable cible)

# Pour présire les nouveaux résultats
X_test = test_encoded # Prend toutes les colonnes sauf la dernière (variable cible)
# Fitting Random Forest Regression to the dataset
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 10, random_state = 0) # nous utilisons une foret de 10 arbres décisionnels, random_state supprime la variabilité des résultats
regressor.fit(X_train, y_train)
# Predict new results on the test set
prediction = np.round(regressor.predict(X_test))
print(prediction)
# Reshape prediction vector into a 2D matrix
prediction = prediction.reshape(len(prediction), 1)

# concatenate X_test with prediction 
dataTest = np.concatenate((X_test, prediction), axis = 1)
print(dataTest)
# Feature importance 
f_im = regressor.feature_importances_.round(3)
ser_rank = pd.Series(f_im,index=test_encoded.columns).sort_values(ascending=False)

plt.figure()
sns.barplot(y=ser_rank.index,x=ser_rank.values,palette='deep')
plt.xlabel('relative importance')
# Options pour afficher davantage de colonnes en sortie
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

# Option pour avoir des résultats consistants
np.random.seed(0)

# Pour faciliter la manipulation des données, repartons de l'ensemble de test duquel nous avons supprimé les colonnes
# inutiles à notre analyse. Ajoutons simplement une colonne prediction à cet ensemble.
test['prediction'] = prediction
# Nous prenons 10 lignes de l'ensemble de test et les ordonnons selon la valeur de la colonne prédiction 
# (par ordre décroissant)
print(test.sample(10).sort_values(by='prediction',ascending=False))
