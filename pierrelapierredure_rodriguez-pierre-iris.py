# BIBLIOTHEQUE

# Directive pour afficher les graphiques dans Jupyter
%matplotlib inline

# Pandas : librairie de manipulation de données
# NumPy : librairie de calcul scientifique
# MatPlotLib : librairie de visualisation et graphiques
# SeaBorn : librairie de graphiques avancés
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns


df = sns.load_dataset("iris")
df

# VISUALISATION DES DONNEES

sns.jointplot("sepal_length", "petal_width", df, kind='kde');

sns.boxplot(x="species", y="sepal_length", data=df)

sns.violinplot(x="species", y="sepal_width", data=df)

fig = sns.FacetGrid(df, hue="species", aspect=3, palette="Set2") # aspect=3 permet d'allonger le graphique
fig.map(sns.kdeplot, "sepal_width", shade=True)
fig.add_legend()

sns.lmplot(x="petal_width", y="petal_length", data=df, fit_reg=False, hue='species') #nuage de pts

#Remarque : on peut changer les paramètres de l'arbre pour avoir un entraînement différent

#ENTRAINEMENT DU MODELE + TEST

data_train = df.sample(frac=0.8, random_state=1)          # 80% des données avec frac=0.8
data_test = df.drop(data_train.index)     # le reste des données pour le test

X_train = data_train.drop(['species'], axis=1)
y_train = data_train['species']
X_test = data_test.drop(['species'], axis=1)
y_test = data_test['species']


plt.figure(figsize=(9,9)) #Notion de proba
logistique = lambda x: np.exp(x)/(1+np.exp(x))
x_range = np.linspace(-10,10,50)       
y_values = logistique(x_range)
plt.plot(x_range, y_values, color="red")

from sklearn.linear_model import LogisticRegression #Régression linéraire + entraînement
lr = LogisticRegression()
lr.fit(X_train,y_train)

y_lr = lr.predict(X_test) #Prédiction
#VERIFICATION DE L'EFFICACITE

from sklearn.metrics import accuracy_score, confusion_matrix

lr_score = accuracy_score(y_test, y_lr) #Pourcentage de réussite
print(lr_score)

cm = confusion_matrix(y_test, y_lr) # Matrice de confusion
print(cm)

#ENTRAINEMENT DU MODELE AVEC UN ARBRE DE CLASSIFICATION

from sklearn import tree

dtc = tree.DecisionTreeClassifier()
dtc.fit(X_train,y_train)
y_dtc = dtc.predict(X_test)

print(accuracy_score(y_test, y_dtc))# % de réussite

#Remarque : on peut changer les paramètres de l'arbre pour avoir un entraînement différent

