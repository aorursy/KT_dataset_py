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
df = pd.read_csv('../input/breast-cancer-wisconsin-data/data.csv')
df.head(10)
df.columns
from IPython.core.display import HTML # permet d'afficher du code html dans jupyter

display(HTML(df.head(10).to_html()))
df.shape
df.describe()
df.columns
df = df.drop(['Unnamed: 32'], axis=1)
df.diagnosis.value_counts()
malin = df.diagnosis=='M'

benin = df.diagnosis=='B'
sns.jointplot("perimeter_worst", "area_worst", df, kind='kde');
plt.figure(figsize=(12,12))

sns.kdeplot(df.perimeter_worst, df.area_worst,  shade=True)
plt.figure(figsize=(12,12))

sns.kdeplot(df[malin].perimeter_worst, df[malin].area_worst, cmap="Reds",  shade=True, alpha=0.3, shade_lowest=False)

sns.kdeplot(df[benin].perimeter_worst, df[benin].area_worst, cmap="Greens", shade=True, alpha=0.3, shade_lowest=False)
sns.boxplot(x="diagnosis", y="perimeter_worst", data=df)
sns.violinplot(x="diagnosis", y="perimeter_worst", data=df)
fig = sns.FacetGrid(df, hue="diagnosis", aspect=3, palette="Set2") # aspect=3 permet d'allonger le graphique

fig.map(sns.kdeplot, "perimeter_worst", shade=True)

fig.add_legend()
sns.lmplot(x="radius_mean", y="texture_mean", data=df, fit_reg=False, hue='diagnosis')
#sns.pairplot(df, hue="diagnosis")
data_train = df.sample(frac=0.8, random_state=1)          # 80% des données avec frac=0.8

data_test = df.drop(data_train.index)     # le reste des données pour le test
X_train = data_train.drop(['diagnosis'], axis=1)

y_train = data_train['diagnosis']

X_test = data_test.drop(['diagnosis'], axis=1)

y_test = data_test['diagnosis']
plt.figure(figsize=(9,9))



logistique = lambda x: np.exp(x)/(1+np.exp(x))   



x_range = np.linspace(-10,10,50)       

y_values = logistique(x_range)



plt.plot(x_range, y_values, color="red")
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()

lr.fit(X_train,y_train)
y_lr = lr.predict(X_test)
from sklearn.metrics import accuracy_score, confusion_matrix
lr_score = accuracy_score(y_test, y_lr)

print(lr_score)
# Matrice de confusion

cm = confusion_matrix(y_test, y_lr)

print(cm)
pd.crosstab(y_test, y_lr, rownames=['Reel'], colnames=['Prediction'], margins=True)
fig = sns.FacetGrid(df, hue="diagnosis", aspect=3) # aspect=3 permet d'allonger le graphique

fig.map(sns.kdeplot, "perimeter_worst", shade=True)

fig.add_legend()
fig = sns.FacetGrid(df[df.perimeter_worst>110], hue="diagnosis", aspect=3) # aspect=3 permet d'allonger le graphique

fig.map(sns.kdeplot, "texture_mean", shade=True)

fig.add_legend()
fig = sns.FacetGrid(df[(df.perimeter_worst>110) & (df.texture_mean>17)], hue="diagnosis", aspect=3) # aspect=3 permet d'allonger le graphique

fig.map(sns.kdeplot, "concave points_mean", shade=True)

fig.add_legend()
from sklearn import tree

dtc = tree.DecisionTreeClassifier()

dtc.fit(X_train,y_train)

y_dtc = dtc.predict(X_test)

print(accuracy_score(y_test, y_dtc))
plt.figure(figsize=(30,30))

tree.plot_tree(dtc, feature_names=X_train.columns, class_names=['benin','malin'], fontsize=14, filled=True)  
dtc1 = tree.DecisionTreeClassifier(max_depth = 3, min_samples_leaf = 20)

dtc1.fit(X_train,y_train)
plt.figure(figsize=(30,30))

tree.plot_tree(dtc1, feature_names=X_train.columns, class_names=['benin','malin'], fontsize=14, filled=True)  
y_dtc1 = dtc1.predict(X_test)

print(accuracy_score(y_test, y_dtc1))
from sklearn import ensemble

rf = ensemble.RandomForestClassifier()

rf.fit(X_train, y_train)

y_rf = rf.predict(X_test)
rf_score = accuracy_score(y_test, y_rf)

print(rf_score)
pd.crosstab(y_test, y_rf, rownames=['Reel'], colnames=['Prediction'], margins=True)
importances = rf.feature_importances_

indices = np.argsort(importances)
#Chargement du dataset contenant les données

df = sns.load_dataset("iris")
#Affichage éléments du dataset (5 par défaut)

df.head()
#Afficher les nuages de points associés aux paramètres

sns.pairplot(df, hue="species")
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



from IPython.core.display import HTML # permet d'afficher du code html dans jupyter

from sklearn.metrics import accuracy_score, confusion_matrix
data_train = df.sample(frac=0.8, random_state=1)  # 80% des données

data_test = df.drop(data_train.index) # 20% des données restantes
X_train = data_train.drop(['species'], axis=1) # Ce sur quoi l'on apprend (Caractéristique)

y_train = data_train['species'] # La cible (Résultat)



X_test = data_test.drop(['species'], axis=1)

y_test = data_test['species']
#Importation de la méthode

from sklearn import tree

#Entrainement

dtc = tree.DecisionTreeClassifier()

dtc.fit(X_train,y_train)

#Prédiction

y_dtc = dtc.predict(X_test)

#Calcul du score

print(accuracy_score(y_test, y_dtc))
pd.crosstab(y_test, y_rf, rownames=['Reel'], colnames=['Prediction'], margins=True)
plt.figure(figsize=(30,30))

tree.plot_tree(dtc, feature_names=X_train.columns, class_names=['setosa','versicolor','virginica'], fontsize=14, filled=True)  
#Importation de la méthode

from sklearn import ensemble

#Entrainement

rf = ensemble.RandomForestClassifier()

rf.fit(X_train, y_train)

#Prédiction

y_rf = rf.predict(X_test)

#Calcul du score

rf_score = accuracy_score(y_test, y_rf)

print(rf_score)
pd.crosstab(y_test, y_rf, rownames=['Reel'], colnames=['Prediction'], margins=True)
#Importation du modèle

from sklearn.linear_model import LogisticRegression

#Entrainement

lr = LogisticRegression()

lr.fit(X_train,y_train)

#Prédiction

y_lr = lr.predict(X_test)

#Calcul du score

lr_score = accuracy_score(y_test, y_lr)

print(lr_score)
pd.crosstab(y_test, y_lr, rownames=['Reel'], colnames=['Prediction'], margins=True)