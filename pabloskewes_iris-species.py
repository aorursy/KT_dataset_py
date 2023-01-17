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
df = pd.read_csv('../input/iris/Iris.csv')
df.head(10)
df.columns
from IPython.core.display import HTML # permet d'afficher du code html dans jupyter

display(HTML(df.head(10).to_html()))
df.shape
df.describe()
df.Species.value_counts()
versicolor = df.Species=='Iris-versicolor'

virginica = df.Species=='Iris-virginica'

setosa = df.Species=='Iris-setosa'
sns.jointplot("SepalLengthCm","SepalWidthCm" , df, kind='kde');
plt.figure(figsize=(12,12))

sns.kdeplot(df.SepalWidthCm, df.SepalWidthCm,  shade=True)
plt.figure(figsize=(12,12))

sns.kdeplot(df[versicolor].SepalLengthCm, df[versicolor].SepalWidthCm, cmap="Reds",  shade=True, alpha=0.3, shade_lowest=False)

sns.kdeplot(df[virginica].SepalLengthCm, df[versicolor].SepalWidthCm, cmap="Greens",  shade=True, alpha=0.3, shade_lowest=False)

sns.kdeplot(df[setosa].SepalLengthCm, df[versicolor].SepalWidthCm, cmap="Blues",  shade=True, alpha=0.3, shade_lowest=False)
sns.boxplot(x="Species", y="SepalLengthCm", data=df)
sns.boxplot(x="Species", y="SepalWidthCm", data=df)
sns.violinplot(x="Species", y="SepalLengthCm", data=df)
sns.violinplot(x="Species", y="SepalWidthCm", data=df)
fig = sns.FacetGrid(df, hue="Species", aspect=3, palette="Set2") # aspect=3 permet d'allonger le graphique

fig.map(sns.kdeplot, "SepalLengthCm", shade=True)

fig.add_legend()
fig = sns.FacetGrid(df, hue="Species", aspect=3, palette="Set2") # aspect=3 permet d'allonger le graphique

fig.map(sns.kdeplot, "SepalWidthCm", shade=True)

fig.add_legend()
sns.lmplot(x="SepalLengthCm", y="SepalWidthCm", data=df, fit_reg=False, hue='Species')
sns.pairplot(df, hue="Species")
sns.jointplot("PetalLengthCm","PetalWidthCm" , df, kind='kde');
plt.figure(figsize=(12,12))

sns.kdeplot(df.SepalWidthCm, df.SepalWidthCm,  shade=True)
plt.figure(figsize=(12,12))

sns.kdeplot(df[versicolor].PetalLengthCm, df[versicolor].PetalWidthCm, cmap="Reds",  shade=True, alpha=0.3, shade_lowest=False)

sns.kdeplot(df[virginica].PetalLengthCm, df[versicolor].PetalWidthCm, cmap="Greens",  shade=True, alpha=0.3, shade_lowest=False)

sns.kdeplot(df[setosa].PetalLengthCm, df[versicolor].PetalWidthCm, cmap="Blues",  shade=True, alpha=0.3, shade_lowest=False)
sns.boxplot(x="Species", y="PetalLengthCm", data=df)
sns.boxplot(x="Species", y="PetalWidthCm", data=df)
sns.violinplot(x="Species", y="PetalLengthCm", data=df)
sns.violinplot(x="Species", y="PetalWidthCm", data=df)
fig = sns.FacetGrid(df, hue="Species", aspect=3, palette="Set2") # aspect=3 permet d'allonger le graphique

fig.map(sns.kdeplot, "PetalLengthCm", shade=True)

fig.add_legend()
fig = sns.FacetGrid(df, hue="Species", aspect=3, palette="Set2") # aspect=3 permet d'allonger le graphique

fig.map(sns.kdeplot, "PetalWidthCm", shade=True)

fig.add_legend()
sns.lmplot(x="PetalLengthCm", y="PetalWidthCm", data=df, fit_reg=False, hue='Species')
sns.pairplot(df, hue="Species")
data_train = df.sample(frac=0.8, random_state=1)          # 80% des données avec frac=0.8

data_test = df.drop(data_train.index)     # le reste des données pour le test
X_train = data_train.drop(['Species'], axis=1)

y_train = data_train['Species']

X_test = data_test.drop(['Species'], axis=1)

y_test = data_test['Species']
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
fig = sns.FacetGrid(df, hue="Species", aspect=3) # aspect=3 permet d'allonger le graphique

fig.map(sns.kdeplot, "SepalLengthCm", shade=True)

fig.add_legend()
fig = sns.FacetGrid(df[df.SepalLengthCm>6], hue="Species", aspect=3) # aspect=3 permet d'allonger le graphique

fig.map(sns.kdeplot, "SepalWidthCm", shade=True)

fig.add_legend()
from sklearn import tree

dtc = tree.DecisionTreeClassifier()

dtc.fit(X_train,y_train)

y_dtc = dtc.predict(X_test)

print(accuracy_score(y_test, y_dtc))
plt.figure(figsize=(30,30))

tree.plot_tree(dtc, feature_names=X_train.columns, class_names=['versicolor','verginica','setosa'], fontsize=14, filled=True)  
dtc1 = tree.DecisionTreeClassifier(max_depth = 3, min_samples_leaf = 20)

dtc1.fit(X_train,y_train)
plt.figure(figsize=(30,30))

tree.plot_tree(dtc1, feature_names=X_train.columns, class_names=['versicolor','verginica','setosa'], fontsize=14, filled=True)  
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
plt.figure(figsize=(12,8))

plt.barh(range(len(indices)), importances[indices], color='b', align='center')

plt.yticks(range(len(indices)), df.columns[indices])

plt.title('Importance des caracteristiques')