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
df = df.drop(['id'], axis=1)
data_train = df.sample(frac=0.8, random_state=1)          # 80% des données avec frac=0.8
data_test = df.drop(data_train.index)     # le reste des données pour le test
X_train = data_train.drop(['diagnosis'], axis=1)
y_train = data_train['diagnosis']
X_test = data_test.drop(['diagnosis'], axis=1)
y_test = data_test['diagnosis']
lr = LogisticRegression()
lr.fit(X_train,y_train)
y_lr = lr.predict(X_test)
lr_score = accuracy_score(y_test, y_lr)
print(lr_score)
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
plt.figure(figsize=(12,8))
plt.barh(range(len(indices)), importances[indices], color='b', align='center')
plt.yticks(range(len(indices)), df.columns[indices])
plt.title('Importance des caracteristiques')
df1 = pd.read_csv('../input/iris/Iris.csv')
df1.head()
sns.pairplot(df1, hue="Species")
df1 = df1.drop(['Id'], axis = 1)
data_train1 = df1.sample(frac=0.8, random_state=1)          # 80% des données avec frac=0.8
data_test1 = df1.drop(data_train1.index)     # le reste des données pour le test
X_train1 = data_train1.drop(['Species'], axis=1)
y_train1 = data_train1['Species']
X_test1 = data_test1.drop(['Species'], axis=1)
y_test1 = data_test1['Species']
rf1 = ensemble.RandomForestClassifier()
rf1.fit(X_train1, y_train1)
y_rf1 = rf1.predict(X_test1)
rf_score1 = accuracy_score(y_test1, y_rf1)

print(rf_score1)
pd.crosstab(y_test1, y_rf1, rownames=['Reel'], colnames=['Prediction'], margins=True)
importances1 = rf1.feature_importances_
indices1 = np.argsort(importances1)
plt.figure(figsize=(12,8))
plt.barh(range(len(indices)), importances[indices], color='b', align='center')
plt.yticks(range(len(indices)), df.columns[indices])
plt.title('Importance des caracteristiques')