%matplotlib inline



from matplotlib import pyplot as plt

import numpy as np

import pandas as pd

import seaborn as sns
# Data base d'iris

data_iris = pd.read_csv("../input/iris/Iris.csv")
data_iris.head(10)
data_iris.count()
data_iris.shape
data_iris.describe()
data_iris.columns
data_iris.Species.value_counts()
virg = data_iris.Species == 'Iris-virginica'

vers = data_iris.Species == 'Iris-versicolor'

setosa = data_iris.Species == 'Iris-setosa'
plt.figure(figsize=(7,7))

sns.kdeplot(data_iris.SepalLengthCm, data_iris.SepalWidthCm,  shade=True)
plt.figure()

sns.boxplot(x="Species", y="SepalLengthCm", data=data_iris)

plt.figure()

sns.boxplot(x="Species", y="SepalWidthCm", data=data_iris)
plt.figure()

sns.violinplot(x="Species", y="SepalLengthCm", data=data_iris)

plt.figure()

sns.violinplot(x="Species", y="SepalWidthCm", data=data_iris)
plt.figure()

fig = sns.FacetGrid(data_iris, hue="Species", aspect=3, palette="Set2")

fig.map(sns.kdeplot, "SepalLengthCm", shade=True)

fig.add_legend()

plt.figure()

fig = sns.FacetGrid(data_iris, hue="Species", aspect=3, palette="Set2")

fig.map(sns.kdeplot, "SepalWidthCm", shade=True)

fig.add_legend()
sns.lmplot(x="SepalLengthCm", y="SepalWidthCm", data=data_iris, fit_reg=False, hue='Species')
plt.figure(figsize=(7,7))

sns.kdeplot(data_iris.PetalLengthCm, data_iris.PetalWidthCm,  shade=True)
plt.figure()

sns.boxplot(x="Species", y="PetalLengthCm", data=data_iris)

plt.figure()

sns.boxplot(x="Species", y="PetalWidthCm", data=data_iris)
plt.figure()

sns.violinplot(x="Species", y="PetalLengthCm", data=data_iris)

plt.figure()

sns.violinplot(x="Species", y="PetalWidthCm", data=data_iris)
plt.figure()

fig = sns.FacetGrid(data_iris, hue="Species", aspect=3, palette="Set2")

fig.map(sns.kdeplot, "PetalLengthCm", shade=True)

fig.add_legend()

plt.figure()

fig = sns.FacetGrid(data_iris, hue="Species", aspect=3, palette="Set2")

fig.map(sns.kdeplot, "PetalWidthCm", shade=True)

fig.add_legend()
sns.lmplot(x="PetalLengthCm", y="PetalWidthCm", data=data_iris, fit_reg=False, hue='Species')
from sklearn.linear_model import LogisticRegression

fonc = LogisticRegression()



iris_train = data_iris.sample(frac = 0.05,random_state = 1)

iris_test = data_iris.drop(iris_train.index)



X_train = iris_train.drop(['Species'],axis = 1)

y_train = iris_train['Species']

X_test = iris_test.drop(['Species'],axis = 1)

y_test = iris_test['Species']



fonc.fit(X_train,y_train)
y_lr = fonc.predict(X_test)
from sklearn.metrics import accuracy_score, confusion_matrix



lr_score = accuracy_score(y_test,y_lr)

print(lr_score)
mat = confusion_matrix(y_test,y_lr)

print(mat)
pd.crosstab(y_test, y_lr, rownames=['Reel'], colnames=['Prediction'], margins=True)
from sklearn import tree

dtc = tree.DecisionTreeClassifier()

dtc.fit(X_train,y_train)

y_dtc = dtc.predict(X_test)

print(accuracy_score(y_test,y_dtc))
plt.figure(figsize=(30,30))

tree.plot_tree(dtc, feature_names=X_train.columns, class_names=['virg','vers','setosa'], fontsize=14, filled=True)  
dtc1 = tree.DecisionTreeClassifier(max_depth = 3, min_samples_leaf = 20)

dtc1.fit(X_train,y_train)
plt.figure(figsize=(30,30))

tree.plot_tree(dtc1, feature_names=X_train.columns, class_names=['virg','vers','setosa'], fontsize=14, filled=True)
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
plt.figure(figsize=(9,4.5))

plt.barh(range(len(indices)), importances[indices], color='b', align='center')

plt.yticks(range(len(indices)), data_iris.columns[indices])

plt.title('Importance des caracteristiques')