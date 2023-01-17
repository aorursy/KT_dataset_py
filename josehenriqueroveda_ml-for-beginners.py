import sklearn

import pandas as pd

import seaborn as sns

import numpy as np

import matplotlib.pyplot as plt
data = pd.read_csv("../input/breast-cancer-wisconsin-data/data.csv")
# Checking if there are null values in the dataset

data.isnull()
# Deleting 'Unnamed: 32' column

data.drop("Unnamed: 32",axis=1,inplace=True)
# Deleting 'id' column

data.drop("id",axis=1,inplace=True)
# Take a look to the data columns:

list(data.columns)
data.head()
data.describe()
# Mapping diagnosis to integer values

data['diagnosis']=data['diagnosis'].map({'M':1,'B':0})
features_mean= list(data.columns[1:11])

features_se= list(data.columns[11:20])

features_worst=list(data.columns[21:31])
sns.set(style='darkgrid', font_scale=1.1)

sns.countplot(data['diagnosis'],label="Count")
corr = data[features_mean].corr()

plt.figure(figsize=(14,14))

sns.heatmap(corr, cbar = True,  square = True, annot=True, fmt= '.2f',annot_kws={'size': 15},

           xticklabels= features_mean, yticklabels= features_mean)
# Based on correlation heatmap, we can select some of the variables to be used on prediction

pred_var = ['texture_mean','radius_mean','smoothness_mean','concavity_mean','symmetry_mean']
g = sns.PairGrid(data, y_vars=pred_var, x_vars=['diagnosis'], aspect=0.8, height=3.0)

g.map(sns.barplot, palette='muted')
data_target = data['diagnosis']

data_features = data.drop(['diagnosis'],axis=1)
# Splitting our dataset into training data and test data

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(data_features, data_target, random_state=0)
print("X_train shape: ", X_train.shape)

print("y_train shape: ", y_train.shape)
print("X_test shape: ", X_test.shape)

print("y_test shape: ", y_test.shape)
# n_neighbors=1 is setting the number of nearest neighbours to 1.

from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=1)
# build the model on the training set, i.e. X_train and y_train.

knn.fit(X_train, y_train)
print("KNN-1 Accuracy on training set:  {:.3f}".format(knn.score(X_train, y_train)))

print("KNN-1 Accuracy on test set: {:.3f}".format(knn.score(X_test, y_test)))
# specify one new instance to be predicted

X_new = np.array([[18.99,

10.30,

123.8,

1001,

0.119,

0.26,

0.30,

0.15,

0.24,

0.08,

1.095,

0.9053,

8.65,

157.4,

0.0064,

0.04904,

0.05373,

0.01587,

0.03003,

0.0053,

25.38,

17.33,

186.5,

2019,

0.1642,

0.6656,

0.7119,

0.2654,

0.4601,

0.1189]])
prediction = knn.predict(X_new)



print(f"Prediction: {'Malignant' if prediction == 1 else 'Benign'}")
knn = KNeighborsClassifier(n_neighbors=4)
knn.fit(X_train, y_train)
print("KNN-4 - Accuracy on training set:  {:.3f}".format(knn.score(X_train, y_train)))

print("KNN-4 - Accuracy on test set: {:.3f}".format(knn.score(X_test, y_test)))
prediction = knn.predict(X_new)



print(f"Prediction: {'Malignant' if prediction == 1 else 'Benign'}")
from sklearn.tree import DecisionTreeClassifier

tree = DecisionTreeClassifier(random_state=0)

tree.fit(X_train, y_train)
print("Decision Tree - Accuracy on training set: {:.3f}".format(tree.score(X_train, y_train)))

print("Decision Tree - Accuracy on test set: {:.3f}".format(tree.score(X_test, y_test)))
tree = DecisionTreeClassifier(max_depth=3, random_state=12)

tree.fit(X_train, y_train)



print("Decision Tree - Accuracy on training set: {:.3f}".format(tree.score(X_train, y_train)))

print("Decision Tree - Accuracy on test set: {:.3f}".format(tree.score(X_test, y_test)))
prediction = tree.predict(X_new)



print(f"Prediction: {'Malignant' if prediction == 1 else 'Benign'}")
tree = DecisionTreeClassifier(max_depth=2, random_state=12)

tree.fit(X_train, y_train)



print("Decision Tree - Accuracy on training set: {:.3f}".format(tree.score(X_train, y_train)))

print("Decision Tree - Accuracy on test set: {:.3f}".format(tree.score(X_test, y_test)))
from sklearn.ensemble import RandomForestClassifier

forest = RandomForestClassifier(n_estimators=1000, random_state=999, max_depth=3)

forest.fit(X_train, y_train)
print("Random Forest - Accuracy on training set: {:.3f}".format(forest.score(X_train, y_train)))

print("Random Forest - Accuracy on test set: {:.3f}".format(forest.score(X_test, y_test)))
prediction = forest.predict(X_new)



print(f"Prediction: {'Malignant' if prediction == 1 else 'Benign'}")