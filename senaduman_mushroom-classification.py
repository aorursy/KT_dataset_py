import numpy as numpy

import pandas as pd 

import matplotlib.pyplot as plt

import seaborn as sns; sns.set()
data = pd.read_csv("../input/mushroom-classification/mushrooms.csv")

data.head()
data.isnull().sum()
plt.figure(figsize=(6,6))

sns.countplot(data=data, x="class", palette=["#DC0A00","#196F3D"])
plt.figure(figsize=(10,5))

sns.countplot(data=data, x="class", palette="viridis",hue="cap-shape")
plt.figure(figsize=(10,5))

sns.countplot(data=data, x="class", palette="viridis", hue="cap-color")
plt.figure(figsize=(10,5))

sns.countplot(data=data, x="class", palette="viridis", hue="habitat")
data.groupby("habitat")["class"].value_counts(normalize=True)[:,"p"]*100
data.dtypes
from sklearn.preprocessing import LabelEncoder

labelencoder = LabelEncoder()

for i in data.columns:

    data[i] = labelencoder.fit_transform(data[i])
data.head()
X = data.drop("class",axis=1)

y = data["class"]



from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
from sklearn.linear_model import LogisticRegression

lr = LogisticRegression()

lr.fit(X_train,y_train)
lr.score(X_test,y_test)
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier()

knn.fit(X_train,y_train)
knn.score(X_test,y_test)
from sklearn.ensemble import RandomForestClassifier

from sklearn.tree import DecisionTreeClassifier

rfc = RandomForestClassifier(random_state=0)

rfc.fit(X_train, y_train)
rfc.score(X_test,y_test)
from sklearn.metrics import confusion_matrix

pred = rfc.predict(X_test)

cm = pd.DataFrame(confusion_matrix(y_test, pred),columns=["Predicted Negative", "Predicted Positive"], index=["Actual Negative", "Actual Positive"])

sns.heatmap(cm,annot=True, cmap='summer', fmt="d")
from sklearn import tree

fig = plt.figure(figsize=(25,20))

tree.plot_tree(rfc.estimators_[7],

               feature_names = data.columns,

               class_names = ["edible", "poisonous"],

               filled = True,

               rounded=True,

               fontsize=10)

fig.savefig("tree.jpg")