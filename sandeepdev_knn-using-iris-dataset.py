# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
# https://www.kaggle.com/skalskip/iris-data-visualization-and-knn-classification/data
dataset = pd.read_csv('../input/Iris.csv')

dataset.shape
dataset.describe()
dataset.groupby('Species').size()
feature_columns = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm','PetalWidthCm']

X = dataset[feature_columns].values

y = dataset['Species'].values
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

y = le.fit_transform(y)
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
from pandas.plotting import parallel_coordinates

plt.figure(figsize=(15,10))

parallel_coordinates(dataset.drop("Id", axis=1), "Species")

plt.title('Parallel Coordinates Plot', fontsize=20, fontweight='bold')

plt.xlabel('Features', fontsize=15)

plt.ylabel('Features values', fontsize=15)

plt.legend(loc=1, prop={'size': 15}, frameon=True,shadow=True, facecolor="white", edgecolor="black")

plt.show()
from pandas.plotting import andrews_curves

plt.figure(figsize=(15,10))

andrews_curves(dataset.drop("Id", axis=1), "Species")

plt.title('Andrews Curves Plot', fontsize=20, fontweight='bold')

plt.legend(loc=1, prop={'size': 15}, frameon=True,shadow=True, facecolor="white", edgecolor="black")

plt.show()
plt.figure()

sns.pairplot(dataset.drop("Id", axis=1), hue = "Species", size=3, markers=["o", "s", "D"])

plt.show()
plt.figure()

dataset.drop("Id", axis=1).boxplot(by="Species", figsize=(15, 10))

plt.show()
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure(1, figsize=(20, 15))

ax = Axes3D(fig, elev=48, azim=134)

ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=y,

           cmap=plt.cm.Set1, edgecolor='k', s = X[:, 3]*50)



for name, label in [('Virginica', 0), ('Setosa', 1), ('Versicolour', 2)]:

    ax.text3D(X[y == label, 0].mean(),

              X[y == label, 1].mean(),

              X[y == label, 2].mean(), name,

              horizontalalignment='center',

              bbox=dict(alpha=.5, edgecolor='w', facecolor='w'),size=25)



ax.set_title("3D visualization", fontsize=40)

ax.set_xlabel("Sepal Length [cm]", fontsize=25)

ax.w_xaxis.set_ticklabels([])

ax.set_ylabel("Sepal Width [cm]", fontsize=25)

ax.w_yaxis.set_ticklabels([])

ax.set_zlabel("Petal Length [cm]", fontsize=25)

ax.w_zaxis.set_ticklabels([])



plt.show()
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import confusion_matrix, accuracy_score

from sklearn.model_selection import cross_val_score
classifier = KNeighborsClassifier(n_neighbors=3)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
# evaluating results
cm = confusion_matrix(y_test, y_pred)

cm
accuracy = accuracy_score(y_test, y_pred)*100

print('Accuracy of our model is equal ' + str(round(accuracy, 2)) + ' %.')
k_list = list(range(1,50,2))

# creating list of cv scores

cv_scores = []
for k in k_list:

    knn = KNeighborsClassifier(n_neighbors=k)

    scores = cross_val_score(knn, X_train, y_train, cv=10, scoring='accuracy')

    cv_scores.append(scores.mean())

MSE = [1 - x for x in cv_scores]
plt.figure()

plt.figure(figsize=(15,10))

plt.title('The optimal number of neighbors', fontsize=20, fontweight='bold')

plt.xlabel('Number of Neighbors K', fontsize=15)

plt.ylabel('Misclassification Error', fontsize=15)

sns.set_style("whitegrid")

plt.plot(k_list, MSE)



plt.show()
best_k = k_list[MSE.index(min(MSE))]

print("The optimal number of neighbors is %d." % best_k)