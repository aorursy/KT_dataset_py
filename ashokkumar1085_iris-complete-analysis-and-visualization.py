# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
# Importing the necessary libraries



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sb



%matplotlib inline
# Loading the Iris Dataset



iris = pd.read_csv("/kaggle/input/iris/Iris.csv")
iris.head()
iris.drop('Id', axis = 1, inplace = True)
# Dimensions of te Data



iris.shape
# Statistical summary of all attributes



iris.describe()
# To know how many variety of species are present in the data



iris.groupby('Species').size()
sb.relplot(x = 'SepalLengthCm', y = 'SepalWidthCm', data = iris, hue = 'Species')
sb.relplot(x = 'PetalLengthCm', y = 'PetalWidthCm', data = iris, hue = 'Species')
sb.relplot(x = 'SepalLengthCm', y = 'SepalWidthCm', hue = 'PetalLengthCm', data = iris, col = 'Species')
sb.relplot(x = 'SepalLengthCm', y = 'SepalWidthCm', hue = 'PetalWidthCm', data = iris, col = 'Species')
sb.catplot(x = 'Species', y = 'SepalLengthCm', data = iris, kind = 'swarm')
sb.catplot(x = 'Species', y = 'SepalWidthCm', data = iris, kind = 'swarm')
sb.catplot(x = 'Species', y = 'PetalLengthCm', data = iris, kind = 'swarm')
sb.catplot(x = 'Species', y = 'PetalWidthCm', data = iris, kind = 'swarm')
sb.catplot(x = 'Species', y = 'SepalLengthCm', data = iris, kind = 'box')
sb.catplot(x = 'Species', y = 'SepalWidthCm', data = iris, kind = 'box')
sb.catplot(x = 'Species', y = 'PetalLengthCm', data = iris, kind = 'box')
sb.catplot(x = 'Species', y = 'PetalWidthCm', data = iris, kind = 'box')
corr = iris.corr(method = 'pearson')

corr
sb.heatmap(corr, xticklabels = corr.columns, yticklabels = corr.columns, annot = True, fmt = '.0%')
sb.pairplot(data = iris, hue = 'Species', height = 3)
# Selecting Dependent and Independent Variables

X = iris.iloc[:, [1, 3]].values

y = iris.iloc[:, -1].values
from sklearn.preprocessing import LabelEncoder

species_encoder = LabelEncoder()

species_encoder.fit(y)



y = species_encoder.transform(y)
# Splitting into training and test sets

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 42)

X_train.shape, X_test.shape, y_train.shape, y_test.shape
from sklearn import svm, tree

from sklearn.metrics import accuracy_score, confusion_matrix



classifier = svm.SVC(gamma = 'auto', kernel = 'rbf')

classifier.fit(X_train, y_train)



y_pred = classifier.predict(X_test)



print("Accuracy Score\t:\t", accuracy_score(y_test, y_pred))

print("Confusion Matrix\t:\t")

print(confusion_matrix(y_test, y_pred))
from matplotlib.pyplot import figure

figure(num=None, figsize=(8, 6), dpi=80, facecolor='w', edgecolor='k')
plt.figure(figsize = (10, 7))



from matplotlib.colors import ListedColormap

X_set, y_set = X_train, y_train

X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),

                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))

plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),

            alpha = 0.75, cmap = ListedColormap(('blue', 'orange', 'green')))

plt.xlim(X1.min(), X1.max())

plt.ylim(X2.min(), X2.max())

for i,j in enumerate(np.unique(y_set)):

    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],

               c = [ListedColormap(('white', 'red', 'black'))(i)], label = j)





plt.title('IRIS-Train Set')

plt.xlabel('Sepal Width')

plt.ylabel('Petal Width')

plt.legend()

plt.show()

plt.savefig("train.png")
plt.figure(figsize = (10, 7))



from matplotlib.colors import ListedColormap

X_set, y_set = X_test, y_test

X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),

                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))

plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),

            alpha = 0.75, cmap = ListedColormap(('blue', 'orange', 'green')))

plt.xlim(X1.min(), X1.max())

plt.ylim(X2.min(), X2.max())

for i,j in enumerate(np.unique(y_set)):

    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],

               c = [ListedColormap(('white', 'red', 'black'))(i)], label = j)

    

plt.title('IRIS - Test Set')

plt.xlabel('Sepal Width')

plt.ylabel('Petal Width')

plt.legend()

plt.show()



plt.savefig("test.png")