%matplotlib inline

import numpy as np

import matplotlib.pyplot as plt

import pandas as pd

from sklearn.model_selection import train_test_split

from pandas.plotting import scatter_matrix

from matplotlib import cm

from sklearn.neighbors import KNeighborsClassifier
fruits = pd.read_table('../input/fruits-with-colors-dataset/fruit_data_with_colors.txt')

print("Head 10 from dataset:")

print(fruits.head(10))

print("\nDescribing dataset:")

print(fruits.describe())

print("\nDataset's shape:")

print(fruits.shape)
# into X we put all features 

X = fruits[['mass','width','height','color_score']]

# into y we put our target 

y = fruits['fruit_label']



X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

print("Dataset splitted.\n")



print("Shapes")

print("\nX_train:")

print(X_train.shape)

print("\tX_test:")

print("\t{}".format(X_test.shape))

print("\ny_train:")

print(y_train.shape)

print("\ty_test:")

print("\t{}".format(y_test.shape))

cmap = cm.get_cmap('gnuplot')

scatter = scatter_matrix(X_train, c= y_train, marker = 'o', s=40, hist_kwds={'bins':15},

                            figsize=(12,12), cmap=cmap)
knn = KNeighborsClassifier(n_neighbors = 6)

# n_neighbors is the number of neighbours that algorithm will consider to predict 

print("KNN created.")



# using our knn to fir our train dataset

knn.fit(X_train, y_train)

print("...")

print("...")

print(knn.fit(X_train, y_train))

print("OK.")
acc = knn.score(X_test, y_test)



print("The accuracy of our model is: %.2f" % round(acc,2))
lookup_fruit_name = dict(zip(fruits.fruit_label.unique(), fruits.fruit_name.unique()))

fruit_prediction = knn.predict([[20, 4.3, 5.5, 0.5]])

lookup_fruit_name[fruit_prediction[0]]
fruit_prediction = knn.predict([[100, 6.3, 8.5, 0.7]])

lookup_fruit_name[fruit_prediction[0]]
# use this plot to help you to choice the number of neighbours

k_range = range(1,20)

scores = []



for k in k_range:

  knn = KNeighborsClassifier(n_neighbors=k)

  knn.fit(X_train, y_train)

  scores.append(knn.score(X_test, y_test))



plt.figure()

plt.xlabel('k')

plt.ylabel('accuracy')

plt.scatter(k_range, scores)

plt.xticks([0,5,10,15,20])
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure()

ax = fig.gca(projection = '3d')

ax.scatter(X_train['width'], X_train['height'], X_train['color_score'], c = y_train, marker = 'o', s=100)

ax.set_xlabel('width')

ax.set_ylabel('height')

ax.set_zlabel('color_score')

plt.show()