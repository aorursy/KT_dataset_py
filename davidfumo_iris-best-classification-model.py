import pandas as pd



# load dataset from sklearn 

from sklearn.datasets import load_iris



# Machine learning

from sklearn.neighbors import KNeighborsClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.cluster import KMeans

from sklearn.decomposition import PCA

from sklearn.model_selection import train_test_split

from sklearn.model_selection import cross_val_score

from sklearn import metrics



# data viz

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline

from itertools import cycle

import pylab as pl
# read in the iris data

iris = load_iris()



# create X (features) and y (response)

X = iris.data

y = iris.target
df = pd.read_csv('../input/Iris.csv')



sns.pairplot(df, hue="Species")
# K-Nearest Neighbors



# use train/test split with different random_state values

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=4)



# check classification accuracy of KNN with K=5

knn = KNeighborsClassifier(n_neighbors=5)



knn.fit(X_train, y_train)



y_pred = knn.predict(X_test)

print(metrics.accuracy_score(y_test, y_pred))
# Another way of evaluating the perfomance of our model is using 

# KFold: This approach is better that train/test split

knn = KNeighborsClassifier(n_neighbors=5)



scores = cross_val_score(knn, X, y, cv=10, scoring='accuracy')



print(scores.mean())
# search for an optimal value of K for KNN

k_range = list(range(1, 31))

k_scores = []

for k in k_range:

    knn = KNeighborsClassifier(n_neighbors=k)

    scores = cross_val_score(knn, X, y, cv=10, scoring='accuracy')

    k_scores.append(scores.mean())

print(k_scores)
# plot the value of K for KNN (x-axis) versus the cross-validated accuracy (y-axis)

plt.plot(k_range, k_scores)

plt.xlabel('Value of K for KNN')

plt.ylabel('Cross-Validated Accuracy')
# 10-fold cross-validation with the best KNN model

# This will allow us to get a better results

knn = KNeighborsClassifier(n_neighbors=20)



print(cross_val_score(knn, X, y, cv=10, scoring='accuracy').mean())
knn.fit(X, y)

# y_pred = knn.predict()
# Logistic Regression



logreg = LogisticRegression()



# train the model on the training set

logreg.fit(X, y)



print(cross_val_score(logreg, X, y, cv=10, scoring='accuracy').mean())
"""

PCA is a dimensionality reduction technique; it lets you 

distill multi-dimensional data down to fewer dimensions, 

selecting new dimensions that preserve variance in the data as best it can



While we can visualize 2 or even 3 dimensions of data pretty easily,

visualizing 4D data isn't something our brains can do. 

"""



# So let's Reduce dimensions from 4 to 2

# because kmeans expect 2 features

pca = PCA(n_components=2) # reducing to 2 dimensions

pca.fit(X)

X2d_train = pca.transform(X) # reduces dimension

print(X2d_train.shape)



# Let's see how much information we've managed to preserve

print("-- Preserved Variance --")

print(pca.explained_variance_ratio_)

print(sum(pca.explained_variance_ratio_))
# draw a plot for iris species



colors = cycle('rgb')

target_ids = range(len(iris.target_names))

pl.figure()

for i, c, label in zip(target_ids, colors, iris.target_names):

    pl.scatter(X2d_train[iris.target == i, 0], X2d_train[iris.target == i, 1],

        c=c, label=label)

pl.legend()

pl.show()
# K-Means





# create clustering to see kmeans in action

# As we are trying to predict 3 diferent species let's create

# 3 clustering(k=3) each one representing a diferent species

kmeans = KMeans(n_clusters=3)



kmeans = kmeans.fit(X2d_train)



# plot 1st and 2nd column; classes = y

plt.scatter(X2d_train[:,0], X2d_train[:,1], c=iris.target) 

plt.show()