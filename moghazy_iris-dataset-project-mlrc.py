%matplotlib inline

import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D

from sklearn import datasets

from sklearn.decomposition import PCA

from sklearn.preprocessing import StandardScaler





import pandas as pd



# Read the dataset inside a pandas dataframe. the dataset is a csv file stored in the following path "../input/Iris.csv"

# YOUR CODE





# map the strings you need to using map(dict) function 

# from the dataframe, This function accepts a dictionary 

# with the values to be replaced as Keys and the values 

# that would replace them as the values for those keys

# YOUR CODE

# X.head()
#UNCOMMENT THE FOLLOWING SEGMENT



# import matplotlib.pyplot as plt

# import numpy as np

# # X.species holds the classes that we have after mapping which are {1, 2, 3}

# classes = np.array(list(X.Species.values))



# # Now we will use matplotlib to plot the classes with the axis x and y representing two of the features that we have

# # We do that to see if some features contribute to the seperablity of the dataset more than the others

# def plotRelation(first_feature, sec_feature):

    

#     plt.scatter(first_feature, sec_feature, c = classes, s=10)

#     plt.xlabel(first_feature.name)

#     plt.ylabel(sec_feature.name)

    

# f = plt.figure(figsize=(25,20))

# f.add_subplot(331)

# plotRelation(X.SepalLengthCm, X.SepalWidthCm)

# f.add_subplot(332)

# plotRelation(X.PetalLengthCm, X.PetalWidthCm)

# f.add_subplot(333)

# plotRelation(X.SepalLengthCm, X.PetalLengthCm)

# f.add_subplot(334)

# plotRelation(X.SepalLengthCm, X.PetalWidthCm)

# f.add_subplot(335)

# plotRelation(X.SepalWidthCm, X.PetalLengthCm)

# f.add_subplot(336)

# plotRelation(X.SepalWidthCm, X.PetalWidthCm)
import seaborn as sns



# Here we use the seaborn library to visualize the correlation matrix

# The correlation matrix shows how much are the features and the target correlated

# This gives us some hints about the feature importance



# UNCOMMENT THE FOLLOWING SEGMENT



# import matplotlib.pyplot as plt

# corr = X.corr()

# f, ax = plt.subplots(figsize=(15, 10))

# cmap = sns.diverging_palette(220, 10, as_cmap=True)

# sns.heatmap(corr, cmap=cmap, vmax=.3, center=0,

#             square=True, linewidths=.5)
# UNCOMMENT THE FOLLOWING SEGMENT



# f = plt.figure(figsize=(25,10))

# f.add_subplot(221)

# X.SepalWidthCm.hist()

# f.add_subplot(222)

# X.SepalLengthCm.hist()

# f.add_subplot(223)

# X.PetalLengthCm.hist()

# f.add_subplot(224)

# X.PetalWidthCm.hist()
from scipy import stats

import numpy as np



# Use scipy.stats to calculate the zscores of the dataset

# YOUR CODE



# Print the number of outlier points

# YOUR CODE
# filter all training points with zscore > 2.5 which means the point is considered as outlier

# YOUR CODE
# Removing the label from the training data

# use y to denote the labels and a small x for the dataset as a convention

# YOUR CODE
# Import StandardScaler from sklearn

# YOUR CODE



# Use StandardScaler() to scale data.

# YOUR CODE
# UNCOMMENT THE FOLLOWING SEGMENT



# fig = plt.figure(1, figsize=(16, 9))

# ax = Axes3D(fig, elev=-150, azim=110)

# X_reduced = PCA(n_components=3).fit_transform(X_scaled)



# ax.scatter(X_reduced[:, 0], X_reduced[:, 1], X_reduced[:, 2], c=y,

#            cmap=plt.cm.Set1, edgecolor='k', s=40)

# ax.set_title("First three PCA directions")

# ax.set_xlabel("1st eigenvector")

# ax.w_xaxis.set_ticklabels([])

# ax.set_ylabel("2nd eigenvector")

# ax.w_yaxis.set_ticklabels([])

# ax.set_zlabel("3rd eigenvector")

# ax.w_zaxis.set_ticklabels([])



# plt.show()

# print("The number of features in the new subspace is " ,X_reduced.shape[1])
# Import train_test_split from scikit learn

# YOUR CODE



# Use train_test_split to split X_reduced and y

# YOUR CODE
# Import Logistic Regression

# YOUR CODE



# Use Logistic regression to classify the data

# YOUR CODE 