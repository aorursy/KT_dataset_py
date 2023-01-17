import itertools



import numpy as np

import pandas as pd



from matplotlib import pyplot as plt

import seaborn as sns



from sklearn.feature_selection import VarianceThreshold

from sklearn.preprocessing import normalize

from sklearn.preprocessing import StandardScaler
x_train = pd.read_csv(filepath_or_buffer="../input/Breast_cancer_data.csv", sep=',')



y_train = x_train["diagnosis"]

x_train = x_train.drop(labels="diagnosis", axis=1)

classes = np.sort(np.unique(y_train))

X_std = StandardScaler().fit_transform(x_train)
x_train
y_train
cov_mat = np.cov(X_std.T)



eig_vals, eig_vecs = np.linalg.eig(cov_mat)



cov_mat
eig_vecs
# Make a list of (eigenvalue, eigenvector) tuples

eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:,i]) for i in range(len(eig_vals))]



# Sort the (eigenvalue, eigenvector) tuples from high to low

eig_pairs.sort(key=lambda x: x[0], reverse=True)



eig_pairs
matrix_w = np.hstack((eig_pairs[0][1].reshape(5,1), 

                      eig_pairs[1][1].reshape(5,1)))



matrix_w
transformed = X_std.dot(matrix_w)
fig = plt.figure(figsize=(10, 7))

ax = fig.add_subplot(1, 1, 1)

colors = ['blue', 'red']

markers = ["o", "o"]

labels = ["Positive", "Negative"]



for class_ix, marker, color, label in zip(

        classes, markers, colors, labels):

    ax.scatter(transformed[np.where(y_train == class_ix), 0],

               transformed[np.where(y_train == class_ix), 1],

               marker=marker, color=color,

               linewidth='1', alpha=0.9, label=label)

    ax.legend(loc='best')

    plt.title(

        "Plot projected on the first 2 principal components")

    plt.xlabel("Principal component 1")

    plt.ylabel("Principal component 2")

plt.show()