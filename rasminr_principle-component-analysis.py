import matplotlib.pyplot as plt

import numpy as np

import pandas as pd

%matplotlib inline

import warnings

warnings.filterwarnings('ignore')
cancer.feature_names.shape
from sklearn.datasets import load_breast_cancer

cancer=load_breast_cancer()

X = pd.DataFrame(cancer['data'],columns=cancer['feature_names'])

Y = cancer['target']

X.head(5)

X.describe()
type(cancer)
cancer['feature_names']
type(cancer['data'])
cancer.target
cancer.target[[10, 50, 85]]

list(cancer.target_names)
type(cancer)
Y[0:10]
X.shape
from sklearn.preprocessing import StandardScaler

scaled_X = StandardScaler().fit_transform(X)

pd.DataFrame(scaled_X, columns=cancer['feature_names']).head()
pd.DataFrame(scaled_X, columns=cancer['feature_names']).describe()
features = scaled_X.T

covariance_matrix = np.cov(features)

print(covariance_matrix)
covariance_matrix.shape
# Calculate the eigen vectors and eigen values of the covariance matrix using linalg.eig()

eig_vals, eig_vecs = np.linalg.eig(covariance_matrix)

#Show the eigenvectors

print("Eigenvectors \n%s" %eig_vecs)
#Show the eigenvalues

print("Eigenvalues \n%s" %eig_vals)
# The first principle component captures about 44% of the variance

eig_vals[0] / sum(eig_vals)
sum(eig_vals)
eig_vals[2] / sum(eig_vals)

#About 19% of the variance is captured
eig_vecs.shape
# Let's choose the first two principle components

V = eig_vecs[:, :2]
V.shape
# Examine our new transformed matrix with PC1 & PC2

projected_X = scaled_X.dot(V)

projected_X
X_pca = pd.DataFrame(data = projected_X, columns = ['PC1', 'PC2'])

X_pca.head()
#Center the Data

X_centered = scaled_X - scaled_X.mean()

#Apply SVD

U, s, V = np.linalg.svd(X_centered)

#Choose Principal Components (Select Eigenvectors with Highest Eigenvalues)

W = V.T[:, :2]

#Project Original Matrix onto Eigenvectors

#Compute dot product to project the new reduced dimensionality dataset



projected_X = X_centered.dot(W)
W.shape
X_pca = pd.DataFrame(data = projected_X, columns = ['PC1', 'PC2'])

X_pca.head()
from sklearn.decomposition import PCA

pca = PCA(n_components=2)

pcomponents = pca.fit_transform(scaled_X)

X_pca = pd.DataFrame(data = pcomponents, columns = ['PC1', 'PC2'])

X_pca.shape
X_pca.head()
plt.figure(figsize=(8,6))

plt.scatter(X_pca['PC1'], X_pca['PC2'], c=cancer['target'])

plt.xlabel('First principle component')

plt.ylabel('Second principle component')
# Get the first PC1 and divide by the total sum of eigenvalues

eig_vals[0] / sum(eig_vals)
pca.explained_variance_ratio_
pd.DataFrame(pca.components_, columns=list(X.columns), index=('PC1','PC2'))
#Fitting the PCA algorithm with our Data

pca2 = PCA().fit(scaled_X)

#Plotting the Cumulative Summation of the Explained Variance

plt.figure()

plt.plot(np.cumsum(pca2.explained_variance_ratio_))

plt.xlabel('Number of Components')

plt.ylabel('Variance (%)') #for each component

plt.title('Pulsar Dataset Explained Variance')

plt.show()
np.cumsum(pca2.explained_variance_ratio_).shape
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split

from sklearn.model_selection import cross_val_score



# Create logistic regression object

clf = LogisticRegression(random_state=0)

# Split into training and test sets using ORIGINAL dataset

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=1)

# Train model

model = clf.fit(X_train, y_train)

# Perform 10-Fold Cross Validation

result = cross_val_score(clf, X_train, y_train, cv=10, scoring='f1')

result.mean()
from sklearn.decomposition import PCA

pca = PCA(n_components=10)

pcomponents = pca.fit_transform(scaled_X)

X_pca = pd.DataFrame(data = pcomponents)

# Split into training and test sets using REDUCED dataset

X_train, X_test, y_train, y_test = train_test_split(X_pca, Y, test_size=0.2, random_state=1)

# Train model

model = clf.fit(X_train, y_train)

# Get predicted probabilities

y_score = clf.predict_proba(X_test)[:,1]

# Perform 10-Fold Cross Validation

result = cross_val_score(clf, X_train, y_train, cv=10, scoring='f1')

result.mean()