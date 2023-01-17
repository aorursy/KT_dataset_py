# Basic packages
import numpy as np
import pandas as pd 
import seaborn as sns; sns.set()
import matplotlib.pyplot as plt

# Scikit-learn package
from sklearn.preprocessing import StandardScaler

# Import data
dataset = pd.read_csv('https://raw.githubusercontent.com/debanga/depurr/master/datasets/Iris.csv').drop(columns=['Id'])

# See a tabular sample
dataset.head()

# Get feature distribution
dataset.describe()

# Generate X and y
X = dataset.drop(columns=['Species'])
y = dataset['Species']

# Standardization
X = StandardScaler().fit_transform(X)
X = pd.DataFrame(X)
X = X.rename(columns={0:'SepalLengthCm',1:'SepalWidthCm',2:'PetalLengthCm',3:'PetalWidthCm'})

# Singular Value Decoposition
u,s,v = np.linalg.svd(X)

# Estimate singular values
singular_values = s*s/(X.shape[0]-1)

# Top k=2 singular values and corresponding eigenvectors
k = 2
print(f"Top {k} eigen values:")
print(singular_values[:k])

print(f"Top {k} eigen vectors:")
print(v.T[:,0])
print(v.T[:,1])

print('Projection matrix is: ')
print(v.T[:,:2])

# Data in reduced dimension
X_reduced = np.matmul(np.array(X),v.T[:,:2])
pd.DataFrame(X_reduced)

# Visualize the samples in reduced space
dataset_new = pd.concat([pd.DataFrame(X_reduced),pd.DataFrame(dataset['Species'])], axis=1)
dataset_new = dataset_new.rename(columns={0:"feature_1",1:"feature_2"})
ax = sns.scatterplot(x="feature_1", y="feature_2", hue="Species", data=dataset_new)
plt.show()

print('Plot of principal components estimated using scikit-learn')
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
X_reduced_scikit = pca.fit_transform(X)
dataset_new_scikit = pd.concat([pd.DataFrame(X_reduced_scikit),pd.DataFrame(dataset['Species'])], axis=1)
dataset_new_scikit = dataset_new.rename(columns={0:"feature_1",1:"feature_2"})
ax = sns.scatterplot(x="feature_1", y="feature_2", hue="Species", data=dataset_new_scikit)
plt.show()