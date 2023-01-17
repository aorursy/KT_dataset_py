import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
ratings = pd.read_csv("/kaggle/input/ratings-pca/Ratings.csv")

ratings
ratings.info()
ratings.describe()
plt.figure(figsize=(7,5))

sns.heatmap(ratings.corr(),annot=True,cmap='coolwarm')

plt.show()
# Step1: Covariance Matrix Computation

covv = np.cov(ratings.T)

covv

sns.heatmap(covv)
plt.figure(figsize=(7,5))

sns.heatmap(ratings.corr(),annot=True,cmap='coolwarm')

plt.title("Correlation")

plt.show()

plt.figure(figsize=(7,5))

sns.heatmap(covv,annot=True,cmap='coolwarm')

plt.title("Covariance")

plt.show()
# Step 2. Eigendecomposition of the Covariance Matrix

# Then you performed the eigendecomposition of the covariance matrix  to obtain the eigenvalues and the eigenvectors 



eigenvalues , eigenvectors = np.linalg.eig(covv)

eigenvalues
eigenvectors
# Step 3: Sort the eigenvectors on the basis of the eigenvalues.

idx = eigenvalues.argsort()[::-1]

eigenvalues = eigenvalues[idx]

eigenvectors = eigenvectors[:,idx]
# Step 4. These eigenvectors are the principal components of the original matrix.



eigenvectors
# Step 5: The eigenvalues denote the amount of variance explained by the eigenvectors.

# Higher the eigenvalues, higher is the variance explained by the corresponding eigenvector.

eigenvalues
# Step 6: These eigenvectors are orthonormal,i.e. they're unit vectors and are perpendicular to each other.



new_rating = np.linalg.inv(eigenvectors) @ ratings.T

new_rating = pd.DataFrame(new_rating.T)

new_rating
100 * np.var(new_rating) / sum(np.var(new_rating))
#### Let's verify the results using PCA

from sklearn.decomposition import PCA

pca = PCA(svd_solver='randomized', random_state=42)

pca.fit(ratings)

#Let's check the components

print("------------------------------------------------------------------------------------------------------------------")

print("|Note that the columns in the eigenvector matrix are to be compared with the rows of pca.components_ matrix.")

print("|Also the directions are reversed for the second axis.")

print("|This wouldn't make a difference as even though they're antiparallel, they would represent the same 2-D space.")

print("|For example, X/Y and X/-Y both cover the entire 2-D plane")

print("------------------------------------------------------------------------------------------------------------------")

print(pca.components_)





# Let's check the variance explained

pca.explained_variance_ratio_
# Importing data



# del train,test,sample

train = pd.read_csv("/kaggle/input/digit-recognizer/train.csv")

test = pd.read_csv("/kaggle/input/digit-recognizer/test.csv")

sample = pd.read_csv("/kaggle/input/digit-recognizer/sample_submission.csv")



train.head()
# train.info(verbose=True)



if len(train.select_dtypes('int').columns) == train.shape[1]:

    print("All int values!")

else:

    print("Few non-int values!")
# Great news!



if len(train.isnull().sum() == 0) == train.shape[1]:

    print("No null values!")

else:

    print("Null values!")
print("Number of Observations are {0} while number of features are {1} ".format(train.shape[0],train.shape[1]))
# Step1 : To find the covvariance matrix

covv = np.cov(train.T)
covv.shape
# Step2: To find eigenvalues and eigenvectors from covariance matrix

# eigenvalues and eigenvectors

eigenvalues,eigenvectors = np.linalg.eig(covv)
idx = eigenvalues.argsort()[::-1]

eigenvalues = eigenvalues[idx]

eigenvectors = eigenvectors[idx]
eigenvalues.shape
eigenvalues[:2]
eigenvectors.shape
# eigenvectors[0,:]
variance_explained = []

for i in eigenvalues:

     variance_explained.append((i/sum(eigenvalues))*100)

        

# print(variance_explained)
cumulative_variance_explained = np.cumsum(variance_explained)

print(cumulative_variance_explained[:1])

# cumulative_variance_explained.shape
# Visualizing the eigenvalues and finding the "elbow" in the graphic

plt.plot(cumulative_variance_explained)

plt.ylim([80,101])

plt.grid()

plt.xlabel("Number of components")

plt.ylabel("Cumulative explained variance")

plt.title("Explained variance vs Number of components")
digit = np.linalg.inv(eigenvectors) @ train.T

digit = pd.DataFrame(digit.T)

digit.head()
100 * np.var(digit) / sum(np.var(digit))

# 100 * np.var(new_rating) / sum(np.var(new_rating))
from sklearn.decomposition import PCA
pca = PCA( random_state=42)

pca
pca.fit(train)
pca.components_
pca.explained_variance_ratio_[0]