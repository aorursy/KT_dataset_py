# Load some necessary libraries

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt
# Let's load the data from sklearn sample datasets

from sklearn.datasets import load_breast_cancer
cancer_data = load_breast_cancer()
cancer_data.keys()
# Description of cancer data

print(cancer_data['DESCR'])
# Let's create a dataframe from this data

df = pd.DataFrame(data = cancer_data.data, columns = cancer_data.feature_names)
df.head()
# Let's import the preprocessing library

from sklearn import preprocessing
# Let's make a standard scaler object

scaler = preprocessing.StandardScaler()
scaler.fit(df)
scaled_data = scaler.transform(df)
# Let's check the scaled data

scaled_data
# Let's import PCA

from sklearn.decomposition import PCA
# Let's make a PCA object

pca = PCA()
# Fitting and transforming the scaled data

pca.fit(scaled_data)



x_pca = pca.transform(scaled_data)
# Compute percentage of variation that each component of principal component accounts for

per_var = np.round(pca.explained_variance_ratio_ * 100, decimals = 1)
# Labels for Principal components

labels = ['PC' + str(x) for x in range(1, len(per_var) + 1)]
# Let's plot the variation of the principal components

fig = plt.figure(figsize = (15, 5))

plt.bar(x = range(1, len(per_var) + 1), height = per_var, tick_label = labels)

plt.xlabel('Percentage of explained variance')

plt.ylabel('Principal Component')

plt.title('Scree Plot')

plt.tight_layout()

plt.xticks(rotation = 'vertical')

plt.show()
pca_df = pd.DataFrame(data = x_pca, index = cancer_data.target, columns = labels)
plt.scatter(pca_df.PC1, pca_df.PC2, c = cancer_data.target)

plt.title('My PCA Graph')

plt.xlabel(f'PC1 - {per_var[0]} %')

plt.ylabel(f'PC2 - {per_var[1]} %')

plt.show()