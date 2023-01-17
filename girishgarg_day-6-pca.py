import numpy as np

import matplotlib.pyplot as plt

import pandas as pd

dataset = pd.read_csv("../input/wine-customer-segmentation/Wine.csv")

dataset
dataset.info()
X = dataset.iloc[:,0:13]

y = dataset.iloc[:, 13]
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()



X_train = sc.fit_transform(X_train)

X_test = sc.fit_transform(X_test)
from sklearn.decomposition import PCA

pca = PCA(n_components = 2)



X_train = pca.fit_transform(X_train)

X_test = pca.transform(X_test)
X_train
X_test
new_dataset_train = pd.DataFrame(data=X_train, columns=['PC1', 'PC2'])

new_dataset_test = pd.DataFrame(data=X_test, columns=['PC1', 'PC2'])

# Con-catenating test and train datasets

new_dataset = pd.concat([new_dataset_train.reset_index (drop=True), new_dataset_test], axis=1)

new_dataset.shape
dataset.shape
X
plt.scatter(new_dataset_train.iloc[:,0],new_dataset_train.iloc[:,1])

plt.xlabel('PC1')

plt.ylabel('PC2')