import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.decomposition import PCA

import matplotlib.pyplot as plt





train = pd.read_csv('../input/train.csv')

train = train.iloc[:, 2:]



for col in train:

    train[col] -= train[col].mean()

    

transformer = PCA(n_components=200)



pca_train = transformer.fit_transform(train)



PCA_arr = np.sort(pca_train.var(0))

org_arr = np.sort(train.values.var(0))



plt.figure(figsize=(20,15))

plt.semilogy(PCA_arr[::-1], '-b', label='after PCA', alpha=.5)

plt.semilogy(org_arr[::-1], '--r', label='before PCA', alpha=0.5)

plt.xlabel('features',fontsize=14)

plt.ylabel('variance',fontsize=14)

plt.legend(fontsize=20)