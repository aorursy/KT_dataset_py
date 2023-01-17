import pandas as pd

import numpy as np

import matplotlib.pyplot as plt



import seaborn as sns



from sklearn.preprocessing import StandardScaler

from sklearn.preprocessing import LabelEncoder

from sklearn.cross_validation import train_test_split

from sklearn.decomposition import PCA



data = pd.read_csv('../input/voice.csv')
x = data.iloc[:, :-1].values

y = data.iloc[:,-1].values



scaler = StandardScaler()

x = scaler.fit_transform(x)
sns.heatmap(np.corrcoef(x, rowvar = 0))
var_explain = []

for i in range(1,16):

    pca = PCA(n_components= i)

    x_pca = pca.fit_transform(x)

    var_explain.append(np.sum(pca.explained_variance_ratio_))



plt.plot(var_explain[:])

plt.xlabel('Number of PCA Components')

plt.ylabel('Explained Variance Ratio')

plt.title('PCA')

plt.yticks(np.arange(.4,1.05,.05))

plt.xticks(np.arange(1,16,1))

plt.show()