import pandas as pd

import numpy as np

import matplotlib.pyplot as plt



import seaborn as sns
data = pd.read_csv("/kaggle/input/lish-moa/train_features.csv")

data.head()
data = data.drop(columns = ["sig_id", "cp_type", "cp_time", "cp_dose"])

data.shape
from sklearn.decomposition import PCA



pca = PCA(n_components = 50).fit(data)

data_transformed = pd.DataFrame(pca.transform(data))
data_transformed
## Variance explained

variance = pca.explained_variance_ratio_



print('Explained variation per principal component: {0}'.format(variance))
# calculate variance ratios

variance_ratio = np.cumsum(np.round(variance, decimals=3)*100)



print('Explained cumulative variation for 30 principal components: {0}'.format(variance_ratio))
plt.figure(figsize=(15, 8))

plt.ylabel('% Variance Explained')

plt.xlabel('# of Features')

plt.title('PCA Analysis')

plt.ylim(0,100)

plt.style.context('seaborn-whitegrid')

plt.plot(variance_ratio)

plt.show()