import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns

from sklearn.cluster import KMeans
from sklearn.metrics import classification_report
seed = pd.read_csv('../input/Seed_Data.csv')
seed.shape
seed.head()
seed.describe()
seed.info()
sns.pairplot(seed.drop('target', axis=1))
sns.lmplot(x='A', y='C', data=seed, hue='target', aspect=1.5, fit_reg=False)
sns.lmplot(x='A', y='A_Coef', data=seed, hue='target', aspect=1.5, fit_reg=False)
sns.lmplot(x='LK', y='C', data=seed, hue='target', aspect=1.5, fit_reg=False)
dist = {}

for k in range(1, 10):
    km = KMeans(n_clusters=k).fit(seed.drop('target', axis=1))
    
    dist[k] = km.inertia_  # Sum of squared distances of samples to their closest cluster center.

dist
plt.plot(list(dist.keys()), list(dist.values()), marker='H', markersize=10)
plt.xlabel('Number of k')
plt.ylabel('Distances')
plt.title('Elbow Curve')
km = KMeans(n_clusters=3, random_state=42)
km_pred = km.fit_predict(seed.drop('target', axis=1))
km_pred
np.array(seed['target'])
# Change the labels of prediction

for i in range(len(km_pred)):
    if km_pred[i] == 1:
        km_pred[i] = 0
    elif km_pred[i] == 0:
        km_pred[i] = 1

km_pred
print('Accuracy of clustering : ' + str(round(sum(km_pred == seed['target']) / seed.shape[0] * 100, 2)) + '%')
print(classification_report(seed['target'], km_pred, target_names=['1', '2', '3'], digits=4))
centers = km.cluster_centers_
centers
seed['klabels'] = km.labels_
seed.head()
seed.klabels.value_counts()
seed.target.value_counts()
f, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, sharey=True, figsize=(12, 8))

ax1.set_title('K-Means (k=3)')
ax1.scatter(x=seed['A'], y=seed['A_Coef'], c=seed['klabels'])
ax1.scatter(x=centers[:, 0], y=centers[:, 5], c='r', s=300, alpha=0.5)

ax2.set_title('Original')
ax2.scatter(x=seed['A'], y=seed['A_Coef'], c=seed['target'])