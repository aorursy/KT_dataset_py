import os

import numpy as np

import pandas as pd

import seaborn as sns

from tqdm import tqdm_notebook



%matplotlib inline

from matplotlib import pyplot as plt

plt.style.use(['seaborn-darkgrid'])

plt.rcParams['figure.figsize'] = (12, 9)

plt.rcParams['font.family'] = 'DejaVu Sans'



from sklearn import metrics

from sklearn.cluster import KMeans, AgglomerativeClustering, SpectralClustering

from sklearn.decomposition import PCA

from sklearn.model_selection import GridSearchCV

from sklearn.preprocessing import StandardScaler

from sklearn.svm import LinearSVC



RANDOM_STATE = 17
# change this if necessary

PATH_TO_SAMSUNG_DATA = "../input"
X_train = np.loadtxt(os.path.join(PATH_TO_SAMSUNG_DATA, "samsung_train.txt"))

y_train = np.loadtxt(os.path.join(PATH_TO_SAMSUNG_DATA,

                                  "samsung_train_labels.txt")).astype(int)



X_test = np.loadtxt(os.path.join(PATH_TO_SAMSUNG_DATA, "samsung_test.txt"))

y_test = np.loadtxt(os.path.join(PATH_TO_SAMSUNG_DATA,

                                  "samsung_test_labels.txt")).astype(int)
# Checking dimensions

assert(X_train.shape == (7352, 561) and y_train.shape == (7352,))

assert(X_test.shape == (2947, 561) and y_test.shape == (2947,))
# Your code here
# np.unique(y)
# n_classes = np.unique(y).size
# Your code here
# Your code here

# pca = 

# X_pca = 
# Your code here
# Your code here
# Your code here

# plt.scatter(, , c=y, s=20, cmap='viridis');
# Your code here
# Your code here

# plt.scatter(, , c=cluster_labels, s=20, cmap='viridis');
# tab = pd.crosstab(y, cluster_labels, margins=True)

# tab.index = ['walking', 'going up the stairs',

#             'going down the stairs', 'sitting', 'standing', 'laying', 'all']

# tab.columns = ['cluster' + str(i + 1) for i in range(6)] + ['all']

# tab
# # Your code here

# inertia = []

# for k in tqdm_notebook(range(1, n_classes + 1)):

#     pass
# ag = AgglomerativeClustering(n_clusters=n_classes, 

#                              linkage='ward').fit(X_pca)
# Your code here
# # Your code here

# scaler = StandardScaler()

# X_train_scaled =

# X_test_scaled = 
svc = LinearSVC(random_state=RANDOM_STATE)

svc_params = {'C': [0.001, 0.01, 0.1, 1, 10]}
# %%time

# # Your code here

# best_svc = None
# best_svc.best_params_, best_svc.best_score_
# y_predicted = best_svc.predict(X_test_scaled)
# tab = pd.crosstab(y_test, y_predicted, margins=True)

# tab.index = ['walking', 'climbing up the stairs',

#              'going down the stairs', 'sitting', 'standing', 'laying', 'all']

# tab.columns = ['walking', 'climbing up the stairs',

#              'going down the stairs', 'sitting', 'standing', 'laying', 'all']

# tab
# Your code here