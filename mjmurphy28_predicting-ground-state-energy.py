# imports

import pandas as pd

import numpy as np

import time

import matplotlib.pyplot as plt

import matplotlib.cm as cm

import seaborn as sns

%matplotlib inline



import warnings

warnings.filterwarnings('ignore')



seed = 13

np.random.seed(seed)
raw_data = pd.read_csv("../input/roboBohr.csv")

X = raw_data.drop(['Unnamed: 0', 'pubchem_id', 'Eat'], axis = 1)

# not sure what the last 25 features are, so I am just going to drop them for now

y = raw_data['Eat']

X.sample(3)
from sklearn.preprocessing import StandardScaler, normalize



X_standardized = StandardScaler().fit_transform(X)

X_normalized = normalize(X)
sns.distplot(y)
from sklearn.decomposition import PCA, KernelPCA

## PCA



pca = PCA(n_components=2, random_state=seed)



start_time = time.time()

X_reduced = pca.fit_transform(X_normalized)

print("--- %s seconds ---" % (time.time() - start_time))

print("Number of components: {}".format(pca.components_.shape[0]))

print("Explained variance: ", pca.explained_variance_ratio_.sum())



fig = plt.figure(figsize=(14,10))

ax  = fig.add_subplot(111)



scatter = ax.scatter(X_reduced[:,0], X_reduced[:,1], c=y, s=45, edgecolors='green', cmap=cm.jet_r, alpha=0.5)

colorbar = fig.colorbar(scatter, ax=ax, label = "E | Ry | ")

plt.xlabel(r'$Z_1$')

plt.ylabel(r'$Z_2$')

plt.title('PCA')

sns.despine()

plt.show()
start_time = time.time()

kpca = KernelPCA(n_components=2, kernel="linear")

X_kpca = kpca.fit_transform(X)

print("--- %s seconds ---" % (time.time() - start_time))

explained_variance = np.var(X_kpca, axis=0)

explained_variance_ratio = explained_variance / np.sum(explained_variance)

print("Variance Explained: ", np.sum(explained_variance_ratio))



fig = plt.figure(figsize=(14,10))

ax  = fig.add_subplot(111)



scatter = ax.scatter(X_kpca[:,0], X_kpca[:,1], c=y, s=60, edgecolors='black', cmap=cm.jet_r)

colorbar = fig.colorbar(scatter, ax=ax, label = "E | Ry | ")

plt.xlabel(r'$k-PCA_1$')

plt.ylabel(r'$k-PCA_1$')

sns.despine()

plt.show()
start_time = time.time()

kpca3 = KernelPCA(n_components=2, kernel="linear", random_state=seed)

X_kpca3 = kpca3.fit_transform(X_normalized)

print("--- %s seconds ---" % (time.time() - start_time))

explained_variance = np.var(X_kpca3, axis=0)

explained_variance_ratio = explained_variance / np.sum(explained_variance)

print("Variance Explained: ", np.sum(explained_variance_ratio))



fig = plt.figure(figsize=(14,10))

ax  = fig.add_subplot(111)



scatter = ax.scatter(X_kpca3[:,0], X_kpca3[:,1], c=y, s=60, edgecolors='black', cmap=cm.jet_r)

colorbar = fig.colorbar(scatter, ax=ax, label = "E | Ry | ")

plt.xlabel(r'$Z_1$')

plt.ylabel(r'$Z_1$')

plt.title('Kernel PCA: Normalized')

sns.despine()

plt.show()
from sklearn.manifold import TSNE



tsne = TSNE(n_components=2, random_state=seed)

X_tsne = tsne.fit_transform(X_normalized)



fig = plt.figure(figsize=(14,10))

ax  = fig.add_subplot(111)



scatter = ax.scatter(X_tsne[:,0], X_tsne[:,1], c=y, s=45, edgecolors='green', cmap=cm.jet_r, alpha=0.5)

colorbar = fig.colorbar(scatter, ax=ax, label = "E | Ry | ")

plt.xlabel(r'$Z_1$')

plt.ylabel(r'$Z_2$')

plt.title('T-SNE: Perplexity = 30')

sns.despine()

plt.show()
from sklearn.model_selection import train_test_split



X_train, X_val, y_train, y_val = train_test_split(X, y, 

                                                    test_size=0.15, 

                                                    random_state=seed)



X_train_train, X_dev, y_train_train, y_dev = train_test_split(X_train, y_train, 

                                                    test_size=0.18, 

                                                    random_state=seed)





print("X: ", X.shape[0])

print("Train: {}".format(X_train_train.shape[0]))

print("Dev: {}".format(X_dev.shape[0]))

print("Val: {}".format(X_val.shape[0]))
from sklearn.metrics import mean_absolute_error, r2_score

from xgboost import XGBRegressor



# Parameters for XGBoost model



params = {}

params['learning_rate'] = 0.09

params['max_depth'] = 8

params['n_estimators'] = 100

params['objective'] = 'reg:linear'

params['booster'] = 'gbtree'

params['gamma'] = 1e-3

params['subsample'] = 0.6

params['reg_alpha'] = 0.115

params['reg_lambda'] = 0.58

params['scale_pos_weight'] = 1

params['base_score'] = 0.5

params['random_state'] = seed

params['silent'] = True

params['num_leaves'] = 17



print('XGBoost')

print('--------------------------------------')

start_time = time.time()



XGB = XGBRegressor(**params)

#XGB.fit(X_train, y_train, verbose=True, eval_metric='rmse')

eval_set = [(X_train_train, y_train_train), (X_dev, y_dev)]

XGB.fit(X_train_train, y_train_train, eval_metric='mae', eval_set=eval_set, verbose=False)

Y_pred_XGB = XGB.predict(X_dev)

print("Mean absolute error", mean_absolute_error(y_dev, Y_pred_XGB))

print('R2 score: %0.5f'% r2_score(y_dev, Y_pred_XGB))



print("Took %s seconds" % (time.time() - start_time))

print('--------------------------------------')



# learning_rate = 0.09

#Mean squared error 0.008758109152032869

#R2 score: 0.
y_val_pred = XGB.predict(X_val)



print("Mean absoulte error: {} kcal/mol".format(313.495392 * mean_absolute_error(y_val, y_val_pred)))

print("R^2: ", r2_score(y_val, y_val_pred))