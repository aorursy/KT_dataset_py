# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import seaborn as sns

sns.set(style="darkgrid")



# to make this notebook's output stable across runs

np.random.seed(42)



# To plot figures

%matplotlib inline

import matplotlib

import matplotlib.pyplot as plt

plt.rcParams['axes.labelsize'] = 14

plt.rcParams['xtick.labelsize'] = 12

plt.rcParams['ytick.labelsize'] = 12



print('Libraries imported.')
transactions = pd.read_csv('../input/train.csv')

print('train data imported.')
transactions.info(verbose=True, null_counts=True)
len(transactions.ID_code.unique())
print('number of rows {}'.format(len(transactions)))

print('number of columns {}'.format(len(transactions.columns)))
transactions.isna().any().any()
transactions.head()
transactions.target.describe()
sns.countplot(transactions.target)

plt.show()
transactions.drop(['target','ID_code'], axis=1).describe()
transactions.drop(['target','ID_code'],axis=1).describe().loc['mean'].sort_values(ascending=False)[:10]
plt.figure(figsize=(8, 4))

sns.distplot(transactions.drop(['target','ID_code'],axis=1).describe().loc['mean'])

plt.title('Distribution of numeric features mean values')

plt.show()
transactions.drop(['target','ID_code'], axis=1).describe().loc['std'].sort_values(ascending=False)[:10]
plt.figure(figsize=(8, 4))

sns.distplot(transactions.drop(['target','ID_code'],axis=1).describe().loc['std'])

plt.title('Distribution of numeric features std values')

plt.show()
plt.figure(figsize=(8, 4))

sns.distplot(transactions.drop(['target','ID_code'],axis=1).describe().loc['max'])

plt.title('Distribution of numeric features max values')

plt.show()
plt.figure(figsize=(8, 4))

sns.distplot(transactions.drop(['target','ID_code'],axis=1).describe().loc['min'])

plt.title('Distribution of numeric features min values')

plt.show()
from sklearn.model_selection import train_test_split



train_set, val_set = train_test_split(transactions, test_size=0.9, random_state=42, stratify=transactions.target)

print(train_set.shape, val_set.shape)
train_set.target.describe()
val_set.target.describe()
X_train = train_set.drop(['target', 'ID_code'], axis=1)

print(X_train.shape)

X_train.head()
kurtosis = X_train.kurtosis().sort_values(ascending=False)

kurtosis[:9]
X_train[kurtosis[:9].index.values.tolist()].hist(bins=50, figsize=(12,8))

plt.show()
X_train[kurtosis[-9:].index.values.tolist()].hist(bins=50, figsize=(12,8))

plt.show()
X_train_transform = X_train[kurtosis[-9:].index.values.tolist()].apply(lambda x: np.sign(x) * np.log(1 + np.abs(x)))

X_train_transform.hist(bins=50, figsize=(12,8))

plt.show()
X_train_transform.kurtosis()
kurtosis[-9:]
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
from sklearn.decomposition import PCA

pca = PCA(n_components=0.95)

pca.fit(X_train_scaled)

pca.n_components_
X_train_scaled_pca = pca.transform(X_train_scaled)

X_train_scaled_pca.shape
np.random.seed(42)



m = 5000

idx = np.random.permutation(len(X_train_scaled_pca))[:m]



X_plot = X_train_scaled_pca[idx]

y_plot = train_set['target'].values[idx]

print(X_plot.shape, y_plot.shape)
def plot_2D_dataset(X_2D, title=None):

    plt.figure(figsize=(8,8))

    plt.scatter(X_2D[:,0], X_2D[:, 1], c=y_plot, cmap='jet')

    plt.xlabel('1st component')

    plt.ylabel('2nd component')

    if title is not None:

        plt.title(title)
from sklearn.manifold import TSNE

import time



tsne = TSNE(n_components=2, random_state=42)



t0 = time.time()

X_2D_tsne = tsne.fit_transform(X_plot)

t1 = time.time()



print("t-SNE took {:.1f}s.".format(t1 - t0))

plot_2D_dataset(X_2D_tsne, title='t-SNE')

plt.show()
from sklearn.decomposition import PCA

import time



pca = PCA(n_components=2, random_state=42)



t0 = time.time()

X_2D_pca = pca.fit_transform(X_plot)

t1 = time.time()



print("PCA took {:.1f}s.".format(t1 - t0))

plot_2D_dataset(X_2D_pca, title='PCA')

plt.show()
from sklearn.manifold import LocallyLinearEmbedding



lle = LocallyLinearEmbedding(n_components=2, random_state=42)



t0 = time.time()

X_2D_lle = lle.fit_transform(X_plot)

t1 = time.time()



print("LLE took {:.1f}s.".format(t1 - t0))

plot_2D_dataset(X_2D_lle, title='LLE')

plt.show()
X_train = train_set.drop(['target', 'ID_code'], axis=1)
from sklearn.pipeline import Pipeline

preparation_pipeline = Pipeline([

    ('scaler', StandardScaler()),

    ('pca', PCA(n_components=0.95))

])
X_train_prepared = preparation_pipeline.fit_transform(X_train)

print(X_train_prepared.shape)

X_train_prepared
y_train = train_set['target']

print(y_train.shape)

y_train.head()
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import RandomizedSearchCV

from scipy.stats import reciprocal



param_distribs = {

        'solver': ['lbfgs', 'liblinear', 'sag'],

        'C': reciprocal(0.01, 10)

    }



lr_clf = LogisticRegression()



lr_rnd_search = RandomizedSearchCV(lr_clf, param_distributions=param_distribs,

                                    n_iter=10, cv=3, random_state=42, scoring='f1')



lr_rnd_search.fit(X_train_prepared, y_train)



print("best parameter: {}".format(lr_rnd_search.best_params_))

print("best score: {}".format(lr_rnd_search.best_score_))

print("best model: {}".format(lr_rnd_search.best_estimator_))
cvres = lr_rnd_search.cv_results_

for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):

    print(mean_score, params)
from sklearn.linear_model import SGDClassifier



param_distribs = {

        'penalty' : ['l2', 'l1', 'elasticnet'],

        'alpha': reciprocal(0.0001, 0.1)

    }



sgd_clf = SGDClassifier()



sgd_rnd_search = RandomizedSearchCV(sgd_clf, param_distributions=param_distribs,

                                    n_iter=10, cv=3, random_state=42, scoring='f1')



sgd_rnd_search.fit(X_train_prepared, y_train)



print("best parameter: {}".format(sgd_rnd_search.best_params_))

print("best score: {}".format(sgd_rnd_search.best_score_))

print("best model: {}".format(sgd_rnd_search.best_estimator_))
cvres = sgd_rnd_search.cv_results_

for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):

    print(mean_score, params)
from sklearn.ensemble import RandomForestClassifier

from scipy.stats import randint



param_distribs = {

    'bootstrap':[True, False],

    'n_estimators': randint(low=10, high=15),

    'max_depth': randint(low=40, high=80)

    }



rf_clf = RandomForestClassifier(random_state=42)

rf_rnd_search = RandomizedSearchCV(rf_clf, param_distributions=param_distribs,

                                   n_iter=10, cv=3, random_state=42, scoring='f1')



rf_rnd_search.fit(X_train_prepared, y_train)



print("best parameter: {}".format(rf_rnd_search.best_params_))

print("best score: {}".format(rf_rnd_search.best_score_))

print("best model: {}".format(rf_rnd_search.best_estimator_))
cvres = rf_rnd_search.cv_results_

for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):

    print(mean_score, params)
clf_models = {

    name: search.best_estimator_ for name, search in zip(

    ('LR','SGD','RF'),

    (lr_rnd_search, sgd_rnd_search, rf_rnd_search))}

clf_models
print('Train Set Scores')

print('----------------')

for key, clf in clf_models.items():

    print("{}: {:.4f}".format(key, clf.score(X_train_prepared, y_train)))
from sklearn.metrics import jaccard_score

from sklearn.metrics import f1_score
print(val_set.shape)

val_set.head()
X_val = val_set.drop(['target','ID_code'], axis=1)



X_val_prepared = preparation_pipeline.transform(X_val)

y_val = val_set['target']



print(X_val_prepared.shape)

print(y_val.shape)



X_val_prepared
jaccard_scores = {key:jaccard_score(y_val, clf.predict(X_val_prepared)) for key, clf in clf_models.items()}

jaccard_scores
f1_scores = {key:f1_score(y_val, clf.predict(X_val_prepared)) for key, clf in clf_models.items()}

f1_scores
print('Valuation Set Scores')

print('----------------')

for key, clf in clf_models.items():

    print("{}: {:.4f}".format(key, clf.score(X_val_prepared, y_val)))
test_set = pd.read_csv('../input/test.csv')

print('test data imported.')
print(test_set.shape)

test_set.head()
X_test = test_set.drop(['ID_code'], axis=1)



X_test_prepared = preparation_pipeline.transform(X_test)



print(X_test_prepared.shape)

X_test_prepared
y_pred = clf_models['SGD'].predict(X_test_prepared)
predictions = test_set = pd.read_csv('../input/sample_submission.csv')

print(predictions.shape)

predictions.head()
predictions.describe()
predictions['target'] = y_pred.ravel()

predictions.describe()
predictions.to_csv('submission.csv', index=False)