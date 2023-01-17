%matplotlib inline

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns; sns.set()
# Condense display values for EDA.

pd.options.display.float_format = '{:,.0f}'.format
train = pd.read_csv('../input/train.csv')
train.head()
train.describe(include='all')
# How many features are there?

len(train.drop(['ID_code', 'target'], axis=1).columns)
# Plot first 100 features.

train.iloc[:, 2:100].plot(kind='box', figsize=[16,8])
# Plot last 100 features.

train.iloc[:, 100:].plot(kind='box', figsize=[16,8])
# Plot densities.

# Densities are easier to visualize if we remove outliers first.

train_x = train.iloc[:, 2:]

train_no_outliers = train_x[train_x.apply(lambda x :(x-x.mean()).abs()<(2*x.std()) ).all(1)]
# Plot densities 1-100.

train_no_outliers.iloc[:, :100].plot.density(ind=1000, figsize=[16,8], legend=False)
# There is one feature that has extremely high density near 5 or 10.
# Plot densities 100-200.

train_no_outliers.iloc[:, 100:].plot.density(ind=1000, figsize=[16,8], legend=False)
# What does the target look like?

train.target.value_counts().plot(kind="bar")

plt.figure()

sns.violinplot(x=train.target.values, y=train.index.values, palette="husl")

plt.figure()

sns.stripplot(x=train.target.values, y=train.index.values,

              jitter=True, color="black", size=0.5, alpha=0.5)
# Scale data

# The plots above are hard to read. Lets scale our data.

# Scaling also is good when doing pca. See https://stats.stackexchange.com/a/69159/7167

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

train_scaled = train.copy()

train_scaled.iloc[:, 2:] = scaler.fit_transform(train.iloc[:, 2:])
train_scaled.plot(kind='box', figsize=[16,8])
train_scaled.iloc[:, 2:100].plot.density(ind=30, figsize=[16,8], legend=False)
train_scaled.iloc[:, 100:].plot.density(ind=30, figsize=[16,8], legend=False)
# Most correlated features

correlations = train.iloc[:, 2:].corr().abs().unstack().sort_values(kind="quicksort").reset_index()

correlations = correlations[correlations['level_0'] != correlations['level_1']]

correlations.tail(10)
# Separate out the features.

x = train_scaled.iloc[:, 2:].values

# Separate out the target.

y = train_scaled.iloc[:, 1].values
#sns.boxplot('var_0','target',data=train, hue='target')

# plot boxplots by target value 0, 1

# imbar on 
from sklearn.decomposition import PCA

pca = PCA(2)

projected = pca.fit_transform(x)
print(projected)
plt.scatter(projected[:, 0], projected[:, 1],

           c=y, edgecolor='none', alpha=0.5,

           cmap=plt.cm.get_cmap('copper', 10))

plt.xlabel('component 1')

plt.ylabel('component 2')

plt.title('Santander 2d PCA scaled')

plt.colorbar();
pca = PCA().fit(x)

plt.plot(np.cumsum(pca.explained_variance_ratio_))

plt.xlabel('number of components')

plt.ylabel('cumulative explained variance');

plt.title('Santander scaled PCA cumulative explained variance')
# The 100 most descriptive features explain 50% of the variance.
# Let's try randomized pca to ignore outliers. Pick the 100 most descriptive features for rpca.
rpca = PCA(n_components=100, svd_solver='randomized')

rpca.fit(x)
plt.plot(np.cumsum(rpca.explained_variance_ratio_))

plt.xlabel('number of components')

plt.ylabel('cumulative explained variance');

plt.title('Santander scaled Radomized PCA cumulative explained variance')
# Ignoring outliers doesn't reduce the number of features we need for explanation.

# This dataset is likely the 200 principle components of some larger dataset.
# PCA without scaling.

# If we don't scale we can't visualize the correlation between component 1 and the target.

# This is still good to look at as we will visualize that we can get

# 90% cumulative explained variance with 100 unscaled features.



x_raw = train.iloc[:, 2:].values

y_raw = train.iloc[:, 1].values

pca_raw = PCA(2)

projected_raw = pca_raw.fit_transform(x_raw)

plt.scatter(projected_raw[:, 0], projected_raw[:, 1],

           c=y, edgecolor='none', alpha=0.5,

           cmap=plt.cm.get_cmap('copper', 10))

plt.xlabel('component 1')

plt.ylabel('component 2')

plt.title('Santander 2d PCA unscaled')

plt.colorbar();



plt.figure()

pca_raw = PCA().fit(x_raw)

plt.plot(np.cumsum(pca_raw.explained_variance_ratio_))

plt.xlabel('number of components')

plt.ylabel('cumulative explained variance');

plt.title('Santander unscaled PCA cumulative explained variance')



plt.figure()

rpca_raw = PCA(n_components=100, svd_solver='randomized')

rpca_raw.fit(x_raw)

plt.plot(np.cumsum(rpca_raw.explained_variance_ratio_))

plt.xlabel('number of components')

plt.ylabel('cumulative explained variance');

plt.title('Santander unscaled Radomized PCA cumulative explained variance')
import lightgbm as lgb

from sklearn.model_selection import train_test_split
# Create stratified validation split.

# Stratifying makes the splits have the same class distribution (purchase/no-purchase).

train_x, validation_x, train_y, validation_y = train_test_split(x, y, stratify=y)
train_data = lgb.Dataset(train_x, label=train_y)
validation_data = lgb.Dataset(validation_x, label=validation_y, reference=train_data)
bst = lgb.train({

    'boosting': 'gbdt', #'dart', # Dropouts meet Multiple Additive Regression Trees, default='gbdt'

    'learning_rate': 0.01, # smaller increases accuracy, default=0.1

    'max_bin': 511, # larger increases accuracy, default=255

    'metric': 'auc',

    'num_leaves': 63, # larger increases accuracy, default=31

    'num_trees': 100,

    'num_iteration': 500, # default=100

    'objective': 'binary',

    },

    train_data,

    num_boost_round=500, # may be redundant with params#num_iteration

    valid_sets=[validation_data],

    early_stopping_rounds=100,

    verbose_eval=100, # logs every 100 trees

)
bst.save_model('model.txt', num_iteration=bst.best_iteration)
lgb.plot_importance(bst, figsize=(16,8))
lgb.create_tree_digraph(bst)
# Generate submission

test = pd.read_csv('../input/test.csv')

test_x = test.iloc[:, 1:].values # Drop the ID_code

ypred = bst.predict(test_x)

test_code = test.iloc[:, 0]

submission = pd.concat([test_code, pd.Series(ypred, name='target')], axis=1)

submission.to_csv('submissions.csv', index=False)

submission.head()
nunique  = train.nunique()
!head submissions.csv