from math import factorial

def find_combinations(n, m):

    return factorial(n) / factorial(n - m) / factorial(m)
prob = 0.7

m = 3

n = 5

nu = 0

for i in range(m, n + 1):

    nu += find_combinations(n, i) * prob ** i * (1 - prob) ** (n - i)

print(round(nu, 4) * 100)
# Disable warnings in Anaconda

import warnings

warnings.filterwarnings('ignore')



import numpy as np

import pandas as pd

%matplotlib inline

import matplotlib.pyplot as plt

import seaborn as sns

sns.set()
from matplotlib import rcParams

rcParams['figure.figsize'] = 11, 8
def fill_nan(table):

    for col in table.columns:

        table[col] = table[col].fillna(table[col].median())

    return table   
data = pd.read_csv('../input/credit_scoring_sample.csv')

data.head()
data.dtypes
ax = data['SeriousDlqin2yrs'].hist(orientation='horizontal', color='red')

ax.set_xlabel("number_of_observations")

ax.set_ylabel("unique_value")

ax.set_title("Target distribution")



print('Distribution of the target:')

data['SeriousDlqin2yrs'].value_counts()/data.shape[0]
independent_columns_names = [x for x in data if x != 'SeriousDlqin2yrs']

independent_columns_names
table = fill_nan(data)
X = table[independent_columns_names]

y = table['SeriousDlqin2yrs']
np.random.seed(0)
def get_bootstrap_samples(data, n_samples):

    """Generate bootstrap samples using the bootstrap method."""

    indices = np.random.randint(0, len(data), (n_samples, len(data)))

    samples = data[indices]

    return samples
def stat_intervals(stat, alpha):

    """Produce an interval estimate."""

    boundaries = np.percentile(stat, [100 * alpha / 2., 100 * (1 - alpha / 2.)])

    return boundaries
good_customers_ages = table[table["SeriousDlqin2yrs"] == 0]["age"].values

bad_customers_ages = table[table["SeriousDlqin2yrs"] == 1]["age"].values
good_cust_boots = [np.mean(sample) for sample in get_bootstrap_samples(good_customers_ages, 1000)]

bad_cust_boots = [np.mean(sample) for sample in get_bootstrap_samples(bad_customers_ages, 1000)]
print("Good customers age interval", stat_intervals(good_cust_boots, 0.1))

print("Bad customers age interval", stat_intervals(bad_cust_boots, 0.1))
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import GridSearchCV, StratifiedKFold
lr = LogisticRegression(random_state=5, class_weight='balanced')
parameters = {'C': (0.0001, 0.001, 0.01, 0.1, 1, 10)}
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=5)
grid_logit = GridSearchCV(lr, parameters, cv=skf, n_jobs=-1, scoring="roc_auc")

grid_logit.fit(X, y)
grid_logit.best_params_
best_lr = LogisticRegression(random_state=5, class_weight='balanced', C=0.001)

best_lr.fit(X, y)
lr_roc_auc_score = best_lr.score(X, y)

lr_roc_auc_score
data.head()
coefs = best_lr.coef_[0]

np.where(coefs == max(coefs))[0][0] + 1
def softmax(x):

    """Compute softmax values for each sets of scores in x."""

    return np.exp(x) / np.sum(np.exp(x), axis=0)
X.head()
logit_without_scaling = LogisticRegression(random_state=5, class_weight='balanced', C=0.001)

logit_without_scaling.fit(X, y)
X_with_more_age = X.copy(deep=True)

X_with_more_age['age'] = X['age'] + 20
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42, 

                            class_weight='balanced')
parameters = {'max_features': [1, 2, 4], 'min_samples_leaf': [3, 5, 7, 9], 'max_depth': [5,10,15]}
grid_rf = GridSearchCV(rf, parameters, cv=skf, scoring="roc_auc", n_jobs=-1)
grid_rf.fit(X, y)
grid_rf.best_params_
best_rf = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42, 

                            class_weight='balanced', max_depth=10, max_features=2, min_samples_leaf=7)

best_rf.fit(X, y)
from sklearn.metrics import roc_auc_score
rf_roc_auc_score = roc_auc_score(y, best_rf.predict(X))

np.abs((rf_roc_auc_score - lr_roc_auc_score)) * 100
rf_coefs = best_rf.feature_importances_

np.where(rf_coefs == min(rf_coefs))[0][0] + 1
from sklearn.ensemble import BaggingClassifier

from sklearn.model_selection import cross_val_score, RandomizedSearchCV



parameters = {'max_features': [2, 3, 4], 'max_samples': [0.5, 0.7, 0.9], 

              'base_estimator__C': [0.0001, 0.001, 0.01, 1, 10, 100]}
lr = LogisticRegression(random_state=42, class_weight='balanced', n_jobs=-1)

bc = BaggingClassifier(random_state=42, base_estimator=lr, n_estimators=100, n_jobs=-1)

rand_bc = RandomizedSearchCV(random_state=1, estimator=bc, scoring="roc_auc", n_jobs=-1, cv=skf, 

                            return_train_score=True, refit=True, n_iter=20, 

                            param_distributions=parameters)
rand_bc.fit(X, y)
rand_bc.best_score_
rand_bc.best_params_