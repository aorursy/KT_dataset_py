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
import numpy as np

def get_bootstrap_samples(data, n_samples):

    """Generate bootstrap samples using the bootstrap method."""

    indices = np.random.randint(0, len(data), (n_samples, len(data)))

    samples = data[indices]

    return samples



def stat_intervals(stat, alpha):

    """Produce an interval estimate."""

    boundaries = np.percentile(stat, [100 * alpha / 2., 100 * (1 - alpha / 2.)])

    return boundaries



np.random.seed(0)

loyal_mean_scores = [np.mean(sample) for sample in get_bootstrap_samples(data[data["SeriousDlqin2yrs"] == 1]["age"].values, 1000)]

stat_intervals(loyal_mean_scores, 0.1)
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import GridSearchCV, StratifiedKFold
lr = LogisticRegression(random_state=5, class_weight='balanced')
parameters = {'C': (0.0001, 0.001, 0.01, 0.1, 1, 10)}
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=5)
grid_search = GridSearchCV(lr, parameters, cv=skf, verbose=1, scoring='roc_auc')

grid_search.fit(X, y)

grid_search.best_params_
grid_search.cv_results_['std_test_score'][grid_search.best_index_]
grid_search.best_score_
from sklearn.preprocessing import StandardScaler



m = LogisticRegression(C=0.001, random_state=5, class_weight='balanced')

m.fit(StandardScaler().fit_transform(X), y)



independent_columns_names[np.argmax(m.coef_[0])]
def softmax(x):

    return np.exp(x) / np.sum(np.exp(x))



softmax(m.coef_[0])[2]
m = LogisticRegression(C=0.001, random_state=5, class_weight='balanced')

m.fit(X, y)

np.exp(m.coef_[0][0] * 20)
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42, 

                            class_weight='balanced')
parameters = {'max_features': [1, 2, 4], 'min_samples_leaf': [3, 5, 7, 9], 'max_depth': [5,10,15]}
grid_search_ = GridSearchCV(rf, parameters, n_jobs=-1, scoring='roc_auc', cv=skf, verbose=True)

grid_search_ = grid_search_.fit(X, y)
grid_search_.best_score_ - grid_search.best_score_
independent_columns_names[np.argmin(grid_search_.best_estimator_.feature_importances_)]
from sklearn.ensemble import BaggingClassifier

from sklearn.model_selection import cross_val_score, RandomizedSearchCV



parameters = {'max_features': [2, 3, 4], 'max_samples': [0.5, 0.7, 0.9], 

              'base_estimator__C': [0.0001, 0.001, 0.01, 1, 10, 100]}
from sklearn.ensemble import BaggingClassifier

from sklearn.model_selection import RandomizedSearchCV

bc = BaggingClassifier(LogisticRegression(class_weight='balanced'), n_estimators=100, random_state=42)

rscv = RandomizedSearchCV(bc, parameters, n_iter=20, random_state=1, scoring='roc_auc', cv=skf)

rscv = rscv.fit(X, y)
rscv.best_score_