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
def get_bootstrap_samples(data, n_samples):

    """Generate samples using bootstrapping."""

    indices = np.random.randint(0, len(data), (n_samples, len(data)))

    samples = data[indices]

    return samples



def stat_intervals(stat, alpha):

    """Make an interval estimate."""

    boundaries = np.percentile(stat, [100 * alpha / 2., 100 * (1 - alpha / 2.)])

    return boundaries



# Save the ages of those who let a delay

churn = data[data['SeriousDlqin2yrs'] == 1]['age'].values



# Set the random seed for reproducibility 

np.random.seed(0)



# Generate bootstrap samples and calculate the mean for each sample

churn_mean_scores = [np.mean(sample) for sample in get_bootstrap_samples(churn, 1000)]



# Print the interval estimate for the sample means

print("Mean interval", stat_intervals(churn_mean_scores, 0.1))
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import GridSearchCV, StratifiedKFold
lr = LogisticRegression(random_state=5, class_weight='balanced')
parameters = {'C': (0.0001, 0.001, 0.01, 0.1, 1, 10)}
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=5)
grid_search = GridSearchCV(lr, parameters, n_jobs=-1, scoring='roc_auc', cv=skf)

grid_search = grid_search.fit(X, y)

grid_search.best_estimator_
grid_search.cv_results_['std_test_score'][1]
grid_search.best_score_
from sklearn.preprocessing import StandardScaler

lr = LogisticRegression(C=0.001, random_state=5, class_weight='balanced')

scal = StandardScaler()

lr.fit(scal.fit_transform(X), y)



pd.DataFrame({'feat': independent_columns_names,

              'coef': lr.coef_.flatten().tolist()}).sort_values(by='coef', ascending=False)
print((np.exp(lr.coef_[0]) / np.sum(np.exp(lr.coef_[0])))[2])
lr = LogisticRegression(C=0.001, random_state=5, class_weight='balanced')

lr.fit(X, y)



pd.DataFrame({'feat': independent_columns_names,

              'coef': lr.coef_.flatten().tolist()}).sort_values(by='coef', ascending=False)
np.exp(lr.coef_[0][0]*20)
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42, 

                            class_weight='balanced')
parameters = {'max_features': [1, 2, 4], 'min_samples_leaf': [3, 5, 7, 9], 'max_depth': [5,10,15]}
%%time

rf_grid_search = GridSearchCV(rf, parameters, n_jobs=-1, scoring='roc_auc', cv=skf, verbose=True)

rf_grid_search = rf_grid_search.fit(X, y)

print(rf_grid_search.best_score_ - grid_search.best_score_)
independent_columns_names[np.argmin(rf_grid_search.best_estimator_.feature_importances_)]
pd.DataFrame({'feat': independent_columns_names,

              'coef': rf_grid_search.best_estimator_.feature_importances_}).sort_values(by='coef', ascending=False)
from sklearn.ensemble import BaggingClassifier

from sklearn.model_selection import cross_val_score, RandomizedSearchCV



parameters = {'max_features': [2, 3, 4], 'max_samples': [0.5, 0.7, 0.9], 

              'base_estimator__C': [0.0001, 0.001, 0.01, 1, 10, 100]}
bg = BaggingClassifier(LogisticRegression(class_weight='balanced'),

                       n_estimators=100, n_jobs=-1, random_state=42)

r_grid_search = RandomizedSearchCV(bg, parameters, n_jobs=-1, 

                                   scoring='roc_auc', cv=skf, n_iter=20, random_state=1,

                                   verbose=True)

r_grid_search = r_grid_search.fit(X, y)
r_grid_search.best_score_
r_grid_search.best_estimator_