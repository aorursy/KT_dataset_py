import numpy as np

import pandas as pd



%matplotlib inline

import matplotlib.pyplot as plt

import seaborn as sns
# To have bigger plot sizes and use sns output in pandas plot-commands.

sns.set(rc={'figure.figsize': (10, 7)})
def fill_nan(table):

    """ Replace NaN values with a median for each column in the table. """

    new_table = pd.DataFrame()

    for col in table.columns:

        new_table[col] = table[col].fillna(table[col].median())

    return new_table
data = pd.read_csv('../input/credit_scoring_sample.csv')

print(data.shape)

data.head()
data.dtypes
data['SeriousDlqin2yrs'].value_counts(normalize=True).plot(kind='barh');
# Replace NaN values with a median:

table = fill_nan(data)
X = table.drop('SeriousDlqin2yrs', axis='columns')

y = table['SeriousDlqin2yrs']



X.shape, y.shape
def get_bootstrap_samples(data, n_samples):

    """ Return subsets according to bootstrap. """

    indices = np.random.randint(0, len(data), (n_samples, len(data)))

    samples = data[indices]

    return samples



def stat_intervals(stat, alpha):

    boundaries = np.percentile(stat, [100 * alpha/2, 100*(1 - alpha/2)])

    return boundaries



didnt_pay = X[y == 1]['age'].values    # in version >= 0.24 - to_numpy()



np.random.seed(0)



# Generate subsets and compute mean.

mean_scores = [np.mean(s) for s in get_bootstrap_samples(didnt_pay, 1000)]



# Interval estimate of the average age.

print('Average age: mean interval', stat_intervals(mean_scores, 0.1).round(2))    # 90% confidence
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import GridSearchCV, StratifiedKFold
logit = LogisticRegression(random_state=5, class_weight='balanced')
parameters = {'C': np.power(10.0, np.arange(-4, 2))}
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=5)
search = GridSearchCV(logit, parameters, cv=skf, scoring='roc_auc', verbose=1, n_jobs=-1)

search.fit(X, y)
search.best_score_, search.best_params_
from sklearn.ensemble import RandomForestClassifier



# Initialize Random Forest with 100 trees and balance target classes.

rf = RandomForestClassifier(n_estimators=100, random_state=42, 

                            class_weight='balanced', n_jobs=-1)
parameters = {'max_features': [1, 2, 4],

              'min_samples_leaf': [3, 5, 7, 9],

              'max_depth': [5,10,15]}
search_rf = GridSearchCV(rf, parameters, cv=skf, scoring='roc_auc', verbose=1, n_jobs=-1)

search_rf.fit(X, y)



search_rf.best_score_, search_rf.best_params_
importances = pd.DataFrame(search_rf.best_estimator_.feature_importances_,

                           index=X.columns,

                           columns=['importance'])

importances.sort_values('importance').plot(kind='barh')
from sklearn.ensemble import BaggingClassifier

from sklearn.model_selection import cross_val_score, RandomizedSearchCV



parameters = {'max_features': [2, 3, 4],

              'max_samples': [0.5, 0.7, 0.9],

              'base_estimator__C': [0.0001, 0.001, 0.01, 1, 10, 100]}
clf = BaggingClassifier(base_estimator=LogisticRegression(class_weight='balanced'),

                        n_estimators=100,

                        random_state=42)

rnd_search = RandomizedSearchCV(clf, parameters, n_iter=20, cv=skf, scoring='roc_auc',

                                verbose=1, random_state=1, n_jobs=-1)

rnd_search.fit(X, y)



rnd_search.best_score_, rnd_search.best_params_