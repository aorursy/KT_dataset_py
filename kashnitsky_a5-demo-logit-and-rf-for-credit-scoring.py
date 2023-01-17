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
# Your code here
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import GridSearchCV, StratifiedKFold
lr = LogisticRegression(random_state=5, class_weight='balanced')
parameters = {'C': (0.0001, 0.001, 0.01, 0.1, 1, 10)}
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=5)
# Your code here
# Your code here
# Your code here
# Your code here
# Your code here
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42, 

                            class_weight='balanced')
parameters = {'max_features': [1, 2, 4], 'min_samples_leaf': [3, 5, 7, 9], 'max_depth': [5,10,15]}
# Your code here
# Your code here
from sklearn.ensemble import BaggingClassifier

from sklearn.model_selection import cross_val_score, RandomizedSearchCV



parameters = {'max_features': [2, 3, 4], 'max_samples': [0.5, 0.7, 0.9], 

              'base_estimator__C': [0.0001, 0.001, 0.01, 1, 10, 100]}
# Your code here