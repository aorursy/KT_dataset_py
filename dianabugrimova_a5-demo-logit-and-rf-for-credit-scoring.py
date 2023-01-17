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

    indices = np.random.randint(0, len(data), (n_samples, len(data)))

    samples = data[indices]

    return samples

def stat_intervals(stat, alpha):

    boundaries = np.percentile(stat, [100 * alpha / 2., 100 * (1 - alpha / 2.)])

    return boundaries
age_deplayed= data[data['SeriousDlqin2yrs'] == 1]['age'].values
np.random.seed(0)

age_d_mean_scores = [np.mean(sample) 

                       for sample in get_bootstrap_samples(age_deplayed, 1000)]

print("Age:  mean interval",  stat_intervals(age_d_mean_scores, 0.1))
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import GridSearchCV, StratifiedKFold
lr = LogisticRegression(random_state=5, class_weight='balanced')
parameters = {'C': (0.0001, 0.001, 0.01, 0.1, 1, 10)}
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=5)
skf_grid = GridSearchCV(lr, param_grid = parameters,scoring='roc_auc',

                          cv = skf)
skf_grid.fit(X,y)
skf_grid.best_params_
skf_grid.cv_results_['std_test_score'][1]
skf_grid.best_score_
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
skf_grid_rf = GridSearchCV(rf, param_grid = parameters,scoring='roc_auc',

                          cv = skf)
skf_grid_rf.fit(X,y)
skf_grid_rf.best_score_
skf_grid_rf.best_score_-skf_grid.best_score_
skf_grid_rf.best_params_
import seaborn as sns

# russian headres

from matplotlib import rc

font = {'family': 'Verdana',

        'weight': 'normal'}

rc('font', **font)
rf = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42, 

                            class_weight='balanced', max_depth=10, max_features=2, min_samples_leaf=7)
rf.fit(X,y)
importances=rf.feature_importances_
features = {"f1":u"age",

"f2":u"NumberOfTime30-59DaysPastDueNotWorse",

"f3":u"DebtRatio",

"f4":u"NumberOfTimes90DaysLate",

"f5":u"NumberOfTime60-89DaysPastDueNotWorse",

"f6":u"MonthlyIncome",

"f7":u"NumberOfDependents"}
indices = np.argsort(importances)[::-1]

num_to_plot = 7

feature_indices = [ind+1 for ind in indices[:num_to_plot]]
print("Feature ranking:")



for f in range(num_to_plot):

    print("%d. %s %f " % (f + 1, 

            features["f"+str(feature_indices[f])], 

            importances[indices[f]]))
plt.figure(figsize=(15,5))

plt.title(u"Важность конструктов")

bars = plt.bar(range(num_to_plot), 

               importances[indices[:num_to_plot]],

       color=([str(i/float(num_to_plot+1)) 

               for i in range(num_to_plot)]),

               align="center")

ticks = plt.xticks(range(num_to_plot), 

                   feature_indices)

plt.xlim([-1, num_to_plot])

plt.legend(bars, [u''.join(features["f"+str(i)]) 

                  for i in feature_indices])
from sklearn.ensemble import BaggingClassifier

from sklearn.model_selection import cross_val_score, RandomizedSearchCV



parameters = {'max_features': [2, 3, 4], 'max_samples': [0.5, 0.7, 0.9], 

              'base_estimator__C': [0.0001, 0.001, 0.01, 1, 10, 100]}
bg= BaggingClassifier(LogisticRegression(class_weight='balanced'), random_state=42, n_estimators=100, n_jobs=-1)
rsv = RandomizedSearchCV(bg, parameters, n_iter=20, cv=skf, random_state=1, scoring='roc_auc', verbose=True, n_jobs=-1)
rsv.fit(X,y)
rsv.best_score_
rsv.best_params_