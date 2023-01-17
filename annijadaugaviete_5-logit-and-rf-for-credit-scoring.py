m = 3
N = 5
p = 0.7

comb1 = 10
comb2 = 5
comb3 = 1

sum1 = comb1 * 0.7**3 * 0.3**2
sum2 = comb2 * 0.7**4 * 0.3**1
sum3 = comb3 * 0.7**5 * 0.3**0

total_sum = sum1 + sum2+ sum3
print(total_sum)
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
    """Generate bootstrap samples using the bootstrap method."""
    indices = np.random.randint(0, len(data), (n_samples, len(data)))
    samples = data[indices]
    return samples

def stat_intervals(stat, alpha):
    """Produce an interval estimate."""
    boundaries = np.percentile(stat, [100 * alpha / 2., 100 * (1 - alpha / 2.)])
    return boundaries

#Save the data about the age for the customers who delayed repayment 
delayed_payment_age = table.loc[table['SeriousDlqin2yrs'] == 1, 'age'].values

np.random.seed(0)

#Generate a sample using bootstrapping and calculate the mean 
loyal_mean_scores = [np.mean(sample) 
                       for sample in get_bootstrap_samples(delayed_payment_age, 1000)]

stat_intervals(loyal_mean_scores, 0.1)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
lr = LogisticRegression(random_state= 5, class_weight='balanced')
parameters = {'C': (0.0001, 0.001, 0.01, 0.1, 1, 10)}
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=5)
gcv = GridSearchCV(lr,parameters, cv=skf, scoring='roc_auc')
gcv.fit(X, y)
gcv.best_estimator_
gcv.best_params_
best_rocauc = gcv.best_score_
stdev = gcv.cv_results_['std_test_score'][gcv.best_index_] 
print(stdev*100)
lr_best = LogisticRegression(C=0.001, class_weight='balanced', dual=False,
          fit_intercept=True, intercept_scaling=1, max_iter=100,
          multi_class='warn', n_jobs=None, penalty='l2', random_state=5,
          solver='warn', tol=0.0001, verbose=0, warm_start=False)
lr_roc_auc = gcv.best_score_
print(lr_roc_auc)
#normalizing
from sklearn import preprocessing
import numpy as np
X_norm = preprocessing.scale(X)

#coefficients
lr_best.fit(X_norm, y)
print(lr_best.coef_)
from sklearn.utils.extmath import softmax
import numpy as np

np.set_printoptions(formatter={'float_kind':'{:f}'.format})
X_softmax = softmax(X_norm)
lr_best.fit(X_softmax, y)
print(lr_best.coef_)
#recalculating logisitc regression
lr_best.fit(X, y)

#modifying age
X1 = X.copy()
X1['age']= X['age'] + 20
X1.head()

#computing predictions with original age values and +20 age values
pred_X = lr_best.predict(X)
pred_X1 = lr_best.predict(X1)

#comparing predicted values
unique, counts = np.unique(pred_X, return_counts=True)
a = dict(zip(unique, counts))

unique, counts = np.unique(pred_X1, return_counts=True)
b = dict(zip(unique, counts))

print(a,b)
34673/(34673+10390)
38853/(6210+38853)
0.8621929298981426/0.7694339036460067
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42, 
                            class_weight='balanced')
parameters = {'max_features': [1, 2, 4], 'min_samples_leaf': [3, 5, 7, 9], 'max_depth': [5,10,15]}
gcv2 = GridSearchCV(rf,parameters, cv=skf, scoring='roc_auc')
gcv2.fit(X, y)
gcv2.best_estimator_

#best parametrs 
# max_features = 2,
# min_samples_leaf = 7,
# max_depth = 10
rf_best = RandomForestClassifier(bootstrap=True, class_weight='balanced',
            criterion='gini', max_depth=10, max_features=2,
            max_leaf_nodes=None, min_impurity_decrease=0.0,
            min_impurity_split=None, min_samples_leaf=7,
            min_samples_split=2, min_weight_fraction_leaf=0.0,
            n_estimators=100, n_jobs=-1, oob_score=False, random_state=42,
            verbose=0, warm_start=False)
rf_best.fit(X,y)
#roc-auc score for rf
rf_roc_auc = gcv2.best_score_
(rf_roc_auc-lr_roc_auc)*100
importances = rf_best.feature_importances_
print(importances, "minimal value is", min(importances))
from sklearn.ensemble import BaggingClassifier
from sklearn.model_selection import cross_val_score, RandomizedSearchCV

parameters = {'max_features': [2, 3, 4], 'max_samples': [0.5, 0.7, 0.9], 
              'base_estimator__C': [0.0001, 0.001, 0.01, 1, 10, 100]}
from sklearn.ensemble import BaggingClassifier
bg = BaggingClassifier(LogisticRegression(),n_estimators = 100, random_state = 42)
rcv = RandomizedSearchCV(bg, parameters, n_iter = 20, cv = skf, scoring = 'roc_auc', random_state = 1)
rcv.fit(X, y)
rcv.best_estimator_
rcv.best_score_