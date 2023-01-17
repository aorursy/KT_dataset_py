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
data.describe().T
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
table.describe().T
X = table[independent_columns_names]
y = table['SeriousDlqin2yrs']
def get_bootstrap_samples(data,n_samples):
    indicies = np.random.randint(0,len(data),(n_samples,len(data))) 
    samples = data[indicies]
    return samples

def compute_intervals(stat, alpha):
    intervals = np.percentile(stat,[100*alpha/2.,100*(1-alpha/2.)])
    return intervals
np.random.seed(0)

bad_customers = table[table['SeriousDlqin2yrs']==1]
age_mean_distribs = [np.mean(sample) for sample in get_bootstrap_samples(bad_customers.age.values,10000)]
print("The bad client's mean age confidence interval (confidence level = 90%): ",compute_intervals(age_mean_distribs,0.1))
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
lr = LogisticRegression(random_state=5, class_weight='balanced')
parameters = {'C': (0.0001, 0.001, 0.01, 0.1, 1, 10)}
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=5)
lr_cv = GridSearchCV(lr,parameters,cv=skf,n_jobs=-1,verbose=1,scoring='roc_auc')
lr_cv.fit(X,y)
lr_cv.best_score_,lr_cv.best_params_
# Your code herelr
lr_cv.cv_results_['std_test_score'][1]*100
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X = scaler.fit_transform(X)
lr = LogisticRegression(C=0.001,class_weight='balanced')
lr.fit(X,y)
# lr_pipe = make_pipeline(Normalizer(),LogisticRegression(C=0.0001))
# lr_pipe.fit(X,y)
lr.coef_

independent_columns_names
lr_coef = pd.DataFrame(lr.coef_.reshape(7,1),index =independent_columns_names,columns=['raw'])
lr_coef['abs'] = lr_coef['raw'].apply(np.abs)
lr_coef
def softmax(X):
    return np.exp(X) / np.sum(np.exp(X))

softmax(lr_coef['raw'].values)
# Your code here
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42, 
                            class_weight='balanced')
parameters = {'max_features': [1, 2, 4], 'min_samples_leaf': [3, 5, 7, 9], 'max_depth': [5,10,15]}
rf_cv = GridSearchCV(rf,parameters,cv=skf,n_jobs=-1,verbose=1,scoring='roc_auc')
rf_cv.fit(X,y)
rf_cv.best_score_ - lr_cv.best_score_
rf_cv.best_estimator_.feature_importances_
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