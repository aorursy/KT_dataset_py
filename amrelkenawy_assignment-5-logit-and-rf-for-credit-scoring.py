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

delay_age = data.loc[data['SeriousDlqin2yrs'] == 1,'age'].values
np.random.seed(0)

delay_age_scores = [np.mean(sample) 
                       for sample in get_bootstrap_samples(delay_age, 1000)]
print("mean interval", stat_intervals(delay_age_scores, 0.05))

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
lr = LogisticRegression(random_state=5, class_weight='balanced')
parameters = {'C': (0.0001, 0.001, 0.01, 0.1, 1, 10)}
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=5)
gcv = GridSearchCV(lr, parameters, n_jobs=-1, cv=skf, verbose=1,scoring='roc_auc')
gcv.fit(X,y)
gcv.best_params_

best_rocauc = gcv.best_score_
std_val = gcv.cv_results_['std_test_score'][gcv.best_index_] * 100 
print ('Standard deviation % on validation = ' + str(std_val) )
print ('LR ROC AUC SCORE % = ' + str(gcv.best_score_* 100))
from sklearn import preprocessing

min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(X.values)
X_norm = pd.DataFrame(x_scaled)
X_norm.mean()


#Double check with correlation matrix and heat map

sns.heatmap(data.corr(),annot=True)

#Highest correlation is between age and seriousdlqin2yrs 
def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=0)

#softmax(X_norm[2])
#gcv.fit(X,y)
#gcv.best_params_

#softmax should be applied to outputs not normalized features
# but which value is demanded in this question? 
# How much Debt Ratio affection could be calculated?

from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42, 
                            class_weight='balanced')
parameters = {'max_features': [1, 2, 4], 'min_samples_leaf': [3, 5, 7, 9], 'max_depth': [5,10,15]}
gcv2 = GridSearchCV(rf, parameters, n_jobs=-1, cv=skf, verbose=1,scoring='roc_auc')
gcv2.fit(X,y)

print ('ROC AUC score of RF is higher by ' + str((gcv2.best_score_ - gcv.best_score_)*100) + '%')
rf.fit(X,y)
rf.feature_importances_

rf.feature_importances_.min()

from sklearn.ensemble import BaggingClassifier
from sklearn.model_selection import cross_val_score, RandomizedSearchCV

parameters = {'max_features': [2, 3, 4], 'max_samples': [0.5, 0.7, 0.9], 
              'base_estimator__C': [0.0001, 0.001, 0.01, 1, 10, 100]}
bg = BaggingClassifier(base_estimator=100,random_state=42 )
bg.fit
rcv = RandomizedSearchCV(bg, parameters, cv=skf,n_iter=20, random_state=1,scoring='roc_auc')
#Roc auc score is missing