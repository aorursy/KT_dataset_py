import numpy as np 

import pandas as pd 

import matplotlib.pyplot as plt



from sklearn.metrics import roc_auc_score

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split

from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import RandomizedSearchCV

from sklearn.model_selection import StratifiedKFold

from sklearn.model_selection import cross_val_score

from sklearn.ensemble import RandomForestClassifier

from sklearn.linear_model import LogisticRegression

import seaborn as sns

import random

from scipy.stats import norm

from collections import Counter

import imblearn

from imblearn.over_sampling import SMOTE

from xgboost import XGBClassifier

sns.set_palette('Dark2')

random.seed(42)

pd.set_option('display.max_columns', None)

%matplotlib inline
df_train = pd.read_csv('../input/minor-project-2020/train.csv')

df_test = pd.read_csv('../input/minor-project-2020/test.csv')
df_train.head()
df_train.describe()
df_train.info()
Counter(df_train['target'])
fig, ax = plt.subplots(22, 4, figsize=(16, 44))

for i in range(22):

    for j in range(4):

        sns.distplot(df_train[df_train.columns[4*i+j+1]].values, ax = ax[i][j], fit = norm)

        ax[i][j].set_title(df_train.columns[4*i+j+1])

fig.tight_layout()

fig.show()
df_train_1 = df_train[df_train['target'] == 1]

df_train_1.describe()
df_train.describe()
X_train = df_train[df_train.columns[1:-1]]

y_train = df_train['target']
Counter(df_train['target'])[1]/Counter(df_train['target'])[0]
cormat = df_train.corr()

abs(cormat['target']).sort_values(ascending = False)[-19:].index
abs(cormat['target']).sort_values(ascending = True)[-20:]
#correlation matrix

f, ax = plt.subplots(figsize=(24, 18))

sns.heatmap(cormat, vmax=.25, square=True);
X_train = X_train.drop(['col_68', 'col_86', 'col_45', 'col_31', 'col_47', 'col_41', 'col_64',

       'col_46', 'col_85', 'col_44', 'col_54', 'col_81', 'col_29',

       'col_71', 'col_87', 'col_27', 'col_50', 'col_21', 'col_39'], axis = 1)

columns = X_train.columns
corrmat = X_train.corr()

f, ax = plt.subplots(figsize=(20, 15))

sns.heatmap(corrmat, vmax=.25, square=True);
X_train_, X_val, y_train_, y_val = train_test_split(X_train, y_train, stratify = y_train, test_size=0.2)

oversample = SMOTE(random_state = 42, sampling_strategy=0.5)

X_train_, y_train_ = oversample.fit_resample(X_train_, y_train_)
Counter(y_train_)[1]/Counter(y_train_)[0]
# classifier = RandomForestClassifier(n_estimators = 200, random_state = 42, verbose = 2, max_depth = 16, min_samples_split = 2, min_samples_leaf = 2, n_jobs=-1)

# classifier.fit(X_train_, y_train_)

# y_pred = classifier.predict_proba(X_val)

# print("AUC: ", roc_auc_score(y_val, np.squeeze(y_pred[:, 1:])))
# print("AUC: ", roc_auc_score(y_val, np.squeeze(y_pred[:, 1:])))
#Submission for Random Forest Code. Not the final submission

# y_pred_rf = classifier.predict_proba((df_test[columns]))

# test_ids = df_test['id'].values

# data = {'id':test_ids, 'target':np.squeeze(y_pred_rf[:, 1:])}

# sample_df = pd.DataFrame(data)

# sample_df.head()

# sample_df.to_csv("submission_rf.csv", index=False)
X_train_, X_val, y_train_, y_val = train_test_split(X_train, y_train, stratify = y_train, test_size=0.2)

oversample = SMOTE(random_state = 42, sampling_strategy=0.8)

X_train_, y_train_ = oversample.fit_resample(X_train_, y_train_)

params = {'C': [0.7, 0.8, 0.9, 1]}

regressor=LogisticRegression(max_iter = 5000)

regressor_gs = GridSearchCV(regressor, params, cv = 5, scoring = 'roc_auc', verbose = 2)

scalar = StandardScaler()

X_train_scaled = scalar.fit_transform(X_train_)

regressor_gs.fit(X_train_scaled, y_train_)



print(regressor_gs.best_estimator_)
regressor_gs.cv_results_
#Best results till now is for c=0.8.

scalar = StandardScaler()

X_train_ = scalar.fit_transform(X_train_)

regressor = LogisticRegression(max_iter = 5000, C = 1)

regressor.fit(X_train_, y_train_)

prob_val = (regressor.predict_proba(scalar.transform(X_val)))

print("AUC: ", roc_auc_score(y_val, np.squeeze(prob_val[:, 1:])))

y_pred = regressor.predict_proba(scalar.transform(df_test[columns]))

test_ids = df_test['id'].values

data = {'id':test_ids, 'target':np.squeeze(y_pred[:, 1:])}

sample_df = pd.DataFrame(data)

sample_df.head()

sample_df.to_csv("submission_logreg.csv", index=False)