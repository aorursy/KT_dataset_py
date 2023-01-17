import os

import json



import pandas as pd



PATH_TO_DATA = '../input/mlcourse-dota2-win-prediction/'
df_train_features = pd.read_csv(os.path.join('../input/mlcourse-dota2-win-prediction/train_features.csv'), index_col='match_id_hash')



df_train_targets = pd.read_csv(os.path.join('../input/mlcourse-dota2-win-prediction/train_targets.csv'), index_col='match_id_hash')

df_test_features = pd.read_csv(os.path.join('../input/mlcourse-dota2-win-prediction/test_features.csv'), index_col='match_id_hash')
from sklearn.linear_model import LogisticRegression

from sklearn.decomposition import PCA

from sklearn.model_selection import train_test_split

from sklearn.metrics import roc_auc_score

seed=73
X= df_train_features.values

y = df_train_targets['radiant_win'].values
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.3,random_state=seed)
m= PCA(n_components=20, svd_solver='randomized', random_state=seed)

pca=m.fit_transform(X)
X_pca, X_v, y_pca, y_v = train_test_split(pca, y, test_size=0.3,random_state=seed)
logr = LogisticRegression(C=3, random_state=seed,verbose=1)

logr.fit(X_pca,y_pca)

y_pred_prob= logr.predict_proba(X_v)[:, 1]

roc_auc_score(y_v, y_pred_prob)
X_test = m.fit_transform(df_test_features.values)

y_test_pred = logr.predict_proba(X_test)[:, 1]



df_submission = pd.DataFrame({'radiant_win_prob': y_test_pred}, 

                                 index=df_test_features.index)
import datetime

submission_filename = 'submission_{}.csv'.format(

    datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))

df_submission.to_csv(submission_filename)

print('Submission saved to {}'.format(submission_filename))
df_submission.head()