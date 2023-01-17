import numpy as np

import pandas as pd

import seaborn as sns

from sklearn.decomposition import PCA

from sklearn.ensemble import RandomForestClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import cross_val_score

from sklearn.metrics import accuracy_score, log_loss

%pylab inline
X_train = pd.read_csv('/kaggle/input/lish-moa/train_features.csv')

y_train_scored = pd.read_csv('/kaggle/input/lish-moa/train_targets_scored.csv')

y_train_nonscored = pd.read_csv('/kaggle/input/lish-moa/train_targets_nonscored.csv')
print(X_train.shape)

X_train.head()
print(y_train_scored.shape)

y_train_scored.head()
X_test = pd.read_csv('/kaggle/input/lish-moa/test_features.csv')

X_test.head()
dict_cp_type = {'trt_cp': 1, 'ctl_vehicle': 0}

dict_cp_dose = {'D1': 1, 'D2': 2}

X_train.cp_type = X_train.cp_type.apply(lambda x: dict_cp_type[x])

X_train.cp_dose = X_train.cp_dose.apply(lambda x: dict_cp_dose[x])

X_test.cp_type = X_test.cp_type.apply(lambda x: dict_cp_type[x])

X_test.cp_dose = X_test.cp_dose.apply(lambda x: dict_cp_dose[x])
X_train_no_id = X_train.iloc[:, 1:]

X_test_no_id = X_test.iloc[:, 1:]
submission = pd.read_csv('/kaggle/input/lish-moa/sample_submission.csv')

submission.head()
disbalanced = []

cv_scores = {}



for m, y in y_train_scored.iloc[:, 1:].items():

    lr = LogisticRegression(solver='liblinear', penalty='l1')

    if y.value_counts().min() < 5:

        disbalanced.append(m)

    else:

        cv_scores[m] = -cross_val_score(lr, X_train_no_id, y, scoring='neg_log_loss').mean()
len(cv_scores)
disbalanced
sorted_cv_scores = {k: v for k, v in sorted(cv_scores.items(), key=lambda x: x[1], reverse=True)}



bad_logregr = {k: v for k, v in sorted_cv_scores.items() if v > 0.018}



pd.DataFrame.from_dict(sorted_cv_scores, orient='index', columns=['cv_score']).to_csv('cv_scores.csv')



len(bad_logregr)
from sklearn.linear_model import LogisticRegression



for m, y in y_train_scored.iloc[:, 1:].items():

    if m in bad_logregr.keys():

        lr = LogisticRegression(C=0.1, solver='liblinear', penalty='l1')

    else:

        lr = LogisticRegression(solver='liblinear', penalty='l1')

    lr.fit(X_train_no_id, y)

    submission[m] = lr.predict_proba(X_test_no_id)[:, 1]
submission.head()
submission.to_csv('submission.csv', index=False)