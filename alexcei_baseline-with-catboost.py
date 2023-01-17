# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
train = pd.read_csv("../input/flight-delays-fall-2018/flight_delays_train.csv.zip")
test = pd.read_csv("../input/flight-delays-fall-2018/flight_delays_test.csv.zip")
train.head()
test.head()
from sklearn.model_selection import TimeSeriesSplit, cross_val_score

time_split = TimeSeriesSplit(n_splits=5)
X_train, y_train = train[['Distance', 'DepTime']].values, train['dep_delayed_15min'].map({'Y': 1, 'N': 0}).values
X_test = test[['Distance', 'DepTime']].values
from sklearn.model_selection import train_test_split

X_train_part, X_valid, y_train_part, y_valid = train_test_split(X_train, y_train, test_size=0.3, random_state=17)
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_scaler = scaler.fit_transform(X_train)
X_test_scaler = scaler.transform(X_test)
X_train_part = scaler.transform(X_train_part)
X_valid = scaler.transform(X_valid)
X_valid[:5]
time_split
%%time
from sklearn.linear_model import LogisticRegression

logit = LogisticRegression()

logit.fit(X_train_part, y_train_part)
logit_valid_pred = logit.predict_proba(X_valid)[:, 1]

print('Train test split LogisticRegression score:% s ROC AUC' % round(roc_auc_score(y_valid, logit_valid_pred), 4))
cross_score_lr = np.mean(cross_val_score(logit, X_train, y_train, scoring = 'roc_auc', cv=time_split))
print('Cross validation LogisticRegression score:% s ROC AUC' % round(cross_score_lr, 4))
%%time
from catboost import CatBoostClassifier

ctb = CatBoostClassifier(random_seed=17, verbose=0)
ctb.fit(X_train_part, y_train_part, verbose=0)
logit_valid_pred = ctb.predict_proba(X_valid)[:, 1]
print('Train test split CatBoostClassifier score:% s ROC AUC' % round(roc_auc_score(y_valid, logit_valid_pred), 4))
cross_score_lr = np.mean(cross_val_score(ctb, X_train, y_train, scoring = 'roc_auc', cv=2, verbose=0)) # time_split
print('Cross validation CatBoostClassifier score:% s ROC AUC' % round(cross_score_lr, 4))
logit.fit(X_train_scaler, y_train)
pred = logit.predict_proba(X_test_scaler)[:, 1]

pd.Series(pred, name='dep_delayed_15min').to_csv('logit_sub.csv', index_label='id', header=True)
ctb.fit(X_train_scaler, y_train, verbose=0)
pred = ctb.predict_proba(X_test_scaler)[:, 1]

pd.Series(pred, name='dep_delayed_15min').to_csv('ctb_sub.csv', index_label='id', header=True)
