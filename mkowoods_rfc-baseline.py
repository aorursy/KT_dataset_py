# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.metrics import roc_auc_score, accuracy_score

from sklearn.ensemble import RandomForestClassifier

train_df = pd.read_csv('../input/train.csv')

test_df = pd.read_csv('../input/test.csv')
x_cols = ['var_%d'%i for i in range(200)]
X_train = train_df[x_cols].values

y_train = train_df['target'].values

X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=42)
rfc = RandomForestClassifier(class_weight='balanced', n_estimators=50)

rfc.fit(X_train, y_train)
def find_best_cutoff(y_true, y_probs):

    best_prob = None

    best_roc = 0.0

    for i in range(0, 100):

        prob = i/100

        roc = roc_auc_score(y_true, y_probs > prob)

        if roc > best_roc:

            best_roc = roc

            best_prob = prob

    return {'best_prob': best_prob, 'best_roc': best_roc}
y_val_prob = rfc.predict_proba(X_val)

best_cut = find_best_cutoff(y_val, y_val_prob[:, 1])

print('ROC:', roc_auc_score(y_val, y_val_prob[:, 1] > best_cut['best_prob']))

print('Accuracy:', accuracy_score(y_val, (y_val_prob[:, 1] > best_cut['best_prob']).astype(int) ))
X_test = test_df[x_cols].values

y_test_pred = rfc.predict_proba(X_test)[:, 1] > best_cut['best_prob']
sub_df = pd.DataFrame({'ID_code': test_df['ID_code'].values})

sub_df['target'] = y_test_pred.astype(int)

sub_df.to_csv('submission.csv', index=False)