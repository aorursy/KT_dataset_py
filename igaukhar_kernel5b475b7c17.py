# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import pandas as pd

test = pd.read_csv("../input/santander-customer-transaction-prediction-dataset/test.csv")

train = pd.read_csv("../input/santander-customer-transaction-prediction-dataset/train.csv")
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import roc_auc_score

from sklearn.linear_model import LogisticRegression

%matplotlib inline

from matplotlib import pyplot as plt

from sklearn.model_selection import train_test_split,  cross_val_score
y = train['target']

X = train.drop(['target','ID_code'],axis=1)

X_test = test.drop('ID_code', axis=1)
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.3, random_state=200)
from scipy.sparse import csr_matrix
X_train.shape, X_valid.shape
logit = LogisticRegression(n_jobs=-1, random_state=17)

logit.fit(X_train, y_train)
y_pred = logit.predict(X_test)
y_pred.shape
def get_auc_lr_valid(X, y, C=1.0, ratio = 0.9, seed=17):

 

    logit = LogisticRegression( C=C, n_jobs=-1, random_state=seed)

    logit.fit(X_train, y_train)        

    valid_pred = logit.predict_proba(X_valid)[:, 1]

    return roc_auc_score(y_valid, valid_pred)
get_auc_lr_valid(X_train, y_train)
sub_df = pd.DataFrame({'ID_code':test.ID_code.values})

sub_df['target'] = y_pred

sub_df.to_csv('submission.csv', index=False)