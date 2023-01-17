import time

import pandas as pd

import numpy as np

import seaborn as sns

from sklearn.preprocessing import LabelEncoder

from sklearn.utils import resample

from xgboost import XGBClassifier

from sklearn.metrics import roc_auc_score,precision_recall_curve,roc_curve

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn.manifold import TSNE

from sklearn.decomposition import PCA, TruncatedSVD
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
# Transaction CSVs

train_transaction = pd.read_csv('../input/train_transaction.csv')

test_transaction = pd.read_csv('../input/test_transaction.csv')

# Identity CSVs - These will be merged onto the transactions to create additional features

train_identity = pd.read_csv('../input/train_identity.csv')

test_identity = pd.read_csv('../input/test_identity.csv')

# Sample Submissions

ss = pd.read_csv('../input/sample_submission.csv')
train_transaction.describe()
train_identity.describe()
train = train_transaction.merge(train_identity,how='left',left_index=True,right_index=True)

y_train = train['isFraud']
X_train, X_test, y_train, y_test= train_test_split( train.drop('isFraud',axis=1), y_train, test_size=.2 , random_state=1 )
train.isna().sum()[train.isna().sum() > 0]
train_transaction.isFraud.value_counts() / train_transaction.shape[0]