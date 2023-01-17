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
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.metrics import log_loss
from sklearn.model_selection import KFold, train_test_split
import lightgbm as lgb
train = pd.read_csv("/kaggle/input/lish-moa/train_features.csv")
test = pd.read_csv("/kaggle/input/lish-moa/test_features.csv")
train_targets = pd.read_csv("/kaggle/input/lish-moa/train_targets_scored.csv")
train_targets_nonscored = pd.read_csv("/kaggle/input/lish-moa/train_targets_nonscored.csv")
sub = pd.read_csv("/kaggle/input/lish-moa/sample_submission.csv")
train.head()
test.head()
train_targets.head()
len(train), len(test), len(train_targets)
train.dtypes[train.dtypes == 'object'].index
train['cp_type'].value_counts()
train_columns = train.columns.to_list()
g_list = [i for i in train_columns if i.startswith('g-')]
c_list = [i for i in train_columns if i.startswith('c-')]
train = pd.get_dummies(data=train,columns=['cp_type', 'cp_dose'], drop_first=True)
test = pd.get_dummies(data=test, columns=['cp_type', 'cp_dose'], drop_first=True)
X = train.drop('sig_id', axis=1)

y = train_targets.drop('sig_id', axis=1)

x_test = test.drop('sig_id', axis=1)
X_train, X_validation, y_train, y_validation = train_test_split(X, y, train_size=0.8, random_state=0)
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import log_loss

from catboost import CatBoostClassifier

# META CODE
from sklearn.multiclass import OneVsRestClassifier, OneVsOneClassifier
from sklearn.preprocessing import MultiLabelBinarizer
#mlb = MultiLabelBinarizer()
#y = mlb.fit_transform(y)
model = OneVsRestClassifier(CatBoostClassifier(objective='MultiClass', task_type="GPU",l2_leaf_reg=50,
                                              bootstrap_type='Bernoulli' ,leaf_estimation_iterations=10,subsample=0.9,random_seed=10))
#model.fit(X_train, y_train,[(X_train,y_train),(X_validation, y_validation)],early_stopping_rounds=300)
model.fit(X, y)
prediction = model.predict_proba(x_test)
prediction
#into dataframe
df4 = pd.DataFrame(prediction, columns=train_targets.columns[1:]).round(4)
df4.insert(0,'sig_id',pd.DataFrame(sub)['sig_id'].values)
df4.head()
#submission
df4.to_csv('submission.csv',index = False)
df4.head()
