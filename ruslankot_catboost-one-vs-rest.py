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
from sklearn.multiclass import OneVsRestClassifier
from catboost import CatBoostClassifier
from sklearn.preprocessing import MultiLabelBinarizer
train = pd.read_csv('/kaggle/input/lish-moa/train_features.csv')
test = pd.read_csv('/kaggle/input/lish-moa/test_features.csv')
targets = pd.read_csv('/kaggle/input/lish-moa/train_targets_scored.csv')
sub = pd.read_csv('/kaggle/input/lish-moa/sample_submission.csv')
train.head(10)
train.describe()
targets.head(10)
targets.describe()
from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()
train['cp_type'] = labelencoder.fit_transform(train['cp_type'])
test['cp_type']=labelencoder.transform(test['cp_type'])
labelencoder = LabelEncoder()
train['cp_dose'] = labelencoder.fit_transform(train['cp_dose'])
test[ 'cp_dose']=labelencoder.transform(test['cp_dose'])
X_train=train.drop('sig_id', 1)
X_test=test.drop('sig_id', 1)
y_train=targets.drop('sig_id', 1)
clf = OneVsRestClassifier(CatBoostClassifier())

# You may need to use MultiLabelBinarizer to encode your variables from arrays [[x, y, z]] to a multilabel 
# format before training.
mlb = MultiLabelBinarizer()
y = mlb.fit_transform(y_train)

clf.fit(X_train, y_train)
preds=clf.predict(X_test)
sub.iloc[:,1:]=preds
sub.to_csv('submission.csv', index = False)
