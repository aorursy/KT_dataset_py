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

train = pd.read_csv('/kaggle/input/sh-coe-mlp-1/train.csv',index_col=0)

train.head()
# numerical feature
train.dtypes
train['proline']
type(train)

train.columns
train.index
train['target'].value_counts()
train['alcohol'].describe()
test = pd.read_csv('/kaggle/input/sh-coe-mlp-1/test.csv',index_col=0)

test.head()
X_train, y_train = train[test.columns], train['target']

X_test = test
X_train
from sklearn.preprocessing import *



transformer = MinMaxScaler() # 0-1

transformer = StandardScaler() # -1,+1, mean=0



transformer.fit(X_train)

X_train = transformer.transform(X_train)

X_test = transformer.transform(X_test)
X_train.describe()
from sklearn.ensemble import GradientBoostingClassifier

from xgboost import XGBClassifier

from lightgbm import LGBMClassifier



clf = LGBMClassifier()

clf.fit(X_train_1,y_train)
submission = pd.read_csv('/kaggle/input/sh-coe-mlp-1/sampleSubmission.csv',index_col=0)

submission.head()
train['is_train'] = 1

test['is_train'] = 0

test['target'] = -1



data = pd.concat([train, test], axis=0)

data.head()
X = data[[col for col in data.columns if col not in ['target','is_train']]].values

y = data['target'].values



X
X.shape, y.shape
from sklearn.preprocessing import StandardScaler



X = StandardScaler().fit_transform(X)

X
X_train, y_train = X[data.is_train==1], y[data.is_train==1]

X_test = X[data.is_train==0]
from sklearn.ensemble import GradientBoostingClassifier

from xgboost import XGBClassifier

from lightgbm import LGBMClassifier

from sklearn.ensemble import VotingClassifier



clf1 = GradientBoostingClassifier(n_estimators=6, random_state=13132)

clf2 = XGBClassifier(n_estimators=6, random_state=1)

clf3 = LGBMClassifier(n_estimators=6, random_state=1)



sclf = VotingClassifier(

        estimators=[

            ('gbdt',clf1),

            ('xgboost',clf2),

            ('lightgbm',clf3)

        ],

        n_jobs=-1,

    )



sclf.fit(X_train,y_train)
from sklearn.model_selection import KFold

from sklearn.metrics import classification_report, accuracy_score



kf = KFold(n_splits=3, random_state=1, shuffle=True)

scores = []

for train_index, test_index in kf.split(X_train):

    print(X_train.shape[0], len(train_index), len(test_index))

    X_train_1, y_train_1 = X_train[train_index], y_train[train_index]

    X_val_1, y_val_1 = X_train[test_index], y_train[test_index]

    

    clf = GradientBoostingClassifier(n_estimators=6, random_state=1)

    clf.fit(X_train_1, y_train_1)

    print("train accuracy", accuracy_score(y_train_1, clf.predict(X_train_1)))

    print("validation accuracy", accuracy_score(y_val_1, clf.predict(X_val_1)))

    scores.append(accuracy_score(y_val_1, clf.predict(X_val_1)))

    

    #print(classification_report(y_val_1, sclf.predict(X_val_1)))



np.mean(scores)
sclf.fit(X_train,y_train)
y_pred = sclf.predict(X_test)
submission['target'] = y_pred

submission.to_csv('./submission.csv')
!head submission.csv