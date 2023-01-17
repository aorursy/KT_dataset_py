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
train = pd.read_csv('/kaggle/input/goagoagoa2/train.csv', index_col=0)

test = pd.read_csv('/kaggle/input/goagoagoa2/test.csv',index_col=0)

submission = pd.read_csv('/kaggle/input/goagoagoa2/sampleSubmission.csv',index_col=0)
X_train, y_train = train[test.columns], train['SalePrice']

X_test = test

X_train.shape, X_test.shape
X_train.dtypes.value_counts()
numerical_cols = [col for col in X_train.columns if X_train.dtypes[col] == np.float64]

numerical_cols
categorical_cols = [col for col in X_train.columns if X_train.dtypes[col] != np.float64]

print(categorical_cols)
X = pd.concat([X_train,X_test])

X1 = pd.get_dummies(X).replace(np.nan,0.)

X1.head()
X_train1,X_test1 = X1.iloc[0:1100],X1.iloc[1100:]
X_train2,X_test2 = X_train.copy(), X_test.copy()
from sklearn.preprocessing import *



for col in categorical_cols:

    print(col)

    enc = LabelEncoder().fit(X_train[col].astype(str).to_list()+X_test[col].astype(str).to_list())

    X_train2[col] = enc.transform(X_train[col].astype(str))

    X_test2[col] = enc.transform(X_test[col].astype(str))
from sklearn.ensemble import GradientBoostingRegressor



from xgboost import XGBRegressor

from lightgbm import LGBMRegressor

from sklearn.ensemble import VotingRegressor



from sklearn.neural_network import MLPRegressor



ne = 80

clf1 = GradientBoostingRegressor(n_estimators=ne, random_state=13132)

clf2 = XGBRegressor(n_estimators=ne, random_state=1)

clf3 = LGBMRegressor(n_estimators=ne, random_state=1)

clf4 = MLPRegressor(random_state=1)



sclf = VotingRegressor(

        estimators=[

            #('gbdt',clf1),

            ('xgboost',clf2),

            ('lightgbm',clf3),

#             ('mlp',clf4)

        ],

        n_jobs=-1,

    )

# sclf = LGBMRegressor(n_estimators=200, random_state=19881102)



sclf.fit(X_train2,y_train)
from sklearn.model_selection import KFold

from sklearn.metrics import mean_squared_error



kf = KFold(n_splits=3, random_state=1, shuffle=True)

scores = []

for train_index, test_index in kf.split(X_train2):

    #print(X_train.shape[0], len(train_index), len(test_index))

    X_train_2, y_train_2 = X_train2.iloc[train_index], y_train.iloc[train_index]

    X_val, y_val = X_train2.iloc[test_index], y_train.iloc[test_index]

    

#     sclf = LGBMRegressor(n_estimators=1000, random_state=19881102)

    sclf.fit(X_train_2, y_train_2)

    score = np.sqrt(mean_squared_error(y_val, sclf.predict(X_val)))

    scores.append(score)



np.mean(scores)
#sclf = LGBMRegressor(n_estimators=200, random_state=19881102)

sclf.fit(X_train2, y_train)

y_pred = sclf.predict(X_test2)
submission['SalePrice'] = y_pred
submission.to_csv('./submission.csv')