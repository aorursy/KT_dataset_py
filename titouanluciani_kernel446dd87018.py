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

sample_submission = pd.read_csv("../input/ieee-fraud-detection/sample_submission.csv")

test_identity = pd.read_csv("../input/ieee-fraud-detection/test_identity.csv")

test_transaction = pd.read_csv("../input/ieee-fraud-detection/test_transaction.csv")

train_identity = pd.read_csv("../input/ieee-fraud-detection/train_identity.csv")

train_transaction = pd.read_csv("../input/ieee-fraud-detection/train_transaction.csv")
train = pd.concat([train_identity.iloc[:,0:6], train_transaction.iloc[:,0:6]], axis=1)

test = pd.concat([test_identity.iloc[:,0:6], test_transaction.iloc[:,0:6]], axis=1)
from sklearn.pipeline import make_pipeline

from sklearn.impute import SimpleImputer

from sklearn.preprocessing import OneHotEncoder, StandardScaler

from sklearn.compose import ColumnTransformer, make_column_transformer
X = train.drop(['isFraud'],axis=1)

y = train.isFraud
X.ProductCD.fillna(value='most_frequent').value_counts()

X.ProductCD.isnull().any().sum()
X_num = X.select_dtypes(exclude='object')

list(X_num)
col_trans = make_column_transformer((list(X_num), SimpleImputer(strategy='mean')),

                              (list(X_num), StandardScaler()),

                              (['ProductCD'],OneHotEncoder(sparse=False)))
col_trans.fit_transform(X)
pipe_cat = make_pipeline(OneHotEncoder(sparse=False))

pipe_num = make_pipeline(SimpleImputer(strategy='most_frequent'),(StandardScaler()))

                         

col_trans2 = make_column_transformer((list(X_num), pipe_num),

                                     (['ProductCD'], pipe_cat), remainder='passthrough')
from sklearn.model_selection import cross_val_score

from sklearn.linear_model import LogisticRegression



model = make_pipeline(col_trans2, LogisticRegression())
model.fit(X,y)
model.score(X,y)
scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
scores
pred = model.predict_proba(test)

prob = pred[:,1]
sample_submission
prob.shape

trid = test.TransactionID.iloc[:,1]

trid
submission = pd.DataFrame({'TransactionID':trid, 'isFraud':pred[:,1]})
submission
filename = 'Fraud Detection.csv'



submission.to_csv(filename,index=False)