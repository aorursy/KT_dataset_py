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
import pandas as pd

train = pd.read_csv('/kaggle/input/home-loan/train.csv')

test = pd.read_csv('/kaggle/input/home-loan/test.csv')

print(train.shape, test.shape)

print(train.dtypes)
train = train.drop(['Loan_ID'], axis=1)

test =  test.drop(['Loan_ID'], axis=1)

train = train.apply(lambda x:x.fillna(x.value_counts().index[0]))

test = test.apply(lambda x:x.fillna(x.value_counts().index[0]))
feature_set = train.drop(['Loan_Status'], axis=1)

X = feature_set.columns[:len(feature_set.columns)]

y = 'Loan_Status'

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(

    train[X], train[y], random_state=0)
from sklearn.compose import ColumnTransformer

from sklearn.preprocessing import Normalizer, OneHotEncoder

colT = ColumnTransformer(

    [("dummy_col", OneHotEncoder(categories=[['Male', 'Female'],

                                           ['Yes', 'No'],

                                            ['0','1', '2','3+'],

                                            ['Graduate', 'Not Graduate'],

                                            ['No', 'Yes'],

                                            ['Semiurban', 'Urban', 'Rural']]), [0,1,2,3,4,10]),

      ("norm", Normalizer(norm='l1'), [5,6,7,8,9])])
X_train = colT.fit_transform(X_train)

X_train
X_test = colT.transform(X_test)
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score, classification_report

random_forest = RandomForestClassifier()

random_forest.fit(X_train, y_train)

y_pred = random_forest.predict(X_test)

print(classification_report(y_test, y_pred, target_names=['Y', 'N']))
test_samp = test[:15]

test_samp = colT.transform(test_samp)

random_forest.predict(test_samp)