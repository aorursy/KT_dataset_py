import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
train_data = pd.read_csv("/kaggle/input/cat-in-the-dat-ii/train.csv")

test_data = pd.read_csv("/kaggle/input/cat-in-the-dat-ii/test.csv")
all_data = pd.concat([train_data, test_data])
all_data.head()
all_data.tail()
all_data.nom_6.unique()
target = train_data['target']

train_id = train_data['id']

test_id = test_data['id']

train_data.drop(['target', 'id'], axis=1, inplace=True)

test_data.drop(['id'], axis=1, inplace=True)



print(train_data.shape)

print(test_data.shape)
train_data.isnull().sum()
train_data.isnull().sum()
for col in train_data.columns:

    all_data[col].fillna(train_data[col].mode()[0], inplace = True)

    train_data[col].fillna(train_data[col].mode()[0], inplace = True)

    test_data[col].fillna(train_data[col].mode()[0], inplace = True)
train_data.bin_0.unique()
(train_data.bin_0.unique() == test_data.bin_0.unique()).sum() == (train_data.bin_0.nunique())
train_data.columns
from sklearn.preprocessing import LabelEncoder



for col in train_data.columns:

    print(col)

    le = LabelEncoder()

    

    le.fit(all_data[col])

    train_data[col] = le.transform(train_data[col])

    test_data[col] = le.transform(test_data[col])
x = train_data

y = target



from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state = 2, stratify = y)
import lightgbm as lgbm
lgbc = lgbm.LGBMClassifier(random_state = 2)

lgbc.fit(x_train, y_train)





y_pred = lgbc.predict(x_test)

from sklearn.metrics import accuracy_score

accuracy_score(y_test, y_pred)
from sklearn.linear_model import LogisticRegression

logreg = LogisticRegression(random_state = 2, solver='lbfgs', max_iter=200, C=0.085)

logreg.fit(x_train, y_train)





y_pred = logreg.predict(x_test)

from sklearn.metrics import accuracy_score

accuracy_score(y_test, y_pred)
submission = pd.DataFrame()

submission['id'] = test_id

# for col in test_data.columns:

#     test_data[col] = test_data[col].astype('category')

submission['target'] = logreg.predict_proba(test_data)[:,1]
submission.to_csv('submission.csv', index= None)
submission.head()