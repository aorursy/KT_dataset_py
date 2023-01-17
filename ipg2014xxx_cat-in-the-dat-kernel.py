# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

%matplotlib inline



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
train_data = pd.read_csv("/kaggle/input/cat-in-the-dat/train.csv", index_col='id')

test_data = pd.read_csv("/kaggle/input/cat-in-the-dat/test.csv", index_col='id')
train_data.head()
y = train_data.pop('target')
print("Training data shape =", train_data.shape)

print("Testing data shape =", test_data.shape)
null_val_train = [(x, y) for (x, y) in train_data.isnull().sum().items() if y > 0]

null_val_test = [(x, y) for (x, y) in test_data.isnull().sum().items() if y > 0]

print("Null Values in training data =", len(null_val_train))

print("Null Values in testing data =", len(null_val_test))
for col in train_data.columns:

    print(col, train_data[col].nunique())
full_data = pd.concat([train_data, test_data], axis=0)

train_rows = train_data.shape[0]

del train_data

del test_data



full_data.shape
full_data['bin_3'] = full_data['bin_3'].map({'T': 1, 'F': 0})

full_data['bin_4'] = full_data['bin_4'].map({'Y': 1, 'N': 0})
for col in ['ord_1', 'ord_2', 'ord_3', 'ord_4']:

    print(col, list(np.unique(full_data[col])))
full_data['ord_5_len'] = full_data['ord_5'].map(len)

full_data['ord_5_len'] -= 2

full_data['ord_5_len'].map(abs).any()
m1 = {'Contributor': 1, 'Expert': 2, 'Grandmaster': 4, 'Master': 3, 'Novice': 0}

full_data['ord_1'] = full_data['ord_1'].map(m1)



m2 = {'Boiling Hot': 4, 'Cold': 1, 'Freezing': 0, 'Hot': 3, 'Lava Hot': 5, 'Warm': 2}

full_data['ord_2'] = full_data['ord_2'].map(m2)



full_data['ord_3'] = full_data['ord_3'].apply(lambda x: ord(x) - ord('a'))

full_data['ord_4'] = full_data['ord_4'].apply(lambda x: ord(x) - ord('A'))



full_data['ord_5a'] = full_data['ord_5'].str[0]

full_data['ord_5b'] = full_data['ord_5'].str[1]

full_data['ord_5a'] = full_data['ord_5a'].map({val : idx for idx, val in enumerate(np.unique(full_data['ord_5a']))})

full_data['ord_5b'] = full_data['ord_5b'].map({val : idx for idx, val in enumerate(np.unique(full_data['ord_5b']))})

full_data.drop(['ord_5', 'ord_5_len'], axis=1, inplace=True)
full_data[['nom_7', 'nom_8', 'nom_9']].head()
full_data.drop(['nom_7', 'nom_8', 'nom_9'], axis=1, inplace=True)
full_data['day'] = full_data['day']/7.0

full_data['month'] = full_data['month']/12.0
full_data = pd.get_dummies(data=full_data, columns=['nom_0', 'nom_1', 'nom_2', 'nom_3', 'nom_4', 'nom_5', 'nom_6'], prefix_sep='_', sparse=True)
full_data.head()
train_data = full_data[:train_rows]

test_data = full_data[train_rows:]

del full_data
train_data.shape
from sklearn.linear_model import LogisticRegression



lr = LogisticRegression(solver='sag', class_weight='balanced', random_state=0, max_iter=200, n_jobs=-1)
from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import StratifiedKFold



kfold = StratifiedKFold(n_splits=5, random_state=0,)

params = {'C': [0.79]}



grid = GridSearchCV(estimator=lr,

                  param_grid=params,

                  cv=kfold

                  )
grid.fit(train_data, y)
model = grid.best_estimator_

print(grid.best_score_)
y_pred = model.predict_proba(test_data)
y_pred = y_pred[:, 1]
y_pred = pd.Series(y_pred)

test_df = pd.DataFrame([test_data.index, y_pred]).transpose()

test_df.columns = ['id', 'target']
test_df['id'] = test_df['id'].astype(int)

test_df.head()
test_df.to_csv('submission.csv', index=False)