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
train_df = pd.read_csv("../input/cat-in-the-dat/train.csv")

train_df.head()
y = train_df.iloc[:,24]

X = train_df.iloc[:,1:24]
X = X.applymap(str)

X.head()
X.dtypes
from sklearn.model_selection import train_test_split



X_train, X_val, Y_train, Y_val = train_test_split(X, y, test_size=0.2)
cat_features = list(range(0,23))
from catboost import CatBoostClassifier
model = CatBoostClassifier(iterations = 100, learning_rate = 0.1, depth = 15)
model.fit(X_train, Y_train, cat_features)
from sklearn.metrics import accuracy_score

val_preds = model.predict(X_val)

print(accuracy_score(Y_val, val_preds))
test_df = pd.read_csv("../input/cat-in-the-dat/test.csv")

test_df.head()
test_data = test_df.iloc[:,1:24]

test_data = test_data.applymap(str)
test_data.dtypes
test_preds = model.predict(test_data)
# Save test predictions to file

# no aug performed better

output = pd.DataFrame({'id': test_df.id,

                       'target': test_preds})

output.to_csv('submission.csv', index=False)