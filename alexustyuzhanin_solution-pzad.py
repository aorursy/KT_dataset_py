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

import numpy as np

from sklearn.preprocessing import OneHotEncoder
#Load training data, sort by id

train = pd.read_csv('/kaggle/input/lish-moa/train_features.csv').sort_values(by='sig_id')

columns = train.columns
train.head()
#drop id

train = train[columns[1:]]
#OHE categorical features

train_cat = train[columns[1:4]]

train.drop(columns[1:4], axis=1, inplace=True)

train_cat.head()
#encode

encoder = OneHotEncoder(categories='auto')

train_cat_OHE = encoder.fit_transform(train_cat).toarray()

train_OHE = pd.concat((train, pd.DataFrame(train_cat_OHE)), axis=1)
#Load training target data, sort by id

train_target_scored = pd.read_csv('/kaggle/input/lish-moa/train_targets_scored.csv').sort_values(by='sig_id')

target_columns = train_target_scored.columns
train_target_scored.head()
#Learn Elastic Net with default parameters

from sklearn.linear_model import ElasticNet



model = ElasticNet()

X, Y = np.array(train_OHE), np.array(train_target_scored[target_columns[1:]])

print("Train X shape is: ", X.shape, '\nTrain Y shape is: ', Y.shape)



model.fit(X, Y)
#Train error

Y_pred = model.predict(X)

error = ((Y - Y_pred)**2).mean(0)

print("Train error by features:\n")

for f_num, col in enumerate(target_columns[1:]):

    print(col, 'feature error is: ', error[f_num])
#Load testing data

test = pd.read_csv('/kaggle/input/lish-moa/test_features.csv')

test.head()
test_id = test[columns[0]].copy()

test = test[columns[1:]]
#Same One-Hot encoding

test_cat = test[columns[1:4]]

test.drop(columns[1:4], axis=1, inplace=True)

test_cat.head()
test_cat_OHE = encoder.transform(test_cat).toarray()

test_OHE = pd.concat((test, pd.DataFrame(test_cat_OHE)), axis=1)
#Test prediction

test_prediction = model.predict(test_OHE)

submission = pd.DataFrame(test_prediction, columns=target_columns[1:])
test_id = pd.DataFrame(test_id, columns=[target_columns[0]])

submission = pd.concat((test_id, submission), axis=1)

submission.head()
submission.to_csv("submission.csv", index=None)