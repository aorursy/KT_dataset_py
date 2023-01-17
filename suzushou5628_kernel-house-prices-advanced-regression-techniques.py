# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import pandas as pd

train_data = pd.read_csv('../input/train.csv')

test_data = pd.read_csv('../input/test.csv')

sample_submission_data = pd.read_csv('../input/sample_submission.csv')



for data in (train_data, test_data, sample_submission_data):

    rows_num = data.shape[0]

    columns_num = data.shape[1]

    print("data has %d rows and %d columns" % (rows_num, columns_num))
#train_dataの前処理

X_df = train_data

y_df = train_data['SalePrice']

X_df = X_dh.drop('Id')

print(X_df.head())

print(y_df.head())
dtype_df = X_df.dtypes.reset_index()

dtype_df.columns = ["Count", "Column Type"]

dtype_df.groupby("Column Type").aggregate('count').reset_index()
X2_df = pd.DataFrame(data=None, index=None, columns=None, dtype=None, copy=False)

for column_name in X_df:

    if X_df[column_name].dtype == 'int64':

        print(column_name)

        X2_df[column_name] = X_df[column_name].values
X2_df.head()
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X2_df.values, y_df.values, random_state=42)
from sklearn.ensemble import RandomForestClassifier

rfc = RandomForestClassifier(n_estimators=5, random_state=2)

rfc.fit(X_train, y_train)
print("Accuracy on training set: {:.3f}".format(rfc.score(X_train, y_train)))

print("Accuracy on test set: {:.3f}".format(rfc.score(X_test, y_test)))