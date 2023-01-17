# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.preprocessing import LabelBinarizer



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
# Read the data

train = pd.read_csv("../input/train.csv")

test = pd.read_csv("../input/test.csv")
def binarize(train_data, test_data, columns):

    train_data_numeric = train_data.drop(columns, axis=1)

    test_data_numeric = test_data.drop(columns, axis=1)

    for col in columns:

        l = LabelBinarizer()

        train_col_xformed = pd.DataFrame(l.fit_transform(train_data[col]))

        test_col_xformed = pd.DataFrame(l.transform(test_data[col]))

        xformed_col_names = [col + '_' + str(i) for i in range(len(train_col_xformed.columns))]

        train_col_xformed.columns = test_col_xformed.columns = xformed_col_names

        train_data_numeric[xformed_col_names] = train_col_xformed

        test_data_numeric[xformed_col_names] = test_col_xformed

    return(train_data_numeric, test_data_numeric)
cols_to_convert = ['X' + str(i) for i in range(7)] + ['X8']



_train, _test = binarize(train, test, cols_to_convert)



## Assert that we have the same columns for train and test data

print(set(_train.columns) - set(_test.columns))     # Should return {'y'}

print(set(_test.columns) - set(_train.columns))     # Should return an empty set