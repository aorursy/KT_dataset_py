# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.ensemble import RandomForestClassifier

from matplotlib import pyplot as plt

import seaborn as sbn



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.

from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble import AdaBoostClassifier

from sklearn.ensemble import BaggingClassifier

from sklearn.ensemble import GradientBoostingClassifier

from sklearn.ensemble import VotingClassifier

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score

from sklearn.metrics import f1_score

from sklearn.metrics import precision_score

from sklearn.metrics import recall_score
test_data = pd.read_csv('../input/test.csv')

train_data = pd.read_csv('../input/train.csv')



train_data = train_data.drop(["PassengerId","Name","Ticket"], axis=1)

test_data = test_data.drop(["Name","Ticket"], axis=1)



train_data.head(10)
def convert_column_to_numeric(dataset, column, fillval="--None--"):

    dataset[column] = dataset[column].fillna(fillval)

    col_uniqs = dataset[column].unique().tolist()

    col_uniqs.sort()

    num_uniqs = len(col_uniqs)

    mapping = dict(zip(col_uniqs, range(num_uniqs)))

    dataset[column] = dataset[column].map(mapping)

    revmapping = dict(zip(range(num_uniqs), col_uniqs))

    return revmapping



def convert_dataframe_to_numeric(dataset, donotprocess=[]):

    dcopy = dataset.copy()

    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']

    nonnum_cols = dcopy.select_dtypes(exclude=numerics).columns.values

    nonnum_cols = [column for column in nonnum_cols

                          if column not in donotprocess]

    print(nonnum_cols)

    mappings = {}

    for column in nonnum_cols:

        mappings[column] = convert_column_to_numeric(dcopy, column)

    return dcopy, mappings



train_n, mappings = convert_dataframe_to_numeric(train_data)

test_n, mappings = convert_dataframe_to_numeric(test_data)



train_n = train_n.dropna()

train_labels = train_n["Survived"]

train_n = train_n.drop(["Survived"], axis=1)



train_n.head(10)
train_labels.head(10)
booster = AdaBoostClassifier()

forest = RandomForestClassifier()

bagger = BaggingClassifier()

xgbooster = GradientBoostingClassifier()

voter = VotingClassifier(estimators=[('ab',booster),('rf',forest),

                                     ('b',bagger),('xg',xgbooster)],

                         voting='soft')



for i in range(40):

    train_X, val_X, train_y, val_y = train_test_split(train_n, train_labels, test_size=0.3)

    booster.fit(train_X, train_y)

    forest.fit(train_X, train_y)

    bagger.fit(train_X, train_y)

    xgbooster.fit(train_X, train_y)

    voter.fit(train_X, train_y)



booster_out_val = booster.predict(val_X)

forest_out_val = forest.predict(val_X)

bagger_out_val = bagger.predict(val_X)

xgbooster_out_val = xgbooster.predict(val_X)

voter_out_val = voter.predict(val_X)



print(accuracy_score(val_y, booster_out_val))

print()

print(accuracy_score(val_y, forest_out_val))

print()

print(accuracy_score(val_y, bagger_out_val))

print()

print(accuracy_score(val_y, xgbooster_out_val))

print()

print(accuracy_score(val_y, voter_out_val))