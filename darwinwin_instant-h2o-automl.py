# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.preprocessing import StandardScaler



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

import gc

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')
%%time

def normal(train, test):

    print('Scaling with StandardScaler\n')

    len_train = len(train)



    traintest = pd.concat([train,test], axis=0, ignore_index=True).reset_index(drop=True)

    

    scaler = StandardScaler()

    cols = [c for c in train.columns if c not in ['id', 'target', 'wheezy-copper-turtle-magic']]

    traintest[cols] = scaler.fit_transform(traintest[cols])

    train = traintest[:len_train].reset_index(drop=True)

    test = traintest[len_train:].reset_index(drop=True)



    return train, test



train, test = normal(train, test)
train.to_csv('train.csv', index=False)

test.to_csv('test.csv', index=False)
del train, test

gc.collect()

gc.collect()
import h2o

print(h2o.__version__)

from h2o.automl import H2OAutoML



h2o.init(max_mem_size='16G')
train = h2o.import_file("train.csv")

test = h2o.import_file("test.csv")
test.shape
train.shape

test.head()
cols = [c for c in train.columns if c not in ['id', 'target']]
x = cols

y = 'target'

w = 'wheezy-copper-turtle-magic'
# For binary classification, response should be a factor

train[y] = train[y].asfactor()

train[w] = train[w].asfactor()

test[w] = test[w].asfactor()
aml = H2OAutoML(seed=137, max_runtime_secs=30000)

aml.train(x=x, y=y, training_frame=train)
# View the AutoML Leaderboard

lb = aml.leaderboard

lb.head(rows=lb.nrows)  # Print all rows instead of default (10 rows)
# The leader model is stored here

aml.leader
preds = aml.predict(test)

preds['p1'].as_data_frame().values.flatten().shape
preds

sample_submission = pd.read_csv('../input/sample_submission.csv')

sample_submission.shape
sample_submission['target'] = preds['p1'].as_data_frame().values

sample_submission.to_csv('submission.csv', index=False)