# This Python 3 environment is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

pd.set_option('display.max_rows', 500) # Show longer results

import matplotlib.pyplot as plt

import seaborn as sns

import xgboost as xgb



# Input data files are available in the "../input/" directory.

# List all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
# Read test and train datasets to dataframes

train1 = pd.read_csv('/kaggle/input/equipfailstest/equip_failures_training_set.csv', na_values='na')

test1 = pd.read_csv('/kaggle/input/equipfailstest/equip_failures_test_set.csv', na_values='na')

# View first few rows

train1.head()
# Verify all columns are numbers

train1.info()
# Fill NaN with median values

train2 = train1

test2 = test1

for col in train1.columns:

    if col != ('id' or 'target'):

        train2[col] = train1[col].fillna(train1[col].median())

        test2[col] = train1[col].fillna(train1[col].median())
# Plot violin plots by target for each column

for col in train1.columns:

    if col != 'id' and col != 'target':

        try:

            sns.violinplot(train1.loc[train2[col].notnull(), 'target'], train2.loc[train2[col].notnull(), col])

            plt.show()

        except:

            print(train2[col].nunique())
# After reviewing violin plots select columns for training where violin plots differ substantially by target

train_cols = ['sensor1_measure',

              'sensor8_measure',

              'sensor14_measure',

              'sensor15_measure',

              'sensor16_measure',

              'sensor17_measure',

              'sensor27_measure',

              'sensor32_measure',

              'sensor33_measure',

              'sensor34_measure',

              'sensor35_measure',

              'sensor36_measure',

              'sensor37_measure',

              'sensor38_measure',

              'sensor44_measure',

              'sensor45_measure',

              'sensor46_measure',

              'sensor47_measure',

              'sensor48_measure',

              'sensor49_measure',

              'sensor53_measure',

              'sensor59_measure',

              'sensor61_measure',

              'sensor67_measure',

              'sensor72_measure',

              'sensor78_measure',

              'sensor89_measure',

              'sensor90_measure',

              'sensor91_measure',

              'sensor94_measure',

              'sensor95_measure']

train3 = train2[train_cols]

test3 = test2[train_cols]
# Train with XGBoost

dtrain = xgb.DMatrix(data=train3, label=train1.target)

param = {'max_depth':6, 'eta':0.3, 'objective':'binary:logistic' }

num_round = 10

bst = xgb.train(param, dtrain, num_round)

print("Training complete")
# Save model

bst.save_model('EquipFail_10-13-19.model')
# Predict!!!

dtest = xgb.DMatrix(data=test3)

pred1 = bst.predict(dtest)

pred2 = pred1.round()
# Create output file and save

sub1 = test1[['id']]

pred3 = pd.DataFrame(pred2, columns=['target'])

pred4 = pred3.astype('int')

sub2 = pd.concat([sub1, pred4], axis=1)

sub2.sample(10)

sub2.to_csv('EquipFail_10-13-19.csv', index=False)