# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import seaborn as sns

import re



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
# define constants



TRAIN_PATH = "../input/reviews_train.csv"

TEST_PATH = "../input/reviews_test.csv"



NUM_FEATURES = 1000

PADDED_LENGTH = 250

# load data



# column names: positive, review_title, review_body

train_df = pd.read_csv(TRAIN_PATH, sep=',', header=0)



train_df.head(10)
# load test data



# column names: id, review_title, review_body

test_df = pd.read_csv(TEST_PATH)
# Solution goes here
# save test results



results_df = pd.DataFrame()

results_df['id'] = test_df['id']

results_df['positive'] = predictions



results_df.head()
results_df.to_csv('results.csv', ',', header=True, index=False)