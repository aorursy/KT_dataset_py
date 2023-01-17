# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# DataFrames of the training data and test data

test = pd.read_csv('../input/test.csv')

train = pd.read_csv('../input/train.csv')



train.head()
