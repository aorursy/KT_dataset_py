# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

# load train data
train = pd.read_csv('../input/dice-dataset/dice_train.csv')

# show columns
print(train.describe());

# show first 10 rows in the dataset
print(train.head(10))

# show two columns
print(train[['isTruthful', 'try10']])
