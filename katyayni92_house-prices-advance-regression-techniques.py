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
import numpy as np

import pandas as pd

import os

from sklearn.model_selection import train_test_split
train_df = pd.read_csv("../input/train.csv")

test_df  = pd.read_csv("../input/test.csv")
train_df.head()
test_df.head()
train_df.info()
train_df.describe().transpose()
test_df.info()
test_df.describe().transpose()
table_df = pd.DataFrame({'train_df': [train_df.shape[0], train_df.shape[1]],

                      'test_df': [test_df.shape[0], test_df.shape[1]]}, index = ['rows','columns'])
table_df
df = pd.DataFrame({'A': [1,2,3],

                   'B': [4,5,6]}, index = ['a','b','c'])

df
table