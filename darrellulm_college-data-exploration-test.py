# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.model_selection import train_test_split

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

df = pd.read_csv('../input/MERGED2013_PP.csv')
train_df, test_df = train_test_split(df, test_size=0.2)

combine = [train_df, test_df]

print(train_df.columns.values)

train_df.head()

train_df.tail()

train_df.info()

train_df.describe()

# Any results you write to the current directory are saved as output.
