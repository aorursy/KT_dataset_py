# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
from kaggle.competitions import nflrush

import pandas as pd



# You can only call make_env() once, so don't lose it!

env = nflrush.make_env()
train_df = pd.read_csv('/kaggle/input/nfl-big-data-bowl-2020/train.csv', low_memory=False)

train_df.head()
train_df.shape
train_df.columns
train_df.info()
print("Count NA df_train per column : \n" + str(train_df.isna().sum()))
print('Total Games Played: ', len(train_df.GameId.value_counts()))

print('Total Stadium Games: ', len(train_df.Stadium.value_counts()))