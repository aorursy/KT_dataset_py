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
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt 

import seaborn as sns

import os

from kaggle.competitions import nflrush

import matplotlib.patches as patches
print('Total File sizes')

print('-'*10)

for f in os.listdir('../input/nfl-big-data-bowl-2020'):

    if 'zip' not in f:

        print(f.ljust(30) + str(round(os.path.getsize('../input/nfl-big-data-bowl-2020/' + f) / 1000000, 2)) + 'MB')
df=pd.read_csv('../input/nfl-big-data-bowl-2020/train.csv',low_memory=False)

df.shape
df.head()
df.isnull().sum()
df.isnull().mean()
df.columns[df.isnull().any()]
env=nflrush.make_env()

iter_test=env.iter_test()

(test_df,sample_prediction_df)=next(iter_test)

test_df.head()
sample_prediction_df.head()
df.columns
df.info()
missing=df.isnull().sum()

missing
missing=missing[missing>0]
missing
missing.sort_values(inplace=True)

missing
print('Total of Games played:',df.GameId.nunique())
plt.figure(figsize=(7,7))

sns.countplot(x=df.Team,palette='Set2')

plt.figure(figsize=(7,7))

sns.distplot(df['Distance'],bins=25,kde=False).set_title("Yards needed for a first down")