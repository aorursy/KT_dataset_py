# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df = pd.read_csv('/kaggle/input/ieee-fraud-detection/train_transaction.csv')
df.head(20)
df.describe()
df_nan  = df.isna().sum()

df_nan = df_nan.loc['V1':'V339']
pd_nan = pd.DataFrame(df_nan)

pd_nan.reset_index(inplace=True)

pd_nan.columns= ['Feature','count']

sns.barplot(x='Feature',y='count',data=pd_nan)