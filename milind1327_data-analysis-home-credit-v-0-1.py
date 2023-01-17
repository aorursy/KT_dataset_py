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
bureau_bal=pd.read_csv("../input/bureau_balance.csv")
bureau_bal
bureau_bal.shape
bureau_bal.columns
bureau_bal['SK_ID_BUREAU'].dtype
df2=bureau_bal[['SK_ID_BUREAU','MONTHS_BALANCE']]
type(df2)
type(bureau_bal['SK_ID_BUREAU'])
bureau_bal.SK_ID_BUREAU.value_counts()
bureau_bal.head()
bureau_bal.tail()
df3=bureau_bal['SK_ID_BUREAU']==5041336
df3
pd.__version__
bureau_bal.loc[5041336]
bureau_bal.iloc[5041336]
bureau_bal.iloc[-1]
bureau_bal.loc[0:5,'SK_ID_BUREAU']
dfg=bureau_bal.groupby('SK_ID_BUREAU')
dfg
type(dfg)
dfg.sum()

