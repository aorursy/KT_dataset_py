# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df = pd.read_csv('../input/default-of-credit-card-clients-dataset/UCI_Credit_Card.csv')
df
len(df)
df.loc[(df['PAY_0'] == -1) & (df['PAY_2'] == -1) & (df['PAY_3'] == -1) & (df['PAY_4'] == -1) & (df['PAY_5'] == -1) & (df['PAY_6'] == -1),'target1'] = 0
df.loc[(df['PAY_0'] >= 1)  | (df['PAY_2'] >= 1) | (df['PAY_3'] >= 1) | (df['PAY_4'] >= 1) | (df['PAY_5'] >= 1) | (df['PAY_6'] >= 1),'target1'] = 1

df.loc[((df['PAY_0'] == -1) | (df['PAY_0'] == 1)) & ((df['PAY_2'] == -1) | (df['PAY_2'] == 1)) & ((df['PAY_3'] == -1) | (df['PAY_3'] == 1)) & ((df['PAY_4'] == -1) | (df['PAY_4'] == 1)) & ((df['PAY_5'] == -1) | (df['PAY_5'] == 1)) & ((df['PAY_6'] == -1) | (df['PAY_6'] == 1)),'target2'] = 0
df.loc[(df['PAY_0'] > 1)  | (df['PAY_2'] > 1) | (df['PAY_3'] > 1) | (df['PAY_4'] > 1) | (df['PAY_5'] > 1) | (df['PAY_6'] > 1),'target2'] = 1
df.loc[((df['PAY_0'] == -1) | (df['PAY_0'] == 1) | (df['PAY_0'] == 2)) & ((df['PAY_2'] == -1) | (df['PAY_2'] == 1) | (df['PAY_2'] == 2)) & ((df['PAY_3'] == -1) | (df['PAY_3'] == 1) | (df['PAY_3'] == 2)) & ((df['PAY_4'] == -1) | (df['PAY_4'] == 1) | (df['PAY_4'] == 2)) & ((df['PAY_5'] == -1) | (df['PAY_5'] == 1) | (df['PAY_5'] == 2)) & ((df['PAY_6'] == -1) | (df['PAY_6'] == 1) | (df['PAY_6'] == 2)),'target3'] = 0
df.loc[(df['PAY_0'] > 2)  | (df['PAY_2'] > 2) | (df['PAY_3'] > 2) | (df['PAY_4'] > 2) | (df['PAY_5'] > 2) | (df['PAY_6'] > 2),'target3'] = 1
print(df['target1'].value_counts())
print(df['target2'].value_counts())
print(df['target3'].value_counts())
df.loc[(df['PAY_2'] == -1) & (df['PAY_3'] == -1) & (df['PAY_4'] == -1) & (df['PAY_5'] == -1) & (df['PAY_6'] == -1),'target4'] = 0
df.loc[(df['PAY_2'] >= 1) | (df['PAY_3'] >= 1) | (df['PAY_4'] >= 1) | (df['PAY_5'] >= 1) | (df['PAY_6'] >= 1),'target4'] = 1
df.loc[((df['PAY_2'] == -1) | (df['PAY_2'] == 1)) & ((df['PAY_3'] == -1) | (df['PAY_3'] == 1)) & ((df['PAY_4'] == -1) | (df['PAY_4'] == 1)) & ((df['PAY_5'] == -1) | (df['PAY_5'] == 1)) & ((df['PAY_6'] == -1) | (df['PAY_6'] == 1)),'target5'] = 0
df.loc[(df['PAY_2'] > 1) | (df['PAY_3'] > 1) | (df['PAY_4'] > 1) | (df['PAY_5'] > 1) | (df['PAY_6'] > 1),'target5'] = 1
df.loc[((df['PAY_2'] == -1) | (df['PAY_2'] == 1) | (df['PAY_2'] == 2)) & ((df['PAY_3'] == -1) | (df['PAY_3'] == 1) | (df['PAY_3'] == 2)) & ((df['PAY_4'] == -1) | (df['PAY_4'] == 1) | (df['PAY_4'] == 2)) & ((df['PAY_5'] == -1) | (df['PAY_5'] == 1) | (df['PAY_5'] == 2)) & ((df['PAY_6'] == -1) | (df['PAY_6'] == 1) | (df['PAY_6'] == 2)),'target6'] = 0
df.loc[(df['PAY_2'] > 2) | (df['PAY_3'] > 2) | (df['PAY_4'] > 2) | (df['PAY_5'] > 2) | (df['PAY_6'] > 2),'target6'] = 1
print(df['target4'].value_counts())
print(df['target5'].value_counts())
print(df['target6'].value_counts())
df.loc[(df['PAY_0'] < 1) & (df['PAY_2'] < 1) & (df['PAY_3'] < 1) & (df['PAY_4'] < 1) & (df['PAY_5'] < 1) & (df['PAY_6'] < 1),'target7'] = 0
df.loc[(df['PAY_0'] >= 1)  | (df['PAY_2'] >= 1) | (df['PAY_3'] >= 1) | (df['PAY_4'] >= 1) | (df['PAY_5'] >= 1) | (df['PAY_6'] >= 1),'target7'] = 1

df.loc[(df['PAY_0'] < 2) & (df['PAY_2'] < 2) & (df['PAY_3'] < 2) & (df['PAY_4'] < 2) & (df['PAY_5'] < 2) & (df['PAY_6'] < 2),'target8'] = 0
df.loc[(df['PAY_0'] >= 2)  | (df['PAY_2'] >= 2) | (df['PAY_3'] >= 2) | (df['PAY_4'] >= 2) | (df['PAY_5'] >= 2) | (df['PAY_6'] >= 2),'target8'] =1 
df.loc[(df['PAY_0'] < 3) & (df['PAY_2'] < 3) & (df['PAY_3'] < 3) & (df['PAY_4'] < 3) & (df['PAY_5'] < 3) & (df['PAY_6'] < 3),'target9'] = 0
df.loc[(df['PAY_0'] >= 3)  | (df['PAY_2'] >= 3) | (df['PAY_3'] >= 3) | (df['PAY_4'] >= 3) | (df['PAY_5'] >= 3) | (df['PAY_6'] >= 3),'target9'] = 1
print(df['target7'].value_counts())
print(df['target8'].value_counts())
print(df['target9'].value_counts())