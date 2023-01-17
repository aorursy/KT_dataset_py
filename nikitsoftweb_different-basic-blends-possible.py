# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from functools import reduce
import seaborn as sns
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
M1 = pd.read_csv('../input/gogomaster/Think1.csv')
M2 = pd.read_csv('../input/gogomaster/11111.csv')
M3 = pd.read_csv('../input/gogomaster/22222.csv')
M4 = pd.read_csv('../input/gogomaster/33333.csv')
M5 = pd.read_csv('../input/gogomaster/44444.csv')
M6 = pd.read_csv('../input/gogomaster/55555.csv')
# Function for merging dataframes efficiently 
def merge_dataframes(dfs, merge_keys):
    dfs_merged = reduce(lambda left,right: pd.merge(left, right, on=merge_keys), dfs)
    return dfs_merged
dfs = [M1,M2,M3,M4,M5,M6]
merge_keys=['SK_ID_CURR']
df = merge_dataframes(dfs, merge_keys=merge_keys)
df.columns = ['SK_ID_CURR','T1','T2','T3','T4','T5','T6']
df.head()
pred_prob = 0.7 * df['T3'] + 0.3 * df['T1']
pred_prob.head()
sub = pd.DataFrame()
sub['SK_ID_CURR'] = df['SK_ID_CURR']
sub['target']= pred_prob
sub.to_csv('ldit.csv', index=False)
B_prob = 0.6 * df['T3'] + 0.2 * df['T1'] + 0.2 * df['T2']
B_prob.head()
SUB = pd.DataFrame()
SUB['SK_ID_CURR'] = df['SK_ID_CURR']
SUB['TARGET'] = B_prob
SUB.to_csv('Blendss.csv', index=False)
df_c = df.copy()
df_c = df.drop(['SK_ID_CURR'],axis=1)
Corr_Mat = df_c.corr()
print(Corr_Mat) # Correlation matrix of five submission files
sns.heatmap(Corr_Mat)
corr_pred = 0.6 * df['T2'] + 0.05 * df['T3'] + 0.05 * df['T4'] + 0.1 * df['T5'] + 0.2 * df['T1']
corr_pred.head()
SuB = pd.DataFrame()
SuB['SK_ID_CURR'] = df['SK_ID_CURR']
SuB['TARGET'] = corr_pred
SuB.to_csv('corr_blend.csv', index=False)
