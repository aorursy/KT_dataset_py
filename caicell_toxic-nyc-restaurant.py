# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.

df_train = pd.read_csv('../input/DOHMH_New_York_City_Restaurant_Inspection_Results.csv')
df_train.head()
print('Null Variables: ', df_train.columns[df_train.isnull().any()].tolist())
null_var = ['DBA', 'BUILDING', 'STREET', 'ZIPCODE', 'PHONE', 'ACTION',
       'VIOLATION CODE', 'VIOLATION DESCRIPTION', 'SCORE', 'GRADE',
       'GRADE DATE', 'INSPECTION TYPE']
null_cnt = df_train[null_var].isnull().sum()
null_cnt = null_cnt.to_frame()
null_cnt.columns = ['null_sum']
null_cnt['not_null'] = df_train.shape[0] - null_cnt['null_sum']
null_cnt.plot.bar(stacked = True)
null_test = df_train.columns[df_train.columns.str.contains('LATION')].tolist()
null_test2 = ['SCORE']
df_train[null_test+null_test2].isnull().sum()
local_lst = ['BORO','ZIPCODE']
f = plt.figure(figsize = (12,4))

tmp_null = df_train[null_test].isnull()
tmp_null = (tmp_null.iloc[:,0] | tmp_null.iloc[:,1])

ax = f.add_subplot(1,2,1)
local = local_lst[0]
tmp = df_train.loc[tmp_null, local].value_counts().to_frame()
tmp2 = df_train.loc[:,local].value_counts().to_frame()
null_ratio = (tmp/tmp2.loc[tmp.index,:])
ax.scatter(x = np.arange(null_ratio.shape[0]) ,y = null_ratio.iloc[:,0], alpha = 0.5)
ax.set_title(local)
ax.set_ylabel('Null Probability')
ax.set_xticklabels(['wow'] + tmp.index.tolist())
ax.tick_params('x', rotation = 70)

ax = f.add_subplot(1,2,2)
local = local_lst[1]
tmp = df_train.loc[tmp_null, local].value_counts().to_frame()
tmp2 = df_train.loc[:,local].value_counts().to_frame()
null_ratio = (tmp/tmp2.loc[tmp.index,:])
ax.scatter(x = np.arange(null_ratio.shape[0]) ,y = null_ratio.iloc[:,0], alpha = 0.5)
ax.set_title(local)
ax.set_ylabel('Null Probability')
ax.tick_params('x', rotation = 70)

plt.show()
df_train.groupby('BORO')['BORO'].size()['Missing']
df_train = df_train.loc[df_train.ZIPCODE.isin(null_ratio[null_ratio.sort_values('ZIPCODE') < 0.2].index),:]
df_train = df_train.loc[(df_train.BORO != 'Missing'),:]
df_train.shape
def level_code(row):
    if type(row) == float: return 99, 'No'
    return int(row[:2]), row[2]
df_train['VIO_lvl'],df_train['VIO_type'] = zip(*df_train['VIOLATION CODE'].apply(lambda row: level_code(row)))
#df_train['RECORD_DATE'] = pd.to_datetime(df_train['RECORD DATE'], format = '%m/%d/%Y', errors='coerce')
df_train['INSPECTION_DATE'] = pd.to_datetime(df_train['INSPECTION DATE'], format = '%m/%d/%Y', errors='coerce')
df_train['GRADE_DATE'] = pd.to_datetime(df_train['GRADE DATE'], format = '%m/%d/%Y', errors='coerce')
df_train.drop(['RECORD DATE', 'INSPECTION DATE', 'GRADE DATE'], axis = 1, inplace = True)

df_train.columns = ['_'.join(x.lower().split()) for x in df_train.columns]
tmp_tab = pd.crosstab(df_train['action'], df_train['grade'])
tmp_tab[['A', 'B', 'C', 'P', 'Z', 'Not Yet Graded']]
f = plt.figure(figsize = (12,4))
ax = f.add_subplot(1,2,1)
tmp_tab = pd.crosstab(df_train['critical_flag'], df_train['grade'])
tmp_crit = tmp_tab[['A', 'B', 'C', 'P', 'Z', 'Not Yet Graded']].T
tmp_crit.plot.bar(stacked = True, ax = ax)
ax.set_title('Stacked Critical Flag')
ax.set_xlabel('Grade')

sum_ = tmp_crit.sum(axis = 1)
for col in tmp_crit.columns:
    tmp_crit[col] = tmp_crit[col].divide(sum_)
ax = f.add_subplot(1,2,2)
tmp_crit.plot.bar(stacked = True, ax = ax)
ax.set_title('Stacked Ratio Critical Flag')
ax.set_xlabel('Grade')

plt.show()
f = plt.figure(figsize = (12,8))
ax = plt.subplot2grid((2,4), (0,0), colspan = 3)
sns.distplot(df_train.vio_lvl.loc[df_train.critical_flag == 'Critical'], ax = ax, kde = False, color = 'r', label = 'Critical')
sns.distplot(df_train.vio_lvl.loc[df_train.critical_flag == 'Not Critical'], ax = ax, kde = False, color = 'b', label = 'Not Critical')
ax.set_title('Violation lvl of Critical Lvl')
ax.legend()
ax = plt.subplot2grid((2,4), (0,3))
sns.distplot(df_train.vio_lvl.loc[df_train.critical_flag == 'Not Applicable'], ax = ax, kde = False, color = 'g', label = 'Not Applicable')
ax.set_title('Violation lvl of Not Applicable')
ax.legend()

ax = plt.subplot2grid((2,4), (1,0), colspan = 4)
tmp_type = df_train[['critical_flag', 'vio_type']].groupby(['critical_flag', 'vio_type']).size().to_frame().reset_index()
tmp_type = tmp_type.pivot('critical_flag', 'vio_type').fillna(0)
tmp_type.columns = tmp_type.columns.droplevel(0)
tmp_type = tmp_type.T
sum_type = tmp_type.sum(axis = 1)
for col in tmp_type.columns:
    tmp_type[col] = tmp_type[col].divide(sum_type)
tmp_type.sort_values('Not Critical').plot.bar(stacked = True, ax = ax)
ax.set_title('Vio Type of Critical Flag')

plt.subplots_adjust(wspace = 0.3, hspace = 0.4)
plt.show()
print('OMN')
print(df_train.loc[df_train.vio_type.isin(('O', 'M', 'N')), 'vio_lvl'].value_counts())
print('KL')
print(df_train.loc[df_train.vio_type.isin(('K', 'L')), 'vio_lvl'].value_counts())
tmp_cnt =df_train.groupby(['vio_type']).agg({'vio_lvl': pd.Series.nunique}).sort_values('vio_lvl').T
tmp_cnt[sorted(tmp_cnt.columns, reverse = False)]
