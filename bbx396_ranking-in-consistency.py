# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import math
import matplotlib
import matplotlib.pyplot as plt

matplotlib.style.use('ggplot')

%matplotlib inline

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
# print(check_output(["ls", "../input"]).decode("utf8"))

# read in data
df_cwur = pd.read_csv('../input/cwurData.csv')
df_expend = pd.read_csv('../input/education_expenditure_supplementary_data.csv', sep=None)
df_attain = pd.read_csv('../input/educational_attainment_supplementary_data.csv', sep=None)
df_sac = pd.read_csv('../input/school_and_country_table.csv')
df_shanghai = pd.read_csv('../input/shanghaiData.csv')
df_times = pd.read_csv('../input/timesData.csv')

# adding country data to the shanghai dataset
df_sac.columns = ['university_name','country']
df_shanghai = df_shanghai.merge(df_sac, how='left', on='university_name')

# updating column name in cwur for consistency
df_cwur = df_cwur.rename(columns = {'institution':'university_name'})

# updating country names in cwur for consistency
df_cwur.drop('country', inplace=True, axis=1)
df_cwur = df_cwur.merge(df_sac, how='left', on='university_name')

# combine the 3 ranking dataframes into one dataframe
for df, l in [(df_times,'t'), (df_cwur,'c'), (df_shanghai,'s')]:
    a = []
    for col in df.columns.values:
        if col not in ['university_name','year']:
            a.append(l + '_' + col)
        else:
            a.append(col)
    df.columns = a

df_full = df_times.merge(df_cwur, how='outer', on=['university_name','year'])
df_full = df_full.merge(df_shanghai, how='outer', on=['university_name','year'])
# creating a dataframe that specifically looks at the rankings
df_ranks = df_full[['university_name','t_country','year','t_world_rank','c_world_rank',
                    's_world_rank']].copy()

# convert world rank columns to float (where necessary)
f = lambda x: int((int(x.split('-')[0]) + int(x.split('-')[1])) / 2) if len(str(x).strip()) > 3 else x

df_ranks['t_world_rank'] = df_ranks['t_world_rank'].str.replace('=','').map(
    f).astype('float')

df_ranks['s_world_rank'] = df_ranks['s_world_rank'].str.replace('=','').map(
    f).astype('float')

df_ranks.dtypes
# note that data is available in all datasets in 2012 - 2015
# let's investigate 2015
df_ranks2015 = df_ranks[df_ranks.year == 2015].copy()

# adding min_rank column to show the best rank for each school
def f(x):
    a = []
    for i in ['t_world_rank','s_world_rank','c_world_rank']:
        try: 
            if x[i] == float(x[i]):
                a.append(x[i])
        except:
            pass
    return min(a)

df_ranks2015['min_rank'] = df_ranks2015.apply(f,axis=1)

# adding average rank column
df_ranks2015['mean_rank'] = df_ranks2015.apply(lambda x: np.mean(
    [x['s_world_rank'],x['t_world_rank'],x['c_world_rank']]).round(), axis=1)

# adding standard deviation column
df_ranks2015['std_dev'] = df_ranks2015.apply(lambda x: np.std(
        [x['s_world_rank'],x['t_world_rank'],x['c_world_rank']]), axis=1)

# plot highest variance in schools that have at least one rank in the top 100
df_ranks2015[df_ranks2015['min_rank'] <=100].sort_values('std_dev',ascending=False)[0:10].iloc[::-1].plot(
    x='university_name',y='std_dev',kind='barh', figsize=(12,6), fontsize=14,
    title='Top 100 schools with the highest standard deviation in ranking')

print(df_ranks2015[df_ranks2015['min_rank'] <=100].sort_values('std_dev',ascending=False).drop(
        't_country',axis=1)[0:10])
# plot lowest variance in schools that have at least one rank in the top 100
df_ranks2015[df_ranks2015['min_rank'] <=100].sort_values('std_dev',ascending=True)[0:10].iloc[::-1].plot(
    x='university_name',y='std_dev',kind='barh', figsize=(12,6), fontsize=14,
    title='Top 100 schools with the lowest standard deviation in ranking')

print(df_ranks2015[df_ranks2015['min_rank'] <=100].sort_values('std_dev',ascending=True).drop(
        't_country',axis=1)[0:10])
# how do ratings change over time?
# note rankings are only available for all 3 in years 2012-2015
df_rankstime = df_ranks[(df_ranks['year'] <=2015) & (df_ranks['year'] >= 2012)]

# times rankings
df_tranks = df_rankstime.pivot('university_name','year','t_world_rank').reset_index()
df_tranks.columns = ['university_name','2012','2013','2014','2015']
df_tranks['std_dev'] = df_tranks.apply(lambda x: np.std(
            [x['2012'],x['2013'],x['2014'],x['2015']]),axis=1)
df_tranks['mean'] = df_tranks.apply(lambda x: np.mean(
            [x['2012'],x['2013'],x['2014'],x['2015']]),axis=1)

# cwur rankings
df_cranks = df_rankstime.pivot('university_name','year','c_world_rank').reset_index()
df_cranks.columns = ['university_name','2012','2013','2014','2015']
df_cranks['std_dev'] = df_cranks.apply(lambda x: np.std(
            [x['2012'],x['2013'],x['2014'],x['2015']]),axis=1)
df_cranks['mean'] = df_cranks.apply(lambda x: np.mean(
            [x['2012'],x['2013'],x['2014'],x['2015']]),axis=1)

# shanghai rankings
df_sranks = df_rankstime.pivot('university_name','year','s_world_rank').reset_index()
df_sranks.columns = ['university_name','2012','2013','2014','2015']
df_sranks['std_dev'] = df_sranks.apply(lambda x: np.std(
            [x['2012'],x['2013'],x['2014'],x['2015']]),axis=1)
df_sranks['mean'] = df_sranks.apply(lambda x: np.mean(
            [x['2012'],x['2013'],x['2014'],x['2015']]),axis=1)
df_tranks[df_tranks['mean'] <= 100].sort_values('std_dev',ascending=False)[0:10].iloc[::-1].plot(
    x='university_name',y='std_dev',kind='barh', figsize=(12,6), fontsize=14,
    title='Top 100 schools with the highest standard deviation in times ranking (2012-2015)')
df_cranks[df_cranks['mean'] <= 100].sort_values('std_dev',ascending=False)[0:10].iloc[::-1].plot(
    x='university_name',y='std_dev',kind='barh', figsize=(12,6), fontsize=14,
    title='Top 100 schools with the highest standard deviation in CWUR ranking (2012-2015)')
df_sranks[df_sranks['mean'] <= 100].sort_values('std_dev',ascending=False)[0:10].iloc[::-1].plot(
    x='university_name',y='std_dev',kind='barh', figsize=(12,6), fontsize=14,
    title='Top 100 schools with the highest standard deviation in Shanghai ranking (2012-2015)')
# which ranking was the most consistent for the top 100 schools?
print('Which ranking was the most consistent from 2012-2015 for the top 100 schools?')
print('\n')
print('Below is the standard deviation for the top 100 schools from each ranking','\n')
print('Times:', round(df_tranks[df_tranks['mean'] <= 100]['std_dev'].mean(),2))
print('CWUR:', round(df_cranks[df_cranks['mean'] <= 100]['std_dev'].mean(),2))
print('Shanghai:', round(df_sranks[df_sranks['mean'] <= 100]['std_dev'].mean(),2))
print('\n')
print('The Shanghai ranking was substantially more consistent than the other two rankings. This suggests',
      'that the Shanghai ranking is less prone to shaking up rankings just for attention.')
