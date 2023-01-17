import numpy as np

import pandas as pd

df1= pd.read_csv('../input/factors-affecting-campus-placement/Placement_Data_Full_Class.csv')

df1
df2= pd.read_csv('../input/bank-marketing-dataset/bank.csv')

df2
df1['salary'].hist(bins=30) 
df1['salary'].plot.hist()
df1['salary'].plot(kind='hist')
df_sub=df1[['ssc_p','hsc_p','degree_t','etest_p','mba_p']].copy()

df_sub.plot.box()
df_sub=df1[['ssc_p','hsc_p','degree_t','etest_p','mba_p']].copy()

df_sub.plot.area()
df1.plot.line(x='degree_t',y='degree_p')
df1.plot.scatter(x='mba_p',y='salary',s=df1['hsc_p']*3)
df1.plot.hexbin(x='ssc_p',y='etest_p',gridsize=15)
df2['duration'].plot.kde()