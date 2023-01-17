import numpy as np
import pandas as pd
%matplotlib inline
df = pd.read_excel('../input/avitoxlsx/avito.xlsx',parse_dates=['date'])
df_all = pd.DataFrame({'date':pd.date_range('2018-5-15','2018-6-11')})
#fill nan
df = df_all.merge(df,on='date',how='left')
def fillna(vals):
    lastval = 0
    lastnum = 0
    nans = []
    for i,num in enumerate(vals):
        if np.isnan(num):
            nans.append(i)
        if not np.isnan(num) and nans:
            for j in nans:
                vals[j]= (num+lastnum)/2
            nans = []
#         print(lastnum,num,not np.isnan(num))
        if not np.isnan(num):
            lastnum=num
    return vals
df['teams'] = fillna(df.teams.values)
df['competitors'] = fillna(df.competitors.values)
df
df.plot.line(x='date',y='competitors')
df.plot.line(x='date',y='teams')
from  sklearn.linear_model import *
lr = Ridge(alpha=10)
lr.fit(df.date.dt.dayofyear.values.reshape(-1,1),df.competitors)
test_df = pd.DataFrame({'date':pd.date_range(start='2018-06-11',end='2018-06-21'),'teams':np.NAN,'competitors':np.NAN})
test_df
test_df['competitors'] = lr.predict(test_df.date.dt.dayofyear.values.reshape(-1,1))
test_df.plot.line(x='date',y='competitors')
lr.fit(df.date.dt.dayofyear.values.reshape(-1,1),df.teams)
test_df['teams'] = lr.predict(test_df.date.dt.dayofyear.values.reshape(-1,1))

test_df.plot.line(x='date',y='teams')
print('teams:',test_df.teams.iloc[-1],'competitimes:',test_df.competitors.iloc[-1])
test_df
