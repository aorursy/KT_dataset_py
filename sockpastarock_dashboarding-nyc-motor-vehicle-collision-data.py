# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np
import pandas as pd
from datetime import datetime
import os

PATH_INPUT_DATA = '../input/nypd-motor-vehicle-collisions.csv'
DATE_FORMATE = '%Y-%m-%d'
src = pd.read_csv(PATH_INPUT_DATA)
df = src
df['DATE'] = pd.to_datetime(df['DATE'])
df.sort_values(by=['DATE'], inplace=True, ascending=True)
df = df.reset_index(drop=True)
def month_year_iter(start_month, start_year, end_month, end_year):
    ym_start= 12*start_year + start_month - 1
    ym_end= 12*end_year + end_month - 1
    for ym in range( ym_start, ym_end ):
        y, m = divmod( ym, 12 )
        start_date = datetime.strptime(
            str(y) + '-' + str(m+1) + '-01', DATE_FORMATE)
        if m + 2 < 13:
            end_date = datetime.strptime(
                str(y) + '-' + str(m+2) + '-01', DATE_FORMATE)
        else:
            end_date = datetime.strptime(
                str(y+1) + '-' + str(1) + '-01', DATE_FORMATE)
        yield start_date, end_date
data = {}

start_month = df.iloc[0]['DATE'].month
start_year = df.iloc[0]['DATE'].year
end_month = df.iloc[-1]['DATE'].month
end_year = df.iloc[-1]['DATE'].year

for start_date, end_date in month_year_iter(start_month, start_year,
                                            end_month, end_year):
    
    mask = (df['DATE'] > start_date) & (df['DATE'] <= end_date)
    tmp = df.loc[mask]
    data[start_date] = {
        'NUMBER OF PERSONS INJURED': tmp['NUMBER OF PERSONS INJURED'].sum(),
        'NUMBER OF PERSONS KILLED': tmp['NUMBER OF PERSONS KILLED'].sum(),
        'NUMBER OF COLLISIONS': len(tmp)
    }

tmp = pd.DataFrame(data).transpose()

tmp0 = tmp[['NUMBER OF COLLISIONS']]
ax0 = tmp0.plot(kind='bar', figsize=(30,10))
ax0.set_xlabel('Month')
ax0.set_ylabel('Count')
ax0.set_title('Motor Vehicle Collisions in NYC')

tmp1 = tmp[['NUMBER OF PERSONS INJURED']]
ax1 = tmp1.plot(kind='bar', figsize=(30,10))
ax1.set_xlabel('Month')
ax1.set_ylabel('Count')
ax1.set_title('Persons Injured by Motor Vehicle Collisions in NYC')

tmp2 = tmp[['NUMBER OF PERSONS KILLED']]
ax2 = tmp2.plot(kind='bar', figsize=(30,10))
ax2.set_xlabel('Month')
ax2.set_ylabel('Count')
ax2.set_title('Persons Killed by Motor Vehicle Collisions in NYC')
