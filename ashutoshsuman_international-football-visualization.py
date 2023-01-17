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
results = pd.read_csv('/kaggle/input/international-football-results-from-1872-to-2017/results.csv')
results.head()
results.describe()
results['Goals'] = results['home_score']+results['away_score']
results['Difference'] = abs(results['home_score']-results['away_score'])
results.head()
def winner(x):
    if(x['home_score']>x['away_score']):
        x['winning_team'] = x['home_team']
        x['losing_team'] = x['away_team']
        x['outcome'] = 'H'
    elif(x['away_score']>x['home_score']):
        x['winning_team'] = x['away_team']
        x['losing_team'] = x['home_team']
        x['outcome'] = 'A'
    else:
        x['winning_team'] = None
        x['losing_team'] = None
        x['outcome'] = 'D'
    return x
    

results = results.apply(winner, axis = 1)
results.head()
def give_date(x):
    t = pd.to_datetime(x['date'])
    x['year'] = t.year
    x['month'] = t.month
    x['day_of_week'] = t.dayofweek
    return x

results = results.apply(give_date, axis = 1)
results.head()
decades = 10 * (results['year'] // 10)
decades = decades.astype('str') + 's'
results['decade'] = decades
results.head()
import matplotlib.pyplot as plt
xh = results.groupby('home_team')['home_score'].sum()
xa = results.groupby('away_team')['away_score'].sum()
xh = xh.sort_values(ascending = False)
xa = xa.sort_values(ascending = False)
xh = xh[:10]
xa = xa[:10]
xh.plot(kind = 'barh'); 
xa.plot(kind = 'barh')
x = results.groupby('home_team')[['home_score','away_score']].sum()
x['total'] = x['home_score'] + x['away_score']
x = x.sort_values('total', ascending = False)
x = x[:10]
x.plot(kind = 'barh')
x = results.groupby('decade')['decade'].count()
print(x)
x.plot()
x = results.groupby('decade')['Goals'].sum()
print(x)
x.plot()
x = results.groupby('year')['Goals'].sum()
x.plot()
x = results.groupby('year')['year'].count()
x.plot()
x = results.groupby('month')['month'].count()
x.plot(kind = 'bar')
x = set(results['home_team'])
y = set(results['away_team'])
print("No. of Home Teams",len(x))
print("No. of Away Teams",len(y))
x = set.union(x)
print("Total No. of teams",len(x))
counthome = results.groupby('home_team')['home_team'].count()
countaway = results.groupby('away_team')['away_team'].count()
print(counthome)
print(countaway)
x = counthome + countaway
x = x.sort_values(ascending = False)
x = x[:10]
x.plot(kind = 'barh')
x = results.pivot_table('Goals', index = 'tournament', aggfunc = 'sum')
x = x.sort_values('Goals', ascending  = False)
x = x[:10]
print(x)
x.plot(kind = 'barh')
x = results.pivot_table('Goals', index = 'home_team', columns = 'tournament', aggfunc = 'sum', fill_value = 0, margins = True
                       , margins_name = 'Total')
x = x.sort_values('Total', ascending = False)
x = x[:20]
x['FIFA World Cup'][1:].plot(kind = 'barh', title = 'FIFA World Cup')
x['Friendly'][1:].plot(kind = 'barh', title = 'Friendly')
x['Copa América'][1:].plot(kind = 'barh', title = 'Copa America')
x['UEFA Euro qualification'][1:].plot(kind = 'barh', title = 'UEFA Euro Qualification')
x = results.pivot_table('Goals',index = 'tournament', columns = 'year', aggfunc = 'sum')
x[2011].sort_values(ascending = False)[:20]
x = results.groupby('home_team')['home_score'].mean()
x
