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
import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

import geopandas as gpd

import numpy as np

from pathlib import Path

import plotly.offline as py

import plotly.express as px

import cufflinks as cf

from sklearn.tree import DecisionTreeRegressor

from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_absolute_error as MAE

from sklearn.cluster import DBSCAN
USA = pd.read_csv('/kaggle/input/covid19-in-usa/us_states_covid19_daily.csv')

WetterNY = pd.read_csv('/kaggle/input/newyorkall2/NewYork_all2.csv',sep = ";")
USA
USA.info()
WetterNY
WetterNY.info()
WetterNY.rename(columns={"date": "Date"})

WetterNY
USA['state'].unique()
NYC = USA[USA.state=='NY']

NYC
NYC.info()
#NYC = NYC[NYC['positive']>0]

#NYC

#New York nur die positiven Fälle
#coronawetterny = pd.merge(NYC, WetterNY, how='right', on='date')

coronawetterny = pd.merge(WetterNY, NYC, how='inner', on='date')

coronawetterny
coronawetterny.set_index('date')
coronawetterny2 = coronawetterny.copy()

coronawetterny2=coronawetterny2.drop(columns=['pending','hospitalizedCurrently', 'negative', 'hospitalizedCumulative', 'inIcuCurrently','inIcuCumulative', 'onVentilatorCurrently', 'onVentilatorCumulative'])

coronawetterny3 = coronawetterny2.copy()

coronawetterny3=coronawetterny3.drop(columns=['lastUpdateEt', 'hash', 'recovered', 'dataQualityGrade', 'dateChecked', 'death', 'hospitalized', 'totalTestResults', 'total', 'posNeg', 'fips', 'deathIncrease'])



coronawetterny4 = coronawetterny3.copy()

coronawetterny4=coronawetterny4.drop(columns=['hospitalizedIncrease', 'negativeIncrease', 'totalTestResultsIncrease'])

coronawetterny4
sortedcoronawetterny = coronawetterny4.sort_values(by='date')

sortedcoronawetterny
sortedcoronawetterny.info()
sortedcoronawetterny['date'] = sortedcoronawetterny['date'].astype(str)
sortedcoronawetterny.set_index('date', drop=True,inplace=True)

sortedcoronawetterny
sortedcoronawetterny['positiveIncrease'].plot()

sortedcoronawetterny['Avg Temperature in C'].plot()

plt.xlabel('Achsenbeschriftung')

sortedcoronawetterny['Avg Temperature in C'].plot(kind='line',x='date',y='Avg Temperature in C')
sortedcoronawetterny.corr(method ='kendall')

#korrelation zwischen -1 und 1

#  1 -> korreliert

#  0 -> kein Zusammenhang

# -1 -> korreliert entgegengesetzt
sortedcoronawetterny['Precipitation'].plot(kind='line')

plt.ylabel('Niederschlag')

plt.xlabel('Datum')

plt.title('Niederschlagswerte')
#ging nur solange Date noch kein String war



#NYpositiveincrease = sortedcoronawetterny[['date', 'positiveIncrease']].copy()

#NYpositiveincrease = NYpositiveincrease.sort_values(by=['positiveIncrease'])

#NYpositiveincrease['date'] = pd.date_range(start='20200304', periods=58, freq='D')

#start_date = NYpositiveincrease.date.min()

#end_date = NYpositiveincrease.date.max()

#print('start date: {}\nend date: {}\n'.format(start_date, end_date))
#ging nur solange Date noch kein String war



#NYpositiveincrease.iplot(x='date', y='positiveIncrease', kind='bar',xTitle='date', yTitle='positiveIncrease',

               # title='New York: Zunahme der positiven Fälle')
#sortedcoronawetterny['new']=sortedcoronawetterny['positiveIncrease'].rolling(10).mean()

#sortedcoronawetterny.info
#sortedcoronawetterny
#sortedcoronawetterny.expanding(10).sum()

#sortedcoronawetterny
for i in range(14):

    i=i-14

    sortedcoronawetterny.loc[:,('positiveIncrease ' + str(i) + ' days after weather')] = sortedcoronawetterny.positiveIncrease.shift(i)
sortedcoronawetterny
korrmat=sortedcoronawetterny.corr(method ='kendall')
korrmat
korrmatcol=korrmat.copy()

korrmatcol=korrmatcol.drop(columns=['positiveIncrease -1 days after weather','positiveIncrease -2 days after weather', 'positiveIncrease -3 days after weather', 'positiveIncrease -5 days after weather', 'positiveIncrease -4 days after weather', 'positiveIncrease -6 days after weather', 'positiveIncrease -7 days after weather', 'positiveIncrease -8 days after weather', 'positiveIncrease -9 days after weather','positiveIncrease -10 days after weather','positiveIncrease -11 days after weather','positiveIncrease -12 days after weather','positiveIncrease -13 days after weather','positiveIncrease -14 days after weather', 'positive', 'positiveIncrease'])

korrmatcol.drop(['Avg Temperature in C', 'Avg Dew Point in C', 'Avg Humidity in Percent', 'Avg Wind Speed in mph', 'Avg Pressure in Hg', 'Precipitation' ])
wetterkorr=WetterNY.corr(method ='kendall')

wetterkorr['Avg Pressure in Hg']