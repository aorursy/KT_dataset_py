# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
arrestsTotal = pd.read_csv('../input/nfl-arrests/arrests.csv')
notes = pd.read_csv('../input/nfl-arrests/notes.csv')
arrestsTotal.head()
arrestsTotal.tail()
arrestsTotal.sample(10)
arrestsTotal.shape
print(arrestsTotal.dtypes)
arrestsTotal.info()
notes.head()
arrestsTotal.describe()
week = arrestsTotal['week_num']
week.hist()
arrestNo=arrestsTotal['arrests']
arrestNo.hist()
arrestsTotal[arrestsTotal['arrests'].gt(8)]
arrestsTotal[arrestsTotal['arrests'].gt(50)]
over30 = arrestsTotal[arrestsTotal['arrests'].gt(30)]
arrestsTotal[arrestsTotal['arrests'].gt(30)]
over30.groupby([ 'home_team', 'arrests']).count()
homeStadiumArrests = arrestsTotal.groupby(['home_team']).agg({'arrests': 'sum'}).sort_values(by='arrests', ascending=False)
print(homeStadiumArrests)
arrestsCopy = arrestsTotal
arrestsCopy['resultScore']=arrestsCopy.apply(lambda row: row.home_score - 
                                  (row.away_score), axis = 1) 

homeStadiumLose = arrestsCopy[arrestsCopy['resultScore'].lt(0)]
homeStatiumWin = arrestsCopy[arrestsCopy['resultScore'].gt(0)]
homeStadiumTie = arrestsCopy[arrestsCopy['resultScore'].eq(0)]
loss = homeStadiumLose.groupby(['home_team']).agg({'arrests': 'sum'}).sort_values(by='arrests', ascending=False)

win = homeStatiumWin.groupby(['home_team']).agg({'arrests': 'sum'}).sort_values(by='arrests', ascending=False)
tie = homeStadiumTie.groupby(['home_team']).agg({'arrests': 'sum'}).sort_values(by='arrests', ascending=False)
print(loss[:10])
print(win[:10])
print(tie[:10])
ax = loss.sort_values(by='arrests', ascending=0)[:10].plot(kind='line', figsize=(10, 10), color='red', stacked=False, rot=90)
win.sort_values(by='arrests', ascending=0)[:10].plot(ax=ax, kind='line', color='green', stacked=False, rot=90)

ax.legend(["loss", "win"])
close =  arrestsCopy[arrestsCopy['resultScore'].between(-6, 6, inclusive=True)]
notClose = arrestsCopy[arrestsCopy['resultScore'].le(-24) | arrestsCopy['resultScore'].gt(24)]
                                                                        
closeGraph = close.groupby(['home_team']).agg({'arrests': 'sum'}).sort_values(by='arrests', ascending=False)
notCloseGraph = notClose.groupby(['home_team']).agg({'arrests': 'sum'}).sort_values(by='arrests', ascending=False)

ax = closeGraph.sort_values(by='arrests', ascending=0)[:10].plot(kind='line', figsize=(10, 10), color='green', stacked=False, rot=90)
notCloseGraph.sort_values(by='arrests', ascending=0)[:10].plot(ax=ax, kind='line', color='red', stacked=False, rot=90)

ax.legend(["close", "not close"])
monArrests = len([x for x in arrestsTotal['day_of_week'] if 'Monday' in x])
thursArrests = len([x for x in arrestsTotal['day_of_week'] if 'Thursday' in x])
satArrests = len([x for x in arrestsTotal['day_of_week'] if 'Saturday' in x])
sunArrests = len([x for x in arrestsTotal['day_of_week'] if 'Sunday' in x])
print('The number of Monday arrests are: ' + str(monArrests))
print('The number of Thursday arrests are: ' + str(thursArrests))
print('The number of Saturday arrests are: ' + str(satArrests))
print('The number of Sunday arrests are: ' + str(sunArrests))
yr2011 = arrestsTotal[arrestsTotal.season == 2011].count()
yr2012 = arrestsTotal[arrestsTotal.season == 2012].count()
yr2013 = arrestsTotal[arrestsTotal.season == 2013].count()
yr2014 = arrestsTotal[arrestsTotal.season == 2014].count()
yr2015 = arrestsTotal[arrestsTotal.season == 2015].count()
print(yr2011['season'])
print(yr2012['season'])
print(yr2013['season'])
print(yr2014['season'])
print(yr2015['season'])
