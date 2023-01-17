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

import pandas as pd
data = pd.read_csv('/kaggle/input/crime-rates/report.csv')
newdata = data.rename(columns = {"crimes_percapita" : "crime_rate"})

newdata2 = newdata.rename(columns = {'agency_jurisdiction':'city'})

#134 

newdata2[newdata2.city=='Chicago, IL'].crime_rate.mean()
#143

newdata2[newdata2.city=='Chicago, IL'].mean().crime_rate
#243

newdata2.query('city == "Chicago, IL"').mean().crime_rate
#423 

newdata2.mean().query('city =="Chicago, IL"').crime_rate
#1234

newdata2.groupby('city').crime_rate.mean().loc['Chicago, IL']
#1324

newdata2.groupby('city').mean().crime_rate.loc['Chicago, IL']
#1423

newdata2.groupby('city').loc['Chicago, IL'].crime_rate.mean()
#1342

newdata2.groupby('city').mean().loc['Chicago, IL'].crime_rate
#231

newdata2.pivot(index='report_year', columns='city', values='crime_rate').loc['Chicago, IL'].mean()
#321

newdata2.loc['Chicago, IL'].pivot(index='report_year', columns='city', values='crime_rate').mean()
#312

newdata2.loc['Chicago, IL'].mean().pivot(index='report_year', columns='city', values='crime_rate')
#123

newdata2.mean().pivot(index='report_year', columns='city', values='crime_rate').loc['Chicago, IL']