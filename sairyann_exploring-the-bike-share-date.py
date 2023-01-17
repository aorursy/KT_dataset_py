# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from pandas.tseries.holiday import USFederalHolidayCalendar

from pandas.tseries.offsets import CustomBusinessDay

from datetime import datetime

import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
trips = pd.read_csv("../input/trip.csv")

weather = pd.read_csv("../input/weather.csv")

stations = pd.read_csv("../input/station.csv")
trips.head()
trips.isnull().sum()

trips['start_date']
trips.describe()
trips.duration /= 60 # change to minutes
trips.duration.describe()
# how many trips are not real?  i.e. short and stop/start in the same stations?

n = ((trips.start_station_id == trips.end_station_id) & (trips.duration<5))

n.sum()
n.sum()/n.count() # not a lot
# lets remove these outliers

trips=trips[~n]
trips.head()
# let's also remove the really long guys... more than 8 hours

trips = trips[trips.duration<60*8]

trips.duration.describe()
# check that the number of stations makes sense

trips.start_station_name.describe()

# there are too many stations!  let's find them...
# too many stations!  Let's find the misspelled stations

start_stations = trips.duration.groupby([trips.start_station_id,trips.start_station_name]).count()

start_stations
# replace the four misspelled names... 

trips=trips.replace({'San Jose Government Center':'Santa Clara County Civic Center','Broadway at Main':'Stanford in Redwood City'})

start_stations[35:50]
trips=trips.replace({'Washington at Kearny':'Washington at Kearney','Post at Kearny':'Post at Kearney'})

trips.start_station_name.describe()
# what is the most used station?

beg=trips.start_station_name.value_counts()

end=trips.end_station_name.value_counts()

total_stations = beg+end

total_stations =total_stations.order(ascending=False)

total_stations.head()
total_stations.tail() # and the least used
# but... some of these stations were installed much later.  can first do a per diem usage. 

stations=pd.read_csv("../input/station.csv")

stations.head()
stations.installation_date[1]
# convert to datetime

stations.installation_date = pd.to_datetime(stations.installation_date,format="%m/%d/%Y")

trips.start_date = pd.to_datetime(trips.start_date,format="%m/%d/%Y %H:%M").dt.date

end_time = trips.start_date.max()

trips['start_date'].head()
# now can we do math on it

stations['Days_operating']=(end_time - stations.installation_date).dt.days

stations.Days_operating.head()
# merge this with the stations visitation series

ts = pd.DataFrame(total_stations,columns=['Total_visits'])

stations=pd.merge(stations,ts,left_on='name',right_index=True,how='outer')

stations.head()
# make a per-diem total visits

stations['visits_per_diem']=stations['Total_visits']/stations['Days_operating']

stations[['name','visits_per_diem']].sort_values(by='visits_per_diem',ascending=False)
# Lets now plot the top 5 visited stations as a function of day

# add a "day" column into trips

trips['day']=trips.start_date.dt.weekday

# then group by start station and day

day_count = trips.duration.groupby([trips.start_station_name,trips.day]).count()

day_count = day_count.unstack()

# plot the top 5

day_count.ix[total_stations.index[0:5]].T.plot(); plt.ylabel('Total Count')
# It's interesting that the caltrain stops are primarily business day routes.  

# lets compare business day to non-business day

calendar = USFederalHolidayCalendar()

us_business_days = CustomBusinessDay(calendar=USFederalHolidayCalendar())

business_days = pd.DatetimeIndex(start=trips.start_date.min(),end = trips.start_date.max(),freq=us_business_days)

business_days = pd.to_datetime(business_days,format="%Y/%m/%d").date

business_days
trips['business_day']=trips.start_date.isin(business_days)

# relabel into business and weekend

trips.business_day=trips.business_day.map(lambda x: 'Work' if x==True else 'Weekend')

trips[['start_date','business_day']].head() 
workday_count = trips.duration.groupby([trips.start_station_name,trips.business_day]).count()

workday_count = workday_count.unstack()

workday_count.ix[total_stations.index[0:5]].plot(kind='barh',stacked=True)
workday_pct = workday_count.div(workday_count.sum(1).astype(float),axis=0)

workday_pct.sort_values(by='Weekend')
workday_pct.ix[total_stations.index[0:5]].plot(kind='barh',stacked=True)