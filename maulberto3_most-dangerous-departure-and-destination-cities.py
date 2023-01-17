import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from matplotlib import pyplot as plt
AirCrashPd = pd.read_csv('../input/Airplane_Crashes_and_Fatalities_Since_1908.csv',sep=',')
AirCrashPd[106:107]
routes=AirCrashPd['Route']

routes=routes.dropna() #drop missing data

routes=routes.str.lower() #lower case

routes=routes.str.split(" - ") #split strings
routes=routes[routes.str.len()==2] #focus on routes involving two cities
clean_routes=pd.DataFrame({'departure': routes.str.get(0), 'destination': routes.str.get(1)})
clean_routes.head()
top_ten_departures=clean_routes['departure'].value_counts()[0:20]
%matplotlib inline

plt.rcParams['figure.figsize'] = (7.0, 4.0)

plt.rc('xtick', labelsize=18)

plt.rc('ytick', labelsize=16)

top_ten_departures.plot(kind='bar')

plt.ylabel('crashes',fontsize=18)
top_ten_destinations=clean_routes['destination'].value_counts()[0:20]
%matplotlib inline

plt.rcParams['figure.figsize'] = (7.0, 4.0)

plt.rc('xtick', labelsize=18)

plt.rc('ytick', labelsize=16)

top_ten_destinations.plot(kind='bar')

plt.ylabel('crashes',fontsize=18)