import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from matplotlib import pyplot as plt

%matplotlib inline
AirCrashPd = pd.read_csv('../input/Airplane_Crashes_and_Fatalities_Since_1908.csv',sep=',')
AirCrashPd[106:107]
routes=AirCrashPd['Route']

routes=routes.dropna() #drop missing data

routes=routes.str.lower() #lower case

routes=routes.str.split(" - ") #split strings
routes=routes[routes.str.len()==2] #focus on routes involving two cities
clean_routes=pd.DataFrame({'departure': routes.str.get(0), 'destination': routes.str.get(1)})
clean_routes.head()
cols=clean_routes.columns.tolist()
top_cities=clean_routes[cols[0]].value_counts()[0:20]
from ipywidgets import interact

#import matplotlib as mpl

@interact( Route=cols, n_cities=(10,25))

def plot(Route,n_cities):

    top_cities=clean_routes[Route].value_counts()[0:n_cities]

    plt.rc('xtick', labelsize=14)

    plt.rc('ytick', labelsize=14)

    top_cities.plot(kind='bar',color="orange")

    plt.ylabel('total n. of crashes',fontsize=14)