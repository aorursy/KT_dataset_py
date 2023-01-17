import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats.mstats import gmean
import os, datetime
#from cachier import cachier

#@cachier(stale_after=datetime.timedelta(hours=12))
def get_source_data(url: str) -> pd.DataFrame:
    return pd.read_csv(url)
csv = 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv'

df = get_source_data(csv)
df = df.drop(['Province/State', 'Lat', 'Long'], axis=1)
dfg = df.groupby(['Country/Region']).sum()

# filter only countries with large last series values
dfg = dfg[dfg.iloc[:,-1]>100]
c = dfg.columns
horizon = 10 # days

# filter up to last hotizon days 
dfp = dfg.drop(c[0:len(c)-horizon],axis=1)

# filter only countries with starting cases more than 100
dfp = dfp[dfp.iloc[:,0]>100]

dfpcoefs : pd.DataFrame = dfp

# add day-by-day increases
for i in range(1,horizon):
    c = dfpcoefs.columns
    # kwargs = {'f'+str(i) : lambda x: (x[c[-i]]-x[c[-i-1]]) / x[c[-i-1]]}
    # dfpcoefs = dfpcoefs.assign(**kwargs)
    dfpcoefs['day-'+str(i)] = (dfpcoefs[c[-i]]-dfpcoefs[c[-i-1]]) / dfpcoefs[c[-i-1]]
    dfpcoefs = dfpcoefs.drop(dfpcoefs.columns[-i-1] , axis = 1)

dfpcoefs = dfpcoefs.drop(dfpcoefs.columns[0] , axis = 1)

dfpcoefs['ROWMEAN'] = gmean(dfpcoefs + 1, axis = 1) - 1

# filter only the top countries by average daily increase of new cases
# as this is day-to-day increase, it needs to be geometrical mean
dpfcoefT = dfpcoefs.sort_values(by = 'ROWMEAN', ascending = False).head(25).T

# calculate the daily mean increase in cases
# this is average over countries, so it is arithmetical mean
dpfcoefT['COLMEAN'] = dpfcoefT.mean(axis = 1)

# This table represents the daily increases in %
# The countries are sorted by average over the defined horizon (e.g. 10 days)
dpfcoefT.T
large = 22; med = 16; small = 12

params = {'axes.titlesize': large,
          'legend.fontsize': med,
          'figure.figsize': (12, 6),
          'axes.labelsize': med,
          'axes.titlesize': med,
          'xtick.labelsize': med,
          'ytick.labelsize': med,
          'figure.titlesize': large}

plt.rcParams.update(params)
plt.style.use('seaborn-whitegrid')
# general trend in the percentile increase of daily cases 
dpfcoefT.iloc[0:-1,]['COLMEAN'][::-1].plot(title = "Global trend in daily increase for " + str(datetime.date.today()))

target = 'Canada'

use_log = True # Enable/disable logarithmic y-axis

plotdata = dfg.loc[target].to_frame()

fig = plt.figure()
ax = fig.add_subplot(111)

if use_log:
  ax.set_yscale('log')
  fig.suptitle(target + ' LOG')
else:
  fig.suptitle(target)
    
fig.subplots_adjust(top=0.85)
ax.xaxis.set_major_locator(plt.AutoLocator())

#plotting the last 100-fold increase
llim = max(plotdata[target])*0.01
ax.plot(plotdata[plotdata[target]>llim])
plt.show()