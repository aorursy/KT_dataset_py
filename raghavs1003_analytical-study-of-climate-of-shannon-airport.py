import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

from scipy import stats

shannonAirportDaily = pd.read_csv("../input/shannon-airport-daily-data/shannon airport daily data.csv")
#pip install pandas-profiling as pp

import pandas_profiling as pp
shannonAirportDaily.head()
shaAirportDaily_1 = shannonAirportDaily[['date', 'maxtp', 'mintp', 'rain', 'sun']]
shaAirportDaily_1.head()
np.mean(shaAirportDaily_1["maxtp"])
np.mean(shaAirportDaily_1["mintp"])
rain_mean = np.mean(shaAirportDaily_1["rain"])
np.mean(shaAirportDaily_1["sun"])
#value = shaAirportDaily_1[(shaAirportDaily_1["sun"] > 3.84)] #and shaAirportDaily_1["rain"] > rain_mean):

value1 = shaAirportDaily_1[((shaAirportDaily_1["rain"] > rain_mean) & (shaAirportDaily_1["sun"] > 3.84))]

print(value1)

print(rain_mean)



np.var(shaAirportDaily_1["maxtp"])
np.var(shaAirportDaily_1["mintp"])
np.var(shaAirportDaily_1["rain"])
np.var(shaAirportDaily_1["sun"])
import matplotlib.pyplot as plt

%matplotlib inline

plt.hist(shaAirportDaily_1['sun'], color = 'yellow', edgecolor = 'black')

plt.title('Histogram of Sunshine (in hours)')

#plt.xlabel('date')

plt.ylabel('Sunshine (in hours)')
import matplotlib.pyplot as plt

%matplotlib inline

plt.hist(shaAirportDaily_1['sun'], color = 'yellow', edgecolor = 'black', cumulative=True)

plt.title('Histogram of Sunlight (in hours)')

plt.xlabel('date')

plt.ylabel('Sunlight (in hours)')
import matplotlib.pyplot as plt

%matplotlib inline

plt.hist(shaAirportDaily_1['sun'], color = 'yellow', edgecolor = 'black', bins=50)

plt.title('Histogram of Sunlight (in hours)')

plt.xlabel('date')

plt.ylabel('Sunlight (in hours)')
%matplotlib inline

import matplotlib.pyplot as plt

plt.hist(shaAirportDaily_1['sun'], color = 'yellow', edgecolor = 'black', bins=50, cumulative=True)

plt.title('Histogram of Sunlight (in hours)')

plt.xlabel('date')

plt.ylabel('Sunlight (in hours)')
import matplotlib.pyplot as plt

%matplotlib inline

plt.hist(shaAirportDaily_1['sun'], color = 'yellow', edgecolor = 'black', bins=100)

plt.title('Histogram of Sunlight (in hours)')

#plt.xlabel('date')

plt.ylabel('Sunlight (in hours)')
import matplotlib.pyplot as plt

%matplotlib inline

plt.hist(shaAirportDaily_1['sun'], color = 'yellow', edgecolor = 'black', bins=100, cumulative=True)

plt.title('Histogram of Sunlight (in hours)')

plt.xlabel('date')

plt.ylabel('Sunlight (in hours)')
stats.mode(shaAirportDaily_1["sun"])
stats.mode(shaAirportDaily_1["rain"])
stats.mode(shaAirportDaily_1["mintp"])
stats.mode(shaAirportDaily_1["maxtp"])
np.median(shaAirportDaily_1["sun"])
np.median(shaAirportDaily_1["rain"])
np.median(shaAirportDaily_1["maxtp"])
np.median(shaAirportDaily_1["mintp"])
shaAirportDaily_1.plot.box(whis=1.5, figsize=(13,7));

#y=['sun']

shaAirportDaily_1.plot.box(y=['rain']);
shaAirportDaily_1.plot.box(y=['mintp']);
shaAirportDaily_1.plot.box(y=['maxtp']);
#shaAirportDaily_1['sun'] = shaAirportDaily_1['sun'].replace(regex='0',value=2.8)

shaAirportDaily_2 = shaAirportDaily_1['sun'].replace(to_replace=0,value=2.8,inplace=False,limit=None,regex=False, method='pad')
shaAirportDaily_2.head()
import matplotlib.pyplot as plt

%matplotlib inline

plt.hist(shaAirportDaily_2, color = 'yellow', edgecolor = 'black', bins=50)

plt.title('Histogram of Sunlight (in hours)')

plt.xlabel('date')

plt.ylabel('Sunlight (in hours)')
shaAirportDaily_1.loc[shaAirportDaily_1['sun'].idxmax()]
shaAirportDaily_1.loc[shaAirportDaily_1['maxtp'].idxmax()]
shaAirportDaily_1.loc[shaAirportDaily_1['mintp'].idxmax()]
shaAirportDaily_1.loc[shaAirportDaily_1['rain'].idxmax()]
shaAirportDaily_1.loc[shaAirportDaily_1['sun'].idxmin()]
shaAirportDaily_1.loc[shaAirportDaily_1['mintp'].idxmin()]
shaAirportDaily_1.loc[shaAirportDaily_1['maxtp'].idxmin()]
shaAirportDaily_1.loc[shaAirportDaily_1['rain'].idxmin()]
import matplotlib.pyplot as plt

%matplotlib inline

plt.hist(shaAirportDaily_1['maxtp'], color = 'orange', edgecolor = 'black')

plt.title('Histogram of Max temperature (in degree celcius)')

#plt.xlabel('bins')

plt.ylabel('maximum temperature (in degree celcius)')
import matplotlib.pyplot as plt

%matplotlib inline

plt.hist(shaAirportDaily_1['rain'], color = 'blue', edgecolor = 'black')

plt.title('Histogram of rain (in mm)')

#plt.xlabel('bins')

plt.ylabel('rain (in mm)')
%matplotlib inline

import matplotlib.pyplot as plt

plt.hist(shaAirportDaily_1['mintp'], color = 'c', edgecolor = 'black')

plt.title('Histogram of Min temperature (in degree celcius)')

#plt.xlabel('bins')

plt.ylabel('minimum temperature (in degree celcius)')
shaAirportDaily_1.plot.box(y=['mintp']);
#pp.ProfileReport(shaAirportDaily_1)

report = pp.ProfileReport(shaAirportDaily_1)

report.to_file('profile_report.html')

import os

os.getcwd()
from scipy import stats

fvalue, pvalue = stats.f_oneway(shaAirportDaily_1["sun"],shaAirportDaily_1["rain"],shaAirportDaily_1["maxtp"],shaAirportDaily_1["mintp"])

print('fvalue',fvalue)

print('pvalue',pvalue)
from scipy import stats

fvalue, pvalue = stats.f_oneway(shaAirportDaily_1["sun"],shaAirportDaily_1["rain"])

print('fvalue',fvalue)

print('pvalue',pvalue)
from scipy import stats

fvalue, pvalue = stats.f_oneway(shaAirportDaily_1["maxtp"],shaAirportDaily_1["mintp"])

print('fvalue',fvalue)

print('pvalue',pvalue)
from scipy import stats

fvalue, pvalue = stats.f_oneway(shaAirportDaily_1["sun"],shaAirportDaily_1["maxtp"])

print('fvalue',fvalue)

print('pvalue',pvalue)
from scipy import stats

fvalue, pvalue = stats.f_oneway(shaAirportDaily_1["sun"],shaAirportDaily_1["mintp"])

print('fvalue',fvalue)

print('pvalue',pvalue)
from scipy import stats

fvalue, pvalue = stats.f_oneway(shaAirportDaily_1["rain"],shaAirportDaily_1["mintp"])

print('fvalue',fvalue)

print('pvalue',pvalue)