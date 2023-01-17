import unicodecsv

import numpy as np                # linear algebra

import pandas as pd               # data frames

import seaborn as sns             # visualizations

import matplotlib.pyplot as plt   # visualizations

from scipy import stats

from scipy.stats import norm              # statistics

from sklearn import preprocessing

import datetime as dt









import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import datetime as dt





# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



pd.options.display.float_format = '{:.0f}'.format



data = pd.read_csv("../input/DelayedFlights.csv")

print(data.head())



# Print the info of df

print(data.info())



# Print the shape of df

print(data.shape) 



# Any results you write to the current directory are saved as output.
#Matriz de correlacion

corrmat = data.corr()

f, ax = plt.subplots(figsize=(12, 9))

sns.heatmap(corrmat, vmax=1, square=True);

plt.show()
data['Fecha'] = pd.to_datetime(data.Year*10000+data.Month*100+data.DayofMonth,format='%Y%m%d')

date_delay = data[['Fecha', 'DepDelay']]

date_delay = date_delay.groupby(by='Fecha').sum()

date_delay.head()

plt.figure(figsize=(15,8))

sns.lineplot(data=date_delay, palette="Set3", linewidth=2.5)

plt.axvline(dt.datetime(2008, 7,4), color='red', linestyle=':') #Year, month, day

plt.axvline(dt.datetime(2008, 12,24), color='red', linestyle=':')

plt.axvline(dt.datetime(2008, 11,27), color='red', linestyle=':')

plt.show()
day_delay = data[['DayOfWeek', 'DepDelay']]

#type(date_delay)

day_delay = day_delay.groupby(by='DayOfWeek').sum()

day_delay=day_delay.reset_index(drop=False)

day_delay.head(7)

sns.barplot(x="DayOfWeek", y="DepDelay", data=day_delay, palette="Set3")

plt.show()
daym_delay = data[['DayofMonth', 'DepDelay']]

daym_delay = daym_delay.groupby(by='DayofMonth').sum()

daym_delay=daym_delay.reset_index(drop=False)

daym_delay.head(7)

plt.figure(figsize=(15,8))

sns.barplot(x="DayofMonth", y="DepDelay", data=daym_delay)

plt.show()
#Aerol = pd.DataFrame(data['UniqueCarrier'].value_counts(dropna=False))

#print(Aerol)

Aerol_delay = data[['UniqueCarrier', 'DepDelay']]

Aerol_delay = Aerol_delay.groupby(by='UniqueCarrier').sum()

Aerol_delay=Aerol_delay.reset_index(drop=False)

Aerol_delay.head(7)

plt.figure(figsize=(15,8))

sns.barplot(x="UniqueCarrier", y="DepDelay", data=Aerol_delay,order=['WN', 'AA', 'UA', 'MQ','OO','XE','CO','DL','EV','YV',

                                                                                'US', 'NW','FL', 'B6','OH','9E',

                                                                                 'AS','F9','HA','AQ'])

plt.show()

print(['WN: Southwest Airlines', 'AA: American Airlines', 'MQ: American Eagle Airlines', 'UA: United Airlines',

       'OO: Skywest Airlines','DL: Delta Airlines','XE: ExpressJet','CO: Continental Airlines','US: US Airways',

       'EV: Atlantic Southeast Airlines', 'NW: Northwest Airlines','FL: AirTran Airways','YV: Mesa Airlines', 

       'B6: JetBlue Airways','OH: Comair','9E: Pinnacle Airlines','AS: Alaska Airlines','F9: Frontier Airlines',

       'HA: Hawaiian Airlines','AQ: Aloha Airlines'])
Aerop = pd.DataFrame(data['Origin'].value_counts(dropna=False))

print(Aerop.head(20))
# nos quedamos con el top 20 de aeropuertos con mayores retrasos

Top20airports = data[(data.Origin == 'ORD') | (data.Origin == 'ATL') |

                               (data.Origin == 'DFW') | (data.Origin == 'DEN') |

                               (data.Origin == 'EWR') | (data.Origin == 'LAX') | 

                               (data.Origin == 'IAH') | (data.Origin == 'PHX') |

                               (data.Origin == 'DTW') | (data.Origin == 'SFO') | 

                               (data.Origin == 'LAS') | (data.Origin == 'DEN') |

                               (data.Origin == 'ORD') | (data.Origin == 'JFK') | 

                               (data.Origin == 'CLT') | (data.Origin == 'LGA') |

                               (data.Origin == 'MCO') | (data.Origin == 'MSP') | 

                               (data.Origin == 'BOS') | (data.Origin == 'PHL')]



#print(Top20airports['Origin'].value_counts())

#print(Top20airports.head())



Aerop_delay = Top20airports[['Origin', 'DepDelay']]

Aerop_delay = Aerop_delay.groupby(by='Origin').sum()

Aerop_delay=Aerop_delay.reset_index(drop=False)

Aerop_delay.head(20)

plt.figure(figsize=(15,8))

sns.barplot(x="Origin", y="DepDelay", data=Aerop_delay,order=['ORD', 'ATL', 'DFW', 'DEN','EWR','LAX','IAH','PHX','DTW',

                                                                     'SFO', 'LAS','JFK','CLT', 'LGA','MCO','MSP','BOS','PHL'])





plt.show()

print(['ORD: Chicago', 'ATL: Atlanta', 'DFW: Dallas Fortworth', 'DEN: Denver','EWR: Newark','LAX: Los Ángeles',

       'IAH: Houston','PHX: Phoenix','DTW: Detroit','SFO: San Francisco','LAS: Las Vegas','JFK: New York','CLT: Charlotte',

       'LGA: La Guardia (NY)','MCO: Orlando','MSP: Minneapolis','BOS Boston','PHL Philadelphia'])



pd.DataFrame(data['Cancelled'].value_counts(dropna=False))
pd.DataFrame(data['CancellationCode'].value_counts(dropna=False))
pd.DataFrame(data['Diverted'].value_counts(dropna=False))

# Nos quedamos con los vuelos que fueron cancelados



cancelled = data.loc[data['Cancelled'] == 1]

cancelled.head()

CancelAerp = pd.DataFrame(cancelled['Origin'].value_counts(dropna=False))

print(CancelAerp.head(10))