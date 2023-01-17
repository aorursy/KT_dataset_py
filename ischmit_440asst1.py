import pandas as pd

from matplotlib import pyplot as plt

import matplotlib

import re



#read in and filter data

crashes = pd.read_csv("../input/Airplane_Crashes_and_Fatalities_Since_1908.csv")

crashes = crashes.tail(579) #limit to 2000s



#build relevant dataframe

op = crashes.loc[:, "Operator"]

ab = crashes.loc[:, "Aboard"]

ft = crashes.loc[:, "Fatalities"]

gd = crashes.loc[:, "Ground"]

tp = crashes.loc[:, "Type"]



rel = pd.DataFrame()

rel["airline"] = op

rel["count"] = 1

rel["aboard"] = ab

rel["fatalities"] = ft

rel["bystanderFatalities"] = gd

rel["aircraftType"] = tp



#remove duplicates and sum values

cld = rel.groupby(rel.airline).sum()

typ = rel.groupby(rel.aircraftType).sum()



#extrapolate data

rto = cld.fatalities/cld.aboard

cld["percentageDead"] = rto

tot = cld.fatalities+cld.bystanderFatalities

cld["totalFatalities"] = tot

trto = typ.fatalities/typ.aboard

typ["percentageDead"] = trto

ttot = typ.fatalities+typ.bystanderFatalities

typ["totalFatalities"] = ttot



#set graph size

matplotlib.rcParams['figure.figsize'] = (12.0, 8.0)



plt.figure(1)

ncpa = cld["count"].sort_values(ascending = False)

ncpa.head(100).plot(kind='bar', title='number of crashes per airline') #limited to top 100



plt.figure(8)

ncpa = typ["count"].sort_values(ascending = False)

ncpa.head(100).plot(kind='bar', title='number of crashes per aircraft') #limited to top 100



plt.figure(2)

ncpa = cld["fatalities"].sort_values(ascending = False)

ncpa.head(100).plot(kind='bar', title='passenger fatalities per airline') #limited to top 100



plt.figure(9)

ncpa = typ["fatalities"].sort_values(ascending = False)

ncpa.head(100).plot(kind='bar', title='passenger fatalities per aircraft') #limited to top 100



plt.figure(3)

ncpa = cld["bystanderFatalities"].sort_values(ascending = False)

ncpa.head(100).plot(kind='bar', title='non-passenger fatalities per airline') #limited to top 100



plt.figure(10)

ncpa = typ["bystanderFatalities"].sort_values(ascending = False)

ncpa.head(100).plot(kind='bar', title='non-passenger fatalities per aircraft') #limited to top 100



plt.figure(4)

ncpa = cld["totalFatalities"].sort_values(ascending = False)

ncpa.head(100).plot(kind='bar', title='total fatalities per airline') #limited to top 100



plt.figure(11)

ncpa = typ["totalFatalities"].sort_values(ascending = False)

ncpa.head(100).plot(kind='bar', title='total fatalities per aircraft') #limited to top 100



plt.figure(5)

lodic = cld["percentageDead"].sort_values(ascending = False)

lodic.tail(100).plot(kind='bar', title='liklihood of dying in a crash per airline') #limited to top 100



plt.figure(12)

lodic = typ["percentageDead"].sort_values(ascending = False)

lodic.tail(100).plot(kind='bar', title='liklihood of dying in a crash per aircraft') #limited to top 100



plt.figure(6)

ax = plt.subplot(111)

cld.plot(kind='scatter', x='count', y='totalFatalities',ax=ax, c='percentageDead', title='number of fatalities by number of crashes per airline')

ax.set_xlim(0,6)

ax.set_ylim(0,3500)



plt.figure(7)

ax = plt.subplot(111)

cld.plot(kind='scatter', x='count', y='totalFatalities',ax=ax, c='percentageDead', title='number of fatalities by number of crashes (excl. 9/11) ')

ax.set_xlim(0,6)

ax.set_ylim(0,400)



plt.figure(13)

ax = plt.subplot(111)

typ.plot(kind='scatter', x='count', y='totalFatalities',ax=ax, c='percentageDead', title='number of fatalities by number of crashes per aircraft')

ax.set_xlim(0,20)

ax.set_ylim(0,3500)