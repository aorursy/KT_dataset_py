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

ds = pd.read_csv('../input/australia-road-fatal-crashes/ardd_fatal_crashes.csv', index_col=['Year']) 

ds.head(100000)

#ds.iloc[0]
col = ds.columns

print(col)
import pandas as pd

import matplotlib.pyplot as plt

ds = pd.read_csv('../input/australia-road-fatal-crashes/ardd_fatal_crashes.csv', index_col=['Crash ID']) 

filteredData = ds.Year == 2019

(ds[filteredData])['State'].value_counts().sort_index().plot.bar()

ds['Month'].value_counts().plot.bar()
ds['Year'].value_counts().plot.bar()
ds['Dayweek'].value_counts().plot.bar()
ds['Time'].value_counts().plot.bar()
ds['Crash Type'].value_counts().plot.bar()
ds['Number Fatalities'].value_counts().plot.bar()
ds['Speed Limit'].value_counts().plot.line()
ds['National Remoteness Areas'].value_counts().plot.bar()
ds['SA4 Name 2016'].value_counts().plot.bar()
ds['National Road Type'].value_counts().plot.bar()
ds['Christmas Period'].value_counts().plot.bar()
ds['Easter Period'].value_counts().plot.bar()
ds['Day of week'].value_counts().plot.bar()
ds['Time of Day'].value_counts().plot.bar()
import matplotlib.pyplot as plt

ds['State'].value_counts().plot.bar()

plt.xlabel('State')  

plt.ylabel("# of Accidents") 

plt.show()
import matplotlib.pyplot as plt

ds['Year'].value_counts().plot.bar()

plt.xlabel('Year')  

plt.ylabel("# of Accidents") 

plt.show()
ds['Speed Limit'].value_counts().plot.line()

plt.xlabel('Speed Limit')  

plt.ylabel("# of Accidents") 
import pandas as pd

import matplotlib.pyplot as plt

ds = pd.read_csv('../input/australia-road-fatal-crashes/ardd_fatal_crashes.csv', index_col=['Crash ID'])

filteredData = ds.Year == 2019

(ds[filteredData])['State'].value_counts().sort_index().plot.bar() 

plt.xlabel('State')  

plt.ylabel("# of Accidents")
ds['Time of Day'].value_counts().plot.bar()

plt.xlabel('TimeOfDay')  

plt.ylabel("# of Accidents")
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

ds = pd.read_csv('../input/australia-road-fatal-crashes/ardd_fatal_crashes.csv', index_col=['Crash ID'])

new_ds = pd.DataFrame(ds)  

new_ds.rename(columns = {'Crash Type':'CrashType', 'Number Fatalities':'NumberFatalities','Bus \nInvolvement':'BusInvolvement',

                         'Heavy Rigid Truck Involvement':'HeavyRigidTruckInvolvement', 'Articulated Truck Involvement':'ArticulatedTruckInvolvement',

                         'Speed Limit':'SpeedLimit','Christmas Period':'ChristmasPeriod','Easter Period':'EasterPeriod','Day of week':'DayOfWeek', 'Time of Day':'TimeOfDay'}, inplace = True)

filteredData2 = new_ds.TimeOfDay =='Night'

(new_ds[filteredData2])['Dayweek'].value_counts().sort_index().plot.bar() 

plt.xlabel('Days of week')  

plt.ylabel("# of Accidents in Night")
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

ds = pd.read_csv('../input/australia-road-fatal-crashes/ardd_fatal_crashes.csv', index_col=['Crash ID'])

new_ds = pd.DataFrame(ds)  

new_ds.rename(columns = {'Crash Type':'CrashType', 'Number Fatalities':'NumberFatalities','Bus \nInvolvement':'BusInvolvement',

                         'Heavy Rigid Truck Involvement':'HeavyRigidTruckInvolvement', 'Articulated Truck Involvement':'ArticulatedTruckInvolvement',

                         'Speed Limit':'SpeedLimit','Christmas Period':'ChristmasPeriod','Easter Period':'EasterPeriod','Day of week':'DayOfWeek', 'Time of Day':'TimeOfDay'}, inplace = True)

filteredData2 = new_ds.TimeOfDay =='Day'

(new_ds[filteredData2])['Dayweek'].value_counts().sort_index().plot.bar() 

plt.xlabel('Days of week')  

plt.ylabel("# of Accidents in Day time")
new_ds = pd.DataFrame(ds)  

new_ds.rename(columns = {'Crash Type':'CrashType', 'Number Fatalities':'NumberFatalities','Bus \nInvolvement':'BusInvolvement',

                         'Heavy Rigid Truck Involvement':'HeavyRigidTruckInvolvement', 'Articulated Truck Involvement':'ArticulatedTruckInvolvement',

                         'Speed Limit':'SpeedLimit','Christmas Period':'ChristmasPeriod','Easter Period':'EasterPeriod','Day of week':'DayOfWeek', 'Time of Day':'TimeOfDay'}, inplace = True)

tod = new_ds.TimeOfDay

counts = tod.value_counts()

counts

percent = tod.value_counts(normalize=True)

percent

percent100 = tod.value_counts(normalize=True).mul(100).round(2).astype(str)+'%'

percent100
new_ds = pd.DataFrame(ds)  

new_ds.rename(columns = {'Crash Type':'CrashType', 'Number Fatalities':'NumberFatalities','Bus \nInvolvement':'BusInvolvement',

                         'Heavy Rigid Truck Involvement':'HeavyRigidTruckInvolvement', 'Articulated Truck Involvement':'ArticulatedTruckInvolvement',

                         'Speed Limit':'SpeedLimit','Christmas Period':'ChristmasPeriod','Easter Period':'EasterPeriod','Day of week':'DayOfWeek', 'Time of Day':'TimeOfDay'}, inplace = True)



percent100 = pd.crosstab(new_ds.TimeOfDay,new_ds.DayOfWeek, normalize='index').rename_axis(None)

percent100 *=100

percent100



plt.bar(percent100.index,percent100.Weekday)

plt.show()
percent100 = pd.crosstab(new_ds.TimeOfDay,new_ds.DayOfWeek, normalize='index').rename_axis(None)

percent100 *=100

percent100
plt.bar(percent100.index,percent100.Weekday)

plt.show()
plt.bar(percent100.index,percent100.Weekend)

plt.show()
import matplotlib.pyplot as plt

import matplotlib.ticker as mtick



# create dummy variable then group by that

# set the legend to false because we'll fix it later

new_ds.assign(dummy = 1).groupby(['dummy','TimeOfDay']).size().groupby(level=0).apply(lambda x: 100 * x / x.sum()).to_frame().unstack().plot(kind='bar',stacked=True,legend=False)



# or it'll show up as 'dummy'

plt.xlabel('Time of Day')

plt.ylabel('% of Fatal Crashes')

plt.title("Percentage of Fatal Crashes based on Time of the Day ", y=1.05);



# disable ticks in the x axis

plt.xticks([])



# fix the legend or it'll include the dummy variable

current_handles, _ = plt.gca().get_legend_handles_labels()

reversed_handles = reversed(current_handles)

correct_labels = reversed(new_ds['TimeOfDay'].unique())



plt.legend(reversed_handles,correct_labels)



plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter())

plt.show()
percent100 = tod.value_counts(normalize=True).mul(100).round(2)

percent100
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

ds = pd.read_csv('../input/australia-road-fatal-crashes/ardd_fatal_crashes.csv', index_col=['Crash ID'])

new_ds = pd.DataFrame(ds)  

new_ds.rename(columns = {'Crash Type':'CrashType', 'Number Fatalities':'NumberFatalities','Bus \nInvolvement':'BusInvolvement',

                         'Heavy Rigid Truck Involvement':'HeavyRigidTruckInvolvement', 'Articulated Truck Involvement':'ArticulatedTruckInvolvement',

                         'Speed Limit':'SpeedLimit','Christmas Period':'ChristmasPeriod','Easter Period':'EasterPeriod','Day of week':'DayOfWeek', 'Time of Day':'TimeOfDay'}, inplace = True)



new_ds.TimeOfDay.value_counts()
data = {'Time of Day': ["Day", "Night"], 'count':[26143, 19816]}

new_ds = pd.DataFrame.from_dict(data)

print(new_ds)



new_ds['Percentage Value'] = ((new_ds['count'] / new_ds['count'].sum()) * 100).round(2)

print(new_ds)



new_ds.plot(kind='bar',ylim =(0,100), x='Time of Day',y='Percentage Value', color='green')

plt.title("Percentage of Fatal Crashes based on Time of the Day ", y=1.05);

plt.ylabel("% of Fatal Crashes", labelpad=14)
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

ds = pd.read_csv('../input/australia-road-fatal-crashes/ardd_fatal_crashes.csv', index_col=['Crash ID'])

new_ds = pd.DataFrame(ds)  

new_ds.rename(columns = {'Crash Type':'CrashType', 'Number Fatalities':'NumberFatalities','Bus \nInvolvement':'BusInvolvement',

                         'Heavy Rigid Truck Involvement':'HeavyRigidTruckInvolvement', 'Articulated Truck Involvement':'ArticulatedTruckInvolvement',

                         'Speed Limit':'SpeedLimit','Christmas Period':'ChristmasPeriod','Easter Period':'EasterPeriod','Day of week':'DayOfWeek', 'Time of Day':'TimeOfDay'}, inplace = True)

filteredData2 = new_ds.TimeOfDay =='Day'

(new_ds[filteredData2])['Dayweek'].value_counts().sort_index().plot.bar() 

plt.xlabel('Days of week')  

plt.ylabel("# of Accidents in Daytime")

plt.title("Percentage of Fatal Crashes in Daytime ", y=1.05);



import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

ds = pd.read_csv('../input/australia-road-fatal-crashes/ardd_fatal_crashes.csv', index_col=['Crash ID'])

new_ds = pd.DataFrame(ds)  

new_ds.rename(columns = {'Crash Type':'CrashType', 'Number Fatalities':'NumberFatalities','Bus \nInvolvement':'BusInvolvement',

                         'Heavy Rigid Truck Involvement':'HeavyRigidTruckInvolvement', 'Articulated Truck Involvement':'ArticulatedTruckInvolvement',

                         'Speed Limit':'SpeedLimit','Christmas Period':'ChristmasPeriod','Easter Period':'EasterPeriod','Day of week':'DayOfWeek', 'Time of Day':'TimeOfDay'}, inplace = True)



filteredData2 = new_ds.TimeOfDay =='Day'

(new_ds[filteredData2])['Dayweek'].value_counts()
data = {'Days of Week': ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"], 'count':[3497, 3513, 3625, 3672, 3986, 4057, 3793]}

new_ds = pd.DataFrame.from_dict(data)

print(new_ds)



new_ds['Percentage Value'] = ((new_ds['count'] / new_ds['count'].sum()) * 100).round(2)

print(new_ds)



new_ds.plot(kind='bar',ylim =(0,20), x='Days of Week',y='Percentage Value', color='green')

plt.title("Percentage of Fatal Crashes based on Daytime of the Day ", y=1.05);

plt.ylabel("% of Fatal Crashes in Daytime", labelpad=14)
new_ds.plot(kind='bar',ylim =(0,100), x='Days of Week',y='Percentage Value', color='green')

plt.title("Percentage of Fatal Crashes based on Daytime of the Day ", y=1.05);

plt.ylabel("% of Fatal Crashes in Daytime", labelpad=14)
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

ds = pd.read_csv('../input/australia-road-fatal-crashes/ardd_fatal_crashes.csv', index_col=['Crash ID'])

new_ds = pd.DataFrame(ds)  

new_ds.rename(columns = {'Crash Type':'CrashType', 'Number Fatalities':'NumberFatalities','Bus \nInvolvement':'BusInvolvement',

                         'Heavy Rigid Truck Involvement':'HeavyRigidTruckInvolvement', 'Articulated Truck Involvement':'ArticulatedTruckInvolvement',

                         'Speed Limit':'SpeedLimit','Christmas Period':'ChristmasPeriod','Easter Period':'EasterPeriod','Day of week':'DayOfWeek', 'Time of Day':'TimeOfDay'}, inplace = True)



filteredData2 = new_ds.TimeOfDay =='Night'

(new_ds[filteredData2])['Dayweek'].value_counts()
data = {'Days of Week': ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"], 'count':[1877, 1900, 2176, 2599, 3526, 4292, 3446]}

new_ds = pd.DataFrame.from_dict(data)

print(new_ds)



new_ds['Percentage Value'] = ((new_ds['count'] / new_ds['count'].sum()) * 100).round(2)

print(new_ds)



new_ds.plot(kind='bar',ylim =(0,25), x='Days of Week',y='Percentage Value', color='green')

plt.title("Percentage of Fatal Crashes based on Night Time of the Day ", y=1.05);

plt.ylabel("% of Fatal Crashes in Nighttime", labelpad=14)
data = {'Days of Week': ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"], 'count':[1877, 1900, 2176, 2599, 3526, 4292, 3446]}

new_ds = pd.DataFrame.from_dict(data)

print(new_ds)



new_ds['Percentage Value'] = ((new_ds['count'] / new_ds['count'].sum()) * 100).round(2)

print(new_ds)



new_ds.plot(kind='bar',ylim =(0,100), x='Days of Week',y='Percentage Value', color='green')

plt.title("Percentage of Fatal Crashes based on Night Time of the Day ", y=1.05);

plt.ylabel("% of Fatal Crashes in Nighttime", labelpad=14)
import matplotlib.pyplot as plt

ds['Year'].value_counts().plot.bar()

plt.xlabel('Year')  

plt.ylabel("# of Accidents") 

plt.show()
data = {'Year': ["1989", "1990", "1991","1992","1993","1994","1995","1996","1997","1998","1999","2000",

                "2001", "2002", "2003", "2004","2005","2006", "2007","2008","2009","2010",

                "2011","2012","2013","2014","2015","2016","2017","2018","2019"], 'tot_population':[16936723,17169768,17378981,17557133,17719090,17893433,18119616,18330079,18510004,18705620,18919210,

                19141036,19386461,19605441,19827155,20046003,20311543,20627547,21016121,21475625,21865623,22172469,

                22522197,22928023,23297777,23640331,23984581,24389684,24773350,25171439,25464116]}

new_ds = pd.DataFrame.from_dict(data)

print(new_ds)
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

ds = pd.read_csv('../input/australia-road-fatal-crashes/ardd_fatal_crashes.csv', index_col=['Crash ID'])

new_ds = pd.DataFrame(ds)  

new_ds.rename(columns = {'Crash Type':'CrashType', 'Number Fatalities':'NumberFatalities','Bus \nInvolvement':'BusInvolvement',

                         'Heavy Rigid Truck Involvement':'HeavyRigidTruckInvolvement', 'Articulated Truck Involvement':'ArticulatedTruckInvolvement',

                         'Speed Limit':'SpeedLimit','Christmas Period':'ChristmasPeriod','Easter Period':'EasterPeriod','Day of week':'DayOfWeek', 'Time of Day':'TimeOfDay'}, inplace = True)

new_ds.Year.value_counts()
data = {'Year': ["1989", "1990", "1991","1992","1993","1994","1995","1996","1997","1998","1999","2000",

                "2001", "2002", "2003", "2004","2005","2006", "2007","2008","2009","2010",

                "2011","2012","2013","2014","2015","2016","2017","2018","2019","2020"], 

        'tot # of Accidents':[2407,2050,1874,1736,1737,1702,1822,1768,1601,1573,1553,1628,

                              1584,1525,1445,1444,1472,1452,1453,1315,1347,1233,

                              1151,1190,1101,1051,1100,1198,1125,1055,1108,159],

        'tot_population':[16936723,17169768,17378981,17557133,17719090,17893433,18119616,18330079,18510004,18705620,18919210,

                19141036,19386461,19605441,19827155,20046003,20311543,20627547,21016121,21475625,21865623,22172469,

                22522197,22928023,23297777,23640331,23984581,24389684,24773350,25171439,25464116,25444328]}

new_ds = pd.DataFrame.from_dict(data)

print(new_ds)
new_ds['per 10000 ppl'] = ((new_ds['tot # of Accidents'] / new_ds['tot_population']) * 10000).round(2)

print(new_ds)



new_ds.plot(kind='bar', x='Year',y='per 10000 ppl',ylim =(0.0,1.6), color='green')

plt.title("Yearly Fatal Crashes per 10,000 People", y=1.05);

plt.ylabel("# of Fatal Crashes per 10,000 People", labelpad=14)
new_ds['per 100000 ppl'] = ((new_ds['tot # of Accidents'] / new_ds['tot_population']) * 100000).round(0)

print(new_ds)



new_ds.plot(kind='bar', x='Year',y='per 100000 ppl', color='green', ylim=(0,16))

plt.title("Yearly Fatal Crashes per 100,000 People", y=1.05);

plt.ylabel("# of Fatal Crashes per 100,000 People", labelpad=14)
import pandas as pd

import matplotlib.pyplot as plt

ds = pd.read_csv('../input/australia-road-fatal-crashes/ardd_fatal_crashes.csv', index_col=['Crash ID'])

filteredData = ds.Year == 2019

(ds[filteredData])['State'].value_counts().sort_index().plot.bar() 

plt.xlabel('State')  

plt.ylabel("# of Accidents")
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

ds = pd.read_csv('../input/australia-road-fatal-crashes/ardd_fatal_crashes.csv', index_col=['Crash ID'])

new_ds = pd.DataFrame(ds)  

new_ds.rename(columns = {'Crash Type':'CrashType', 'Number Fatalities':'NumberFatalities','Bus \nInvolvement':'BusInvolvement',

                         'Heavy Rigid Truck Involvement':'HeavyRigidTruckInvolvement', 'Articulated Truck Involvement':'ArticulatedTruckInvolvement',

                         'Speed Limit':'SpeedLimit','Christmas Period':'ChristmasPeriod','Easter Period':'EasterPeriod','Day of week':'DayOfWeek', 'Time of Day':'TimeOfDay'}, inplace = True)



filteredData = new_ds.Year == 2019

(new_ds[filteredData])['State'].value_counts()
data = {'State': ["NSW", "VIC", "QLD","WA","SA","TAS","NT","ACT"], 

        'tot # of Accidents':[331,252,195,154,110,32,28,6],

        'tot_population':[8117976,6629870,5115451,2630557,1756494,535500,245562,428060]}

new_ds = pd.DataFrame.from_dict(data)

print(new_ds)
new_ds['per 10000 ppl'] = ((new_ds['tot # of Accidents'] / new_ds['tot_population']) * 10000).round(2)

print(new_ds)



new_ds.plot(kind='bar', x='State',y='per 10000 ppl', color='green', ylim=(0.0,1.6))

plt.title("2019 Each State Fatal Crashes per 10,000 People", y=1.05);

plt.ylabel("# of Fatal Crashes per 10,000 People", labelpad=14)
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

ds = pd.read_csv('../input/australia-road-fatal-crashes/ardd_fatal_crashes.csv', index_col=['Crash ID'])

new_ds = pd.DataFrame(ds)  

new_ds.rename(columns = {'Crash Type':'CrashType', 'Number Fatalities':'NumberFatalities','Bus \nInvolvement':'BusInvolvement',

                         'Heavy Rigid Truck Involvement':'HeavyRigidTruckInvolvement', 'Articulated Truck Involvement':'ArticulatedTruckInvolvement',

                         'Speed Limit':'SpeedLimit','Christmas Period':'ChristmasPeriod','Easter Period':'EasterPeriod','Day of week':'DayOfWeek', 'Time of Day':'TimeOfDay'}, inplace = True)



# filteredData = new_ds.Year == 2019

new_ds['State'].value_counts()
data = {'State': ["NSW", "VIC", "QLD","WA","SA","TAS","NT","ACT"], 

        'tot # of Accidents':[14260,10161,9081,5459,3918,1336,1383,424],

        'tot_population':[8117976,6629870,5115451,2630557,1756494,535500,245562,428060]}

new_ds = pd.DataFrame.from_dict(data)

print(new_ds)



new_ds['per 10000 ppl'] = ((new_ds['tot # of Accidents'] / new_ds['tot_population']) * 10000).round()

print(new_ds)



new_ds.plot(kind='bar', x='State',y='per 10000 ppl', color='pink', ylim=(0,60))

plt.title("Each State Fatal Crashes per 10,000 People", y=1.05);

plt.ylabel("# of Fatal Crashes per 10,000 People", labelpad=14)
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

ds = pd.read_csv('../input/australia-road-fatal-crashes/ardd_fatal_crashes.csv', index_col=['Crash ID'])

new_ds = pd.DataFrame(ds)  

new_ds.rename(columns = {'Crash Type':'CrashType', 'Number Fatalities':'NumberFatalities','Bus \nInvolvement':'BusInvolvement',

                         'Heavy Rigid Truck Involvement':'HeavyRigidTruckInvolvement', 'Articulated Truck Involvement':'ArticulatedTruckInvolvement',

                         'Speed Limit':'SpeedLimit','Christmas Period':'ChristmasPeriod','Easter Period':'EasterPeriod','Day of week':'DayOfWeek', 'Time of Day':'TimeOfDay'}, inplace = True)



new_ds['SpeedLimit'].value_counts()
data = {'SpeedLimit': ["<=40","41-50","51-60","61-70","71-80","81-90","91-100","101-110",">110"], 

        'tot # of Accidents':[373,2598,12698,2493,5351,971,15194,4962,87]}

new_ds = pd.DataFrame.from_dict(data)

print(new_ds)



new_ds['Crash %'] = ((new_ds['tot # of Accidents'] / new_ds['tot # of Accidents'].sum()) * 100).round(2)

print(new_ds)



new_ds.plot(kind='bar', x='SpeedLimit',y='Crash %', color='brown', ylim=(0,40))

plt.title("% of Fatal Crashes based on Speed Limit", y=1.05);

plt.ylabel("% of Fatal Crashes", labelpad=14)
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

ds = pd.read_csv('../input/australia-road-fatal-crashes/ardd_fatal_crashes.csv', index_col=['Crash ID'])

new_ds = pd.DataFrame(ds)  

new_ds.rename(columns = {'Crash Type':'CrashType', 'Number Fatalities':'NumberFatalities','Bus \nInvolvement':'BusInvolvement',

                         'Heavy Rigid Truck Involvement':'HeavyRigidTruckInvolvement', 'Articulated Truck Involvement':'ArticulatedTruckInvolvement',

                         'Speed Limit':'SpeedLimit','Christmas Period':'ChristmasPeriod','Easter Period':'EasterPeriod','Day of week':'DayOfWeek', 'Time of Day':'TimeOfDay'}, inplace = True)



filteredData2 = new_ds.ChristmasPeriod == 'Yes'

(new_ds[filteredData2])['Year'].value_counts().sort_index().plot.bar() 

# new_ds['SpeedLimit'].value_counts()
filteredData2 = new_ds.ChristmasPeriod == 'Yes'

(new_ds[filteredData2])['Year'].value_counts()
data = {'Year': ["1989", "1990", "1991","1992","1993","1994","1995","1996","1997","1998","1999","2000",

                "2001", "2002", "2003", "2004","2005","2006", "2007","2008","2009","2010",

                "2011","2012","2013","2014","2015","2016","2017","2018","2019"], 

        'tot # of Accidents':[64,61,56,39,60,56,39,51,59,63,57,54,

                              43,43,53,41,53,50,32,38,37,27,

                              46,46,32,22,35,34,40,43,24]}



new_ds = pd.DataFrame.from_dict(data)

print(new_ds)



new_ds['Crash %'] = ((new_ds['tot # of Accidents'] / new_ds['tot # of Accidents'].sum()) * 100).round(2)

print(new_ds)
data = {'Year': ["1989", "1990", "1991","1992","1993","1994","1995","1996","1997","1998","1999","2000",

                "2001", "2002", "2003", "2004","2005","2006", "2007","2008","2009","2010",

                "2011","2012","2013","2014","2015","2016","2017","2018","2019"], 

        'tot # of Accidents':[64,61,56,39,60,56,39,51,59,63,57,54,

                              43,43,53,41,53,50,32,38,37,27,

                              46,46,32,22,35,34,40,43,24]}



# new_ds = pd.DataFrame.from_dict(data)

# print(new_ds)



new_ds['Crash %'] = ((new_ds['tot # of Accidents'] / new_ds['tot # of Accidents'].sum()) * 100).round(2)

print(new_ds)



new_ds.plot(kind='bar', x='Year',y='Crash %', color='purple', ylim=(0,5))

plt.title("% of Fatal Crashes based in Christmas Period", y=1.05);

plt.ylabel("% of Fatal Crashes", labelpad=14)
data = {'Year': ["2006", "2007","2008","2009","2010",

                "2011","2012","2013","2014","2015","2016","2017","2018","2019"], 

        'tot # of Accidents':[50,32,38,37,27,

                              46,46,32,22,35,34,40,43,24]}



new_ds = pd.DataFrame.from_dict(data)

# print(new_ds)



new_ds['Crash %'] = ((new_ds['tot # of Accidents'] / new_ds['tot # of Accidents'].sum()) * 100).round(2)

print(new_ds)



new_ds.plot(kind='bar', x='Year',y='Crash %', color='purple', ylim=(0,12))

plt.title("% of Fatal Crashes in Christmas Period", y=1.05);

plt.ylabel("% of Fatal Crashes", labelpad=14)
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

ds = pd.read_csv('../input/australia-road-fatal-crashes/ardd_fatal_crashes.csv', index_col=['Crash ID'])

new_ds = pd.DataFrame(ds)  

new_ds.rename(columns = {'Crash Type':'CrashType', 'Number Fatalities':'NumberFatalities','Bus \nInvolvement':'BusInvolvement',

                         'Heavy Rigid Truck Involvement':'HeavyRigidTruckInvolvement', 'Articulated Truck Involvement':'ArticulatedTruckInvolvement',

                         'Speed Limit':'SpeedLimit','Christmas Period':'ChristmasPeriod','Easter Period':'EasterPeriod','Day of week':'DayOfWeek', 'Time of Day':'TimeOfDay'}, inplace = True)



filteredData3 = new_ds.EasterPeriod == 'Yes'

(new_ds[filteredData3])['Year'].value_counts().sort_index().plot.bar() 
(new_ds[filteredData3])['Year'].value_counts()
data = {'Year': ["2006", "2007","2008","2009","2010",

                "2011","2012","2013","2014","2015","2016","2017","2018","2019"], 

        'tot # of Accidents':[18,25,14,18,12,

                              20,10,19,12,19,8,11,14,18]}



new_ds = pd.DataFrame.from_dict(data)

# print(new_ds)



new_ds['Crash %'] = ((new_ds['tot # of Accidents'] / new_ds['tot # of Accidents'].sum()) * 100).round(2)

print(new_ds)



new_ds.plot(kind='bar', x='Year',y='Crash %', color='orange', ylim=(0,12))

plt.title("% of Fatal Crashes in Easter Period", y=1.05);

plt.ylabel("% of Fatal Crashes", labelpad=14)
import matplotlib.pyplot as plt



Christmas = [9.88,6.32,7.51,7.31,5.34,9.09,9.09,6.32,4.35,6.92, 6.72,7.91,8.50,4.74]

Easter = [8.26,11.47,6.42,8.26,5.50,9.17 ,4.59 ,8.72,5.50,8.72,3.67,5.05,6.42,8.26]

index = ["2006", "2007","2008","2009","2010","2011","2012","2013","2014","2015","2016","2017","2018","2019"]

df = pd.DataFrame({'Christmas Period': Christmas,

                   'Easter Period': Easter}, index=index)

ax = df.plot.bar(rot=90)





# new_ds.plot(kind='bar', x='Year',y='Crash %', color='orange', ylim=(0,12))

plt.title("Fatal crashes Comparison of Christmas & Easter Period", y=1.05);

plt.ylabel("% of Fatal Crashes", labelpad=14)

plt.xlabel("Year", labelpad=14)



# speed = [0.1, 17.5, 40, 48, 52, 69, 88]

# lifespan = [2, 8, 70, 1.5, 25, 12, 28]

# index = ['snail', 'pig', 'elephant',

#          'rabbit', 'giraffe', 'coyote', 'horse']

# df = pd.DataFrame({'Christmas Period': Christmas,

#                    'Easter Period': Easter}, index=index)

# ax = df.plot.bar(rot=0)




