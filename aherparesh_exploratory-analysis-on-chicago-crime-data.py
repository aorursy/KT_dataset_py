import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas import read_csv
crimes = read_csv('../input/Chicago_Crimes_2012_to_2017.csv',error_bad_lines=False)
crimes.head()
crimes = crimes.iloc[:, 3: ]
crimes.head()
crimes.index = pd.to_datetime(crimes.index)
crimes.shape
s = crimes[['Primary Type']]
s.head()
crime_count.shape
crimes.Date = pd.to_datetime(crimes.Date, format = '%m/%d/%Y %I:%M:%S %p')
crimes.index = pd.DatetimeIndex(crimes.Date)
crimes['Primary Type'] = pd.Categorical(crimes['Primary Type'])
crimes['Description'] = pd.Categorical(crimes['Description'])
crimes['Location Description'] = pd.Categorical(crimes['Location Description'])
#here no of crimes per year ,per  months and per day has been calculated on basis of chicago data from 2012 to 2017.
#out has been plotted as below

plt.figure(figsize = (8,5))
crimes.groupby([crimes.index.year]).size().plot.bar()
plt.title('Crime Over the Years')
plt.xlabel('Year')
plt.ylabel('Number of Crimes Acts')
plt.show()
plt.figure(figsize = (8,5))

crimes.groupby([crimes.index.month]).size().plot.bar()
plt.title('Crime Per Month')
plt.xlabel('Month')
plt.ylabel('Number of Crimes Acts')
plt.show()

crimes.groupby([crimes.index.day]).size().plot.bar()
plt.title('Crime Per Day Of Month')
plt.xlabel('Day')
plt.ylabel('Number of Crimes Acts')
plt.show()

crimes.groupby([crimes.index.hour]).size().plot.bar()
plt.title('Crime Per Hour')
plt.xlabel('Hour')
plt.ylabel('Number of Crimes Acts')
plt.show()
#here, i've plotted the types of crime in primary type such as, arson,assault,battery,burglary etc..
#main intention to plot data of primary type is to find the frequency of crime over 5 year.

crimes_date = crimes.pivot_table('ID',aggfunc = np.size, columns = 'Primary Type',index = crimes.index.date, fill_value = 0)
crimes_date.index = pd.DatetimeIndex(crimes_date.index)
Plot = crimes_date.plot(figsize = (20,30), subplots = True, layout = (6,6),
                                  sharex = False, sharey = False)
plt.show()
crime_count = pd.DataFrame(s.groupby('Primary Type').size().sort_values(ascending=False).rename('counts').reset_index())

crime_count.head()
#Attributes of primary type is plotted in the form of bar to find and analyse highest type crime activity done form 2012 to 2017

import seaborn as sns
import matplotlib.pyplot as plt
sns.set(style="whitegrid")

f, ax = plt.subplots(figsize=(10, 15))
sns.set_color_codes("pastel")
sns.barplot(y="Primary Type", x="counts", data=crime_count.iloc[:10, :],label="Total", color="grey")

ax.legend(ncol=2, loc="lower right", frameon=True)
ax.set(ylabel="Type",xlabel="Crimes")
sns.despine(left=True, bottom=True)
plt.show()
b = pd.DataFrame(crimes ,columns=['Primary Type','Arrest'])
crime_count1 = pd.DataFrame(b.groupby(['Primary Type','Arrest']).size().sort_values(ascending=False).rename('counts').reset_index())
crime_count1
#here,i've plotted the entities of primary type  and arrest section in form of bar plot to find the result on criminal activity
#in the graph red color implies to "Arrest" and other color implies  to "NOt arrested" which help us to show how effective fbi is working
import seaborn as sns
import matplotlib.pyplot as plt


f, ax = plt.subplots(figsize=(10,20))

# Plot the total crashes
sns.set_color_codes("pastel")
sns.barplot( y="Primary Type",x="counts" , data=crime_count1.iloc[:20, :], hue='Arrest', color='red')

ax.legend(ncol=2, loc="lower right", frameon=True)
ax.set(ylabel="Type",xlabel="Crimes")
sns.despine(left=True, bottom=True)

#highest rate of crime is done on street,residence,apartment and sidewalk

crimes_2012 = crimes.loc['2012']
crimes_2013 = crimes.loc['2013']
crimes_2014 = crimes.loc['2014']
crimes_2015 = crimes.loc['2015']
crimes_2016 = crimes.loc['2016']
crimes_2017 = crimes.loc['2017']

arrest_yearly = crimes[crimes['Arrest'] == True]['Arrest']
#Due to lesser of number of arrest in previous graph, here i have plotted arrest on basis of year,month,weekly and day

plt.subplot()

# yearly arrest
arrest_yearly.resample('A').sum().plot()
plt.title('Yearly arrests')
plt.show()

# Monthly arrest
arrest_yearly.resample('M').sum().plot()
plt.title('Monthly arrests')
plt.show()

# Weekly arrest
arrest_yearly.resample('W').sum().plot()
plt.title('Weekly arrests')
plt.show()

# daily arrest
arrest_yearly.resample('D').sum().plot()
plt.title('Daily arrests')
plt.show()
plt.show()
