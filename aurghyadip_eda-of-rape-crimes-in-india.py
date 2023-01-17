#import the libraries

import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

from tabulate import tabulate



# configure matplotlib to output inline

%matplotlib inline
# victims of rape DataFrame

rape_victims = pd.read_csv('../input/20_Victims_of_rape.csv')



# let's see how the data is structured

rape_victims.head()
# let's clean the data a bit

# we don't need the Total Rape Victims subgroup (we can do that in Pandas), let's remove it

rape_victims = rape_victims[rape_victims['Subgroup'] != 'Total Rape Victims']



# let's check if the all the rape cases are reported

rape_victims[rape_victims['Victims_of_Rape_Total'] != rape_victims['Rape_Cases_Reported']].head()
rape_victims['Unreported_Cases'] = rape_victims['Victims_of_Rape_Total'] - rape_victims['Rape_Cases_Reported']



# let's taka a look at the new dataframe

rape_victims[rape_victims['Unreported_Cases'] > 0].head()
# let's plot the unreported rape cases sorted by states throughout 2001 to 2010

unreported_victims_by_state = rape_victims.groupby('Area_Name').sum()

unreported_victims_by_state.drop('Year', axis = 1, inplace = True)



# let's finally plot it

plt.subplots(figsize = (15, 6))

ct = unreported_victims_by_state[unreported_victims_by_state['Unreported_Cases'] 

                                 > 0]['Unreported_Cases'].sort_values(ascending = False)

#print(ct)

ax = ct.plot.bar()

ax.set_xlabel('Area Name')

ax.set_ylabel('Total Number of Unreported Rape Victims from 2001 to 2010')

ax.set_title('Statewise total Unreported Rape Victims throughout 2001 to 2010')

plt.show()
# let's take some general data and plot some simple charts

rape_victims_by_state = rape_victims.groupby('Area_Name').sum()

rape_victims_by_state.drop('Year', axis = 1, inplace = True)

print('Total Rape Victims = ' ,rape_victims_by_state['Rape_Cases_Reported'].sum())

rape_victims_by_state.sort_values(by = 'Rape_Cases_Reported', ascending = False).head()
# let's make a heatmap variable

rape_victims_heatmap = rape_victims_by_state.drop(['Rape_Cases_Reported', 

                                                   'Victims_of_Rape_Total', 

                                                   'Unreported_Cases'], axis = 1)

plt.subplots(figsize = (10, 10))

ax = sns.heatmap(rape_victims_heatmap, cmap="Blues")

ax.set_xlabel('Age Group')

ax.set_ylabel('State Name')

ax.set_title('Statewise Victims of Rape Cases based on Age Group')

plt.show()
# let's first plot only the total number of rape cases reported in each state

plt.subplots(figsize = (15, 6))

ct = rape_victims_by_state['Rape_Cases_Reported'].sort_values(ascending = False)

#print(ct)

ax = ct.plot.bar()

#ax = sns.barplot(x = rape_victims_by_state.index, y = rape_victims_by_state['Rape_Cases_Reported'])

ax.set_xlabel('Area Name')

ax.set_ylabel('Total Number of Reported Rape Victims from 2001 to 2010')

ax.set_title('Statewise total Reported Rape Victims throught the Years 2001 to 2010')

plt.show()

print(ct)
mp_rape_victims = rape_victims[rape_victims['Area_Name'] == 'Madhya Pradesh']



# let's have a look in the data

mp_rape_victims.head()
# Let's have a look at yearly distribution of number of rape victims in Madhya Pradesh

mp_rape_victims_by_year = mp_rape_victims.groupby('Year').sum()



# plotting the data

plt.subplots(figsize = (15, 6))

ax = mp_rape_victims_by_year['Rape_Cases_Reported'].plot()

ax.xaxis.set_ticks(np.arange(2001, 2011, 1))

ax.set(xlabel = 'Year', ylabel = 'Total Number of Reported Rape Victims', 

       title = 'Number of Rape Victims throught the years 2001 to 2010 of Madhya Pradesh')

plt.show()
# let's first see the mp_rape_victims dataframe

#mp_rape_victims.head()



# plot the dataframe

mp_incest_rape_cases = mp_rape_victims[mp_rape_victims['Subgroup'] == 'Victims of Incest Rape']

plt.subplots(figsize = (15,6))

ct = mp_incest_rape_cases.groupby('Year').sum()

ax = ct['Rape_Cases_Reported'].plot.bar()

for p in ax.patches:

    ax.annotate(format(p.get_height()), (p.get_x()+0.1, p.get_height()+2),fontsize=12)

ax.set_xlabel('Year')

ax.set_ylabel('Total Number of Incest Rape Victims from 2001 to 2010')

ax.set_title('Total Reported Incest Rape Victims of Madhya Pradesh throught the Years 2001 to 2010')

plt.show()
wb_rape_victims = rape_victims[rape_victims['Area_Name'] == 'West Bengal']



# Let's have a look at yearly distribution of number of rape victims in Madhya Pradesh

wb_rape_victims_by_year = wb_rape_victims.groupby('Year').sum()



# plotting the data

plt.subplots(figsize = (15, 6))

ax = wb_rape_victims_by_year['Rape_Cases_Reported'].plot()

ax.xaxis.set_ticks(np.arange(2001, 2011, 1))

ax.set(xlabel = 'Year', ylabel = 'Total Number of Reported Rape Victims',

       title = 'Number of Rape Victims throught the years 2001 to 2010 of West Bengal')

plt.show()
# let's calculate the percentage increase of number of rapes in West Bengal and compare 

# it with Madhya Pradesh

plt.subplots(figsize = (15, 6))

ax = (wb_rape_victims_by_year['Rape_Cases_Reported'].pct_change() * 100).plot(legend = True, 

                                                                              label = 'West Bengal')

(mp_rape_victims_by_year['Rape_Cases_Reported'].pct_change() * 100).plot(ax = ax, legend = True, 

                                                                         label = 'Madhya Pradesh')

ax.set(xlabel = 'Year', ylabel = 'Percent', 

       title = 'Yearly increase and decrease in number of rapes in West Bengal and Madhya Pradesh')

ax.axhline(0, color = 'black')

ax.axvline(2002, color = 'black')

plt.show()

print('Overall Increase in number of rapes in West Bengal =', 

      '{0:.2f}'.format(((wb_rape_victims_by_year.iloc[9]['Rape_Cases_Reported'] 

                         - wb_rape_victims_by_year.iloc[0]['Rape_Cases_Reported'])

                        /wb_rape_victims_by_year.iloc[9]['Rape_Cases_Reported']) * 100), 'Percent')

print('Overall Increase in number of rapes in Madhya Pradesh =', 

      '{0:.2f}'.format(((mp_rape_victims_by_year.iloc[9]['Rape_Cases_Reported'] 

                         - mp_rape_victims_by_year.iloc[0]['Rape_Cases_Reported'])

                        /wb_rape_victims_by_year.iloc[9]['Rape_Cases_Reported']) * 100), 'Percent')
# incest rape cases in Bengal

wb_incest_rape_cases = wb_rape_victims[wb_rape_victims['Subgroup'] == 'Victims of Incest Rape']

plt.subplots(figsize = (15,6))

ct = wb_incest_rape_cases.groupby('Year').sum()

ax = ct[ct['Rape_Cases_Reported'] > 0]['Rape_Cases_Reported'].plot.bar()

for p in ax.patches:

    ax.annotate(format(p.get_height()), (p.get_x() + .15, p.get_height()+1),fontsize=13)

ax.set_xlabel('Year')

ax.set_ylabel('Total Number of Incest Rape Victims from 2001 to 2010')

ax.set_title('Total Reported Incest Rape Victims of West Bengal throught the Years 2001 to 2010')

plt.show()