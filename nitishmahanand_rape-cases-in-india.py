import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tabulate import tabulate

%matplotlib inline
rape_victims = pd.read_csv('../input/rape-victims/Victims_of_rape.csv')
rape_victims = rape_victims[rape_victims['Subgroup'] != 'Total Rape Victims']
rape_victims[rape_victims['Victims_of_Rape_Total'] != rape_victims['Rape_Cases_Reported']]
rape_victims['Unreported_Cases'] = rape_victims['Victims_of_Rape_Total'] - rape_victims['Rape_Cases_Reported']
rape_victims[rape_victims['Unreported_Cases'] > 0].head(5)
state = rape_victims.groupby('Area_Name').sum()
area = rape_victims['Area_Name']
old = rape_victims['Victims_Above_50_Yrs']
teen1 = rape_victims['Victims_Between_10-14_Yrs']
teen2 = rape_victims['Victims_Between_14-18_Yrs']
teen3 = rape_victims['Victims_Upto_10_Yrs']
below18n = teen1+teen2+teen3

fig = plt.figure(figsize = (10,10))
plt.grid(color='black', linestyle='dashed', linewidth=1)
plt.barh(area,old)
plt.title('State Rape Victims who were above 50 age')
plt.xlabel('No of cases')
plt.ylabel('State')
plt.show()
fig = plt.figure(figsize = (10,10))
plt.grid(color='black', linestyle='dashed', linewidth=1)
plt.barh(area,below18n)
plt.title('State Rape Victims who were below 18 age')
plt.xlabel('No of cases')
plt.ylabel('State')
plt.show()
unreported_victims_by_state = rape_victims.groupby('Area_Name').sum()
unreported_victims_by_state.drop('Year', axis = 1, inplace = True)

plt.subplots(figsize = (15, 6))
ct = unreported_victims_by_state[unreported_victims_by_state['Unreported_Cases'] 
                                 > 0]['Unreported_Cases'].sort_values(ascending = False)

ax = ct.plot.bar()
ax.set_xlabel('Area Name')
ax.set_ylabel('Total Number of Unreported Rape Victims from 2001 to 2010')
ax.set_title('Statewise total Unreported Rape Victims throughout 2001 to 2010')
plt.show()
rape_victims_by_state = rape_victims.groupby('Area_Name').sum()
rape_victims_by_state.drop('Year', axis = 1, inplace = True)
print('Total Rape Victims = ' ,rape_victims_by_state['Rape_Cases_Reported'].sum())
rape_victims_by_state.sort_values(by = 'Rape_Cases_Reported', ascending = False).head()
rape_victims_heatmap = rape_victims_by_state.drop(['Rape_Cases_Reported', 
                                                   'Victims_of_Rape_Total', 
                                                   'Unreported_Cases'], axis = 1)
plt.subplots(figsize = (10, 10))
ax = sns.heatmap(rape_victims_heatmap, cmap="Blues")
ax.set_xlabel('Age Group')
ax.set_ylabel('State Name')
ax.set_title('Statewise Victims of Rape Cases based on Age Group')
plt.show()
plt.subplots(figsize = (15, 6))
ct = rape_victims_by_state['Rape_Cases_Reported'].sort_values(ascending = False)

ax = ct.plot.bar()
ax.set_xlabel('Area Name')
ax.set_ylabel('Total Number of Reported Rape Victims from 2001 to 2010')
ax.set_title('Statewise total Reported Rape Victims throught the Years 2001 to 2010')
plt.show()
print(ct)
delhi = rape_victims[rape_victims['Area_Name'] == 'Delhi']

delhi.head()
delhiv = delhi.groupby('Year').sum()

plt.subplots(figsize = (15, 6))
ax = delhiv['Rape_Cases_Reported'].plot()
ax.xaxis.set_ticks(np.arange(2001, 2011, 1))
ax.set(xlabel = 'Year', ylabel = 'Total Number of Reported Rape Victims', 
       title = 'Number of Rape Victims throught the years 2001 to 2010 of Delhi')
plt.show()
mp_rape_victims = rape_victims[rape_victims['Area_Name'] == 'Madhya Pradesh']

mp_rape_victims.head(2)
mp_rape_victims_by_year = mp_rape_victims.groupby('Year').sum()

plt.subplots(figsize = (15, 6))
ax = mp_rape_victims_by_year['Rape_Cases_Reported'].plot()
ax.xaxis.set_ticks(np.arange(2001, 2011, 1))
ax.set(xlabel = 'Year', ylabel = 'Total Number of Reported Rape Victims', 
       title = 'Number of Rape Victims throught the years 2001 to 2010 of Madhya Pradesh')
plt.show()
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

wb_rape_victims_by_year = wb_rape_victims.groupby('Year').sum()

plt.subplots(figsize = (15, 6))
ax = wb_rape_victims_by_year['Rape_Cases_Reported'].plot()
ax.xaxis.set_ticks(np.arange(2001, 2011, 1))
ax.set(xlabel = 'Year', ylabel = 'Total Number of Reported Rape Victims',
       title = 'Number of Rape Victims throught the years 2001 to 2010 of West Bengal')
plt.show()
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
rape_victims
data = pd.read_csv("../input/rape-cases/women_crimes.csv")
data
data_state = data.groupby('STATE/UT').sum()
data_state.drop('Year', axis = 1, inplace = True)

plt.subplots(figsize = (15, 6))
ct = data_state[data_state['Rape'] 
                                 > 0]['Rape'].sort_values(ascending = False)

ax = ct.plot.bar()
ax.set_xlabel('STATE/UT')
ax.set_ylabel('Total Number of  Rape  from 2001 to 2010')
ax.set_title('Statewise total Rape throughout 2001 to 2010')
plt.show()
data_state = data.groupby('STATE/UT').sum()
data_state.drop('Year', axis = 1, inplace = True)
print('Total Rape Victims = ' ,data_state['Rape'].sum())
data_state.sort_values(by = 'Rape', ascending = False).head()
data_state1 = data.groupby('STATE/UT').sum()
data_state1.drop('Year', axis = 1, inplace = True)
print('Total Kidnapping and Abduction = ' ,data_state1['Kidnapping and Abduction'].sum())
data_state4 = data.groupby('STATE/UT').sum()
data_state4.drop('Year', axis = 1, inplace = True)
print('Total cases on women with intent to outrage her modesty = ' ,data_state4['Assault on women with intent to outrage her modesty'].sum())
data_state5 = data.groupby('STATE/UT').sum()
data_state5.drop('Year', axis = 1, inplace = True)
print('Total cases on insult to modesty of Women = ' ,data_state5['Insult to modesty of Women'].sum())
data_state6 = data.groupby('STATE/UT').sum()
data_state6.drop('Year', axis = 1, inplace = True)
print('Total cases on cruelty by husband/relatives = ' ,data_state6['Cruelty by Husband or his Relatives'].sum())
data_state2 = data.groupby('STATE/UT').sum()
data_state2.drop('Year', axis = 1, inplace = True)
print('Total Importation of Girls = ' ,data_state2['Importation of Girls'].sum())
data_state3 = data.groupby('STATE/UT').sum()
data_state3.drop('Year', axis = 1, inplace = True)
print('Total Deaths due to dowry = ' ,data_state3['Dowry Deaths'].sum())
cases = ['Kidnapping-Abduction','Dowry Deaths','Assault on women with intent to outrage her modesty','Insult to modesty of Women','Cruelty by Husband or his Relatives','Importation of Girls']
deaths = [746198,215480,1212258,282756,2244888,1872]

plt.barh(cases,deaths)
plt.grid()
plt.xlabel("TOTAL CASES ")
plt.ylabel("CRIME AGAINST WOMENs")
plt.show()
plt.subplots(figsize = (15, 6))
data_state = data.groupby('STATE/UT').sum()
rapelist = data_state['Rape'].sort_values(ascending = False)
#print(ct)
ax = rapelist.plot.bar()
#ax = sns.barplot(x = rape_victims_by_state.index, y = rape_victims_by_state['Rape_Cases_Reported'])
ax.set_xlabel('State/Union Territory')
ax.set_ylabel('Total Number of Rape happened from 2001 to 2010')
ax.set_title('Statewise total Rape throught the Years 2001 to 2010')
plt.show()

print(rapelist.head(10))
data_state = data.groupby('STATE/UT').sum()
dowry = data_state['Dowry Deaths'].sort_values(ascending = False)
print(dowry.head(10).unique)
modesty = data_state['Insult to modesty of Women'].sort_values(ascending = False)
print(modesty.head(10).unique)
kidnap = data_state['Kidnapping and Abduction'].sort_values(ascending = False)
print(kidnap.head(10).unique)
assault = data_state['Assault on women with intent to outrage her modesty'].sort_values(ascending = False)
print(assault.head(10).unique)
cruelty = data_state['Cruelty by Husband or his Relatives'].sort_values(ascending = False)
print(cruelty.head(10).unique)
importation = data_state['Importation of Girls'].sort_values(ascending = False)
print(importation.head(10).unique)
'''------------------------------------------MOST IN DIFFERENT CASES---------------------------------------------------
        RAPE CASES -----------------------------------------: 1. MADHYA PRADESH 2. WEST BENGAL     3. UTTAR PRADESH
                        DOWRY DEATHS ---------------------------------------: 1. UTTAR PRADESH  2. BIHAR           3. MADHYA PRADESH
                                                INSULT TO MODESTY OF WOMEN -------------------------: 1. ANDHRA PRADESH 2. UTTAR PRADESH   3. MAHARASHTRA 
                                                                                    ASSAULT ON WOMEN WITH INTENT TO OUTRAGE HER MODESTY : 1. MADHYA PRADESH 2. ANDHRA PRADESH  3. MAHARASHTRA
                                                                CRUELTY BY HUSBAND/RELATIVES -----------------------: 1. WEST BENGAL    2. ANDHRA PRADESH  3. RAJASTHAN
                                        IMPORTATION OF GIRLS -------------------------------: 1. BIHAR          2. JHARKHAND       3. WEST BENGAL'''
