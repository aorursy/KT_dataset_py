import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


#separate dataframe into cities, states and countries
#Disclaimer: OpenTable did not provide data on all cities, states and countries 
pd.read_csv('/kaggle/input/opentable-reservation-data/YoY_Seated_Diner_Data.csv')
country = resv[resv.Type=='country']
states= resv[resv.Type == 'state']
city = resv[resv.Type == 'city']
states.describe() #row indices are states, columns are dates of OpenTable data
resv.Type.value_counts() #Breakdown of number of cities, states, and countries OpenTable Collected data on
plt.figure(figsize = (15,10))
plt.plot(np.transpose(states[states.Name=='Maryland'])[2:],'bo:',np.array([0 for zero in range(len(np.transpose(states[states.Name =='Maryland'])[2:]))]),
         'k--',) #Using the transpose so reservation dates become rows
plt.xticks(states[states.Name=='Maryland'].columns[2::7],fontsize = 12) #Separating the xticks by week to make less cluttered
plt.yticks(fontsize = 12)
plt.title("Maryland's Year-over-Year Change in Reservations",fontsize = 30,fontweight = 'bold')
plt.xlabel('Month/day in 2020', fontsize = 18)
plt.ylabel('Change in Reservations YOY (percent)',fontsize = 18)
plt.figtext(0.5, 0.04, 'Source: opentable.com/state-of-industry', wrap=True, horizontalalignment='center', fontsize=12)
plt.show()
usa = country[country.Name == 'United States']
uk = country[country.Name == 'United Kingdom']
ger = country[country.Name =='Germany']
can = country[country.Name =='Canada']
plt.figure(figsize = (15,10))
plt.plot(np.transpose(usa)[2:],'bo:',np.transpose(uk)[2:],'go:',np.transpose(ger)[2:],'yo:',np.transpose(can)[2:],'ro:',
         np.array([0 for zero in range(len(np.transpose(country[country.Name =='United Kingdom'])[2:]))]),
         'k--',alpha = 0.6)
plt.xticks(usa.columns[2::7],fontsize = 12)
plt.yticks(fontsize = 12)
plt.title('Country Changes in Reservations Year-over-Year',fontsize = 30, fontweight = 'bold')
plt.legend(('USA','United Kingdom','Germany','Canada'),loc = 'upper left',fontsize = 12)
plt.xlabel('Month/Day in Year 2020',fontsize = 18)
plt.ylabel('Percent change from pervious year',fontsize = 18)
plt.figtext(0.5, 0.04, 'Source: opentable.com/state-of-industry', wrap=True, horizontalalignment='center', fontsize=12)
plt.show()
hou = np.transpose(city[city.Name == 'Houston'])[2:]
bal = np.transpose(city[city.Name == 'Baltimore'])[2:]
nyc = np.transpose(city[city.Name == 'New York'])[2:]
sea = np.transpose(city[city.Name == 'Seattle'])[2:]
mia = np.transpose(city[city.Name == 'Miami'])[2:]
la = np.transpose(city[city.Name == 'Los Angeles'])[2:]
no = np.transpose(city[city.Name == 'New Orleans'])[2:]
plt.figure(figsize = (15,10))
plt.plot(hou,'bo:',mia,'ro:',nyc,'go:',sea,'yo:',no,'ko:',
         la,'mo:',
         np.array([0 for zero in range(len(np.transpose(city[city.Name =='Baltimore'])[2:]))]),
         'k--',alpha = 0.52)
plt.xticks(usa.columns[2::7],fontsize = 12)
plt.xticks(usa.columns[2::7],fontsize = 12)
plt.yticks(fontsize = 12)
plt.title('City Changes in Reservations Year-over-Year',fontsize = 30, fontweight = 'bold')
plt.legend(('Houston','Miami','New York City','Seattle','New Orleans','Los Angeles'),loc = 'upper right')
plt.xlabel('Month/Day in Year 2020',fontsize = 18)
plt.ylabel('Percent change from pervious year',fontsize = 18)
plt.figtext(0.5, 0.04, 'Source: opentable.com/state-of-industry', wrap=True, horizontalalignment='center', fontsize=12)
plt.show()
tx = np.transpose(states[states.Name == 'Texas'])[2:]
pa = np.transpose(states[states.Name == 'Pennsylvania'])[2:]
ny = np.transpose(states[states.Name == 'New York'])[2:]
nj = np.transpose(states[states.Name == 'New Jersey'])[2:]
fl = np.transpose(states[states.Name == 'Florida'])[2:]
ca = np.transpose(states[states.Name == 'California'])[2:]
mi = np.transpose(states[states.Name == 'Michigan'])[2:]
plt.figure(figsize = (15,10))
plt.plot(tx,'bo:',pa,'ro:',ny,'go:',nj,'yo:',fl,'ko:',
         ca,'mo:', mi,'co:',
         np.array([0 for zero in range(len(np.transpose(city[city.Name =='Baltimore'])[2:]))]),
         'k--',alpha = 0.52)
plt.xticks(usa.columns[2::7],fontsize = 12)
plt.xticks(usa.columns[2::7],fontsize = 12)
plt.yticks(fontsize = 12)
plt.title('State Changes in Reservations Year-over-Year',fontsize = 30, fontweight = 'bold')
plt.legend(('Texas','Pennsylvania','New York','New Jersey','Florida','California', 'Michigan'),loc = 'upper right')
plt.xlabel('Month/Day in Year 2020',fontsize = 18)
plt.ylabel('Percent change from pervious year',fontsize = 18)
plt.figtext(0.5, 0.04, 'Source: opentable.com/state-of-industry', wrap=True, horizontalalignment='center', fontsize=12)
plt.show()
def resv_plot(land,name):  #land is either a city, state or country. Name is the city/state/country name to extract data from
    geo = resv[resv.Type == land]
    if land not in ['city','state','country']:
        print('land argument must be either: city, state or country and formatted as a string')
    else:
        n = geo[geo.Name == name]
        
        try:
            plt.figure(figsize = (15,10))
            plt.plot(np.transpose(n)[2:],'o:' ,np.array([0 for zero in range(len(np.transpose(n)[2:]))]),'k:')
            plt.xticks(geo.columns[2::7],fontsize = 12)
            plt.yticks(fontsize = 12)
            plt.title('{} Changes in Reservations Year-over-Year'.format(name),fontsize = 30, fontweight = 'bold')
            plt.xlabel('Month/Day in Year 2020',fontsize = 18)
            plt.ylabel('Percent change from pervious year',fontsize = 18)
            plt.figtext(0.5, 0.04, 'Source: opentable.com/state-of-industry', wrap=True, horizontalalignment='center', fontsize=12)
            plt.show()

        except ZeroDivisionError:
            print('{} is not in the OpenTable dataset'.format(land.capitalize()))
        
resv_plot('city','San Francisco')
