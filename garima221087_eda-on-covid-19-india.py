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





#### Descriptive stats on the data



#Individual data

india_data = pd.read_csv('/kaggle/input/covid19-in-india/covid_19_india.csv')

india_data.describe()

india_data.tail(50)

ax = plt.axes()

sns.heatmap(india_data.isnull(), cbar=False, ax = ax)

ax.set_title('Missingness in Daily cases Data')

plt.show()



#Individual data

pop_dt = pd.read_csv('/kaggle/input/covid19-in-india/population_india_census2011.csv')

pop_dt.describe()

ax = plt.axes()

sns.heatmap(pop_dt.isnull(), cbar=False, ax = ax)

ax.set_title('Missingness in Census Data')

plt.show()



#Individual data

hop_dt = pd.read_csv('/kaggle/input/covid19-in-india/HospitalBedsIndia.csv')

hop_dt.describe()

ax = plt.axes()

sns.heatmap(hop_dt.isnull(), cbar=False, ax = ax)

ax.set_title('Missingness in Hospital beds data Data')

plt.show()



#Individual data

indvl_dt = pd.read_csv('/kaggle/input/covid19-in-india/IndividualDetails.csv')

indvl_dt.describe()

ax = plt.axes()

sns.heatmap(indvl_dt.isnull(), cbar=False, ax = ax)

ax.set_title('Missingness in Individual Data')

plt.show()

percent_missing = indvl_dt.isnull().sum() * 100 / len(indvl_dt)

missing_value_df = pd.DataFrame({'column_name': indvl_dt.columns,

                                 'percent_missing': percent_missing})

missing_value_df.sort_values('percent_missing', inplace=True, ascending=False)

missing_value_df[missing_value_df.columns[1:2]]



############################ Confirmed Cases #################################



india_data = pd.read_csv('/kaggle/input/covid19-in-india/covid_19_india.csv')

#india_data

########## Time-series of cases

india_data.Date=pd.to_datetime(india_data.Date,dayfirst=True)

#ndia_data['Date']=pd.to_datetime(india_data['Date'],format = '%d/%m/%y')

                             

india_data['TotalCases'] = india_data['ConfirmedIndianNational']+india_data['ConfirmedForeignNational']

india_data_ts = india_data.groupby('Date')['Date', 'TotalCases', 'Deaths'].sum()

india_data_ts.sort_values('Date')

india_data.head()

india_data.describe()



# Confirmed cases trends in India



import matplotlib.pyplot as plt

plt.figure(figsize=(10,5))

plt.plot(india_data_ts.index,india_data_ts['TotalCases'], color='g',label='Confirmed Cases' )

plt.plot(india_data_ts.index,india_data_ts['Deaths'], color='k', label = 'Deaths')

plt.title('Confirmed cases in India', fontsize=16)

plt.legend(loc='upper center', shadow=True)

plt.show()





########## Statewise cases

statewise_df = india_data.pivot_table(index='State/UnionTerritory', values= ['ConfirmedIndianNational','ConfirmedForeignNational','Cured','Deaths'], aggfunc = sum)

statewise_df['TotalCases'] = statewise_df['ConfirmedIndianNational']+statewise_df['ConfirmedForeignNational']

statewise_df['PctDeaths'] = statewise_df['Deaths']/statewise_df['TotalCases']

statewise_df = statewise_df.sort_values(['TotalCases'],ascending=False)



plt.figure(figsize=(15,8))

plt.barh(statewise_df.index,statewise_df['TotalCases'])

plt.title('Confirmed Cases by States in India',fontsize=16)

plt.show()

############################ Available Hospital bed data #################################



hop_dt = pd.read_csv('/kaggle/input/covid19-in-india/HospitalBedsIndia.csv')

hop_dt.describe()

hop_dt2 = hop_dt.iloc[:-2,:] #removing last two rows as those are column sums and NAN

hop_dt2.iloc[:,7] = pd.to_numeric(hop_dt2.iloc[:,7])

hop_dt2['TotalBeds'] = hop_dt2.iloc[:,[7,9,11]].sum(axis = 1)

hop_dt2_sorted = hop_dt2.sort_values('TotalBeds',ascending = False)





plt.figure(figsize=(15,5))

plt.bar(hop_dt2_sorted['State/UT'],hop_dt2_sorted['TotalBeds'], color = 'g')

plt.xticks(hop_dt2_sorted['State/UT'], rotation=90)

plt.title('Total hospital beds by states',fontsize=16)

plt.ylabel('No. of hospital beds')

plt.show()



############################ Population and density data #################################

pop_dt = pd.read_csv('/kaggle/input/covid19-in-india/population_india_census2011.csv')

pop_dt.describe()

merged_cases_bed = pd.merge(statewise_df, hop_dt2_sorted, left_index=True, right_on='State/UT')

merged_cases_bed2 = merged_cases_bed.iloc[:,[7,0,1,2,3,4,5,15,17,20]]

merged_cases_bed_pop = pd.merge(merged_cases_bed2,pop_dt,left_on ='State/UT', right_on = 'State / Union Territory')

merged_cases_bed_pop.head()

india_fnl_dt = merged_cases_bed_pop



india_fnl_dt.rename(columns = {'NumRuralBeds_NHP18':'RuralBeds'}, inplace = True) 

india_fnl_dt.rename(columns = {'NumUrbanBeds_NHP18':'UrbanBeds'}, inplace = True) 

india_fnl_dt['PctBedtoPop'] = india_fnl_dt['TotalBeds']/india_fnl_dt['Population']

india_fnl_dt['Density'] = india_fnl_dt['Density'].str.split('/',expand=True)

india_fnl_dt['Density'] = india_fnl_dt['Density'].str.replace(',','')

india_fnl_dt['Density'] = india_fnl_dt['Density'].astype(str).astype(int)



############# Population density vs Hospital Beds

import seaborn as sns

india_fnl_dt = india_fnl_dt.sort_values('TotalBeds',ascending=False)

fig, ax1 = plt.subplots(figsize=(10,5))

color = 'tab:blue'

ax1.set_title('Hospital beds available vs Population density', fontsize=16)

ax1 = sns.barplot(x=india_fnl_dt['State/UT'], y=india_fnl_dt['TotalBeds'], palette='summer',label = 'Available Beds', ax=ax1)

ax1.tick_params(axis='y')

ax1.set_xticklabels(ax1.get_xticklabels(),rotation=90)

ax1.legend(loc = 'upper left', shadow=True)



ax2 = ax1.twinx()

color = 'tab:red'

ax2 = sns.lineplot(x=india_fnl_dt['State/UT'], y=india_fnl_dt['Density'] ,color=color, label = 'Population Density',ax=ax1)

ax2.tick_params(axis='y', color=color)






