# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import folium
import warnings
warnings.filterwarnings('ignore')

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# import data 
data = pd.read_csv('/kaggle/input/world-development-indicators/Indicators.csv')
# analyzed indicators
immunization_DPT = 'Immunization, DPT'
immunization_measles = 'Immunization, measles'
mortality_infant = 'Mortality rate, under-5'
data_DPT = data[data['IndicatorName'].str.contains(immunization_DPT)]
data_mortality = data[data['IndicatorName'].str.contains(mortality_infant)]
data_measles = data[data['IndicatorName'].str.contains(immunization_measles)]
# range of years for every indicator

print('Range of years for each indicator: \n{0} : from {1} to {2}\n{3}: from {4} to {5} \n{6}: from {7} to {8}'.format( \
       data_DPT['IndicatorName'].iloc[0],min(data_DPT['Year']), max(data_DPT['Year']), \
       data_measles['IndicatorName'].iloc[0],min(data_measles['Year']), max(data_measles['Year']), \
       data_mortality['IndicatorName'].iloc[0],min(data_mortality['Year']), max(data_mortality['Year']))) 
# truncate the data in data_mortality DataFrame
data_mortality_years = [x for x in data_mortality['Year'] if x>=1980 and x<=2014]
data_mortality_trunc = data_mortality[data_mortality['Year'].isin(data_mortality_years)]
print('New range of years for mortality indicator: from {} to {}'. \
      format(min(data_mortality_trunc['Year']),max(data_mortality_trunc['Year'])))
# Since the values for indicator mortality are presented as quantity per 1000, they should be transformed into percents
data_mortality_trunc['Value_in_percents'] = data_mortality_trunc['Value']/10
data_mortality_trunc.head()
# average values for each indicator per year
avg_mortality = data_mortality_trunc[['Year','Value_in_percents']].groupby('Year').mean()
avg_immunization_DPT = data_DPT[['Year','Value']].groupby('Year').mean()
avg_immunization_measles = data_measles[['Year','Value']].groupby('Year').mean()
%matplotlib inline
import matplotlib.pyplot as plt

fig,axis = plt.subplots()

axis.yaxis.grid(True)
axis.xaxis.grid(True)

axis.set_title('The effect of immunitizatin DPT on kids mortality',loc='center',fontsize=20)
axis.set_xlabel('Mortality rate under 5 years per 1000 in percents',fontsize=20)
axis.set_ylabel(data_DPT['IndicatorName'].iloc[0],fontsize=20)

X1 = avg_mortality['Value_in_percents']
Y1 = avg_immunization_DPT['Value'] 
fig.set_figwidth(20)
fig.set_figheight(10)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.ylim(30,100)
plt.xlim(3,10.5)
axis.scatter(X1,Y1)

plt.show()

fig,axis = plt.subplots()

axis.yaxis.grid(True)
axis.xaxis.grid(True)

axis.set_title('The effect of immunitizatin measles on kids mortality',loc='center',fontsize=20)
axis.set_xlabel('Mortality rate under 5 years per 1000 in percents',fontsize=20)
axis.set_ylabel(data_measles['IndicatorName'].iloc[0],fontsize=20)

X2 = avg_mortality['Value_in_percents']
Y2 = avg_immunization_measles['Value'] 
fig.set_figwidth(20)
fig.set_figheight(10)

plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.ylim(30,100)
plt.xlim(3,10.5)
axis.scatter(X2,Y2)

plt.show()
# Correlation between mortality and DPT immunization indicators
corr_1 = np.corrcoef(X1,Y1)
print('Correlation between mortality and DPT immunization indicators: ',round(corr_1[1][0],2))
# Correlation between mortality and measles immunization indicators
corr_2 = np.corrcoef(X2,Y2)
print('Correlation between mortality and measles immunization indicators: ',round(corr_2[1][0],2))
# the dimension for indicators of immunization different that's why in the next step the average values will be compared
print('The number of values for {0} indicator  is {1}\nThe number of values for {2} indicator  is {3}\n'.format(\
    data_DPT['IndicatorName'].iloc[0],len(data_DPT['Value']), data_measles['IndicatorName'].iloc[0],len(data_measles['Value'])))
fig,axis = plt.subplots()

axis.yaxis.grid(True)
axis.xaxis.grid(True)

axis.set_title('',loc='center',fontsize=20)
axis.set_xlabel(data_DPT['IndicatorName'].iloc[0],fontsize=20)
axis.set_ylabel(data_measles['IndicatorName'].iloc[0],fontsize=20)

X3 = avg_immunization_DPT['Value'] 
Y3 =  avg_immunization_measles['Value'] 
fig.set_figwidth(20)
fig.set_figheight(10)

plt.xticks(fontsize=15)
plt.yticks(fontsize=15)

axis.scatter(X3,Y3)

plt.show()
year = 2014
indicator_year = data['Year'].isin([year])
indicator_name = data['IndicatorName'].str.contains('Immunization')

data_2014 = data[indicator_year & indicator_name]
avg_values_immunization = data_2014[['CountryCode','Value']].groupby('CountryCode').mean() 
avg_values_immunization['Codes']  = sorted(data_2014['CountryCode'].unique().tolist())
avg_values_immunization.head()
map = folium.Map(location=[40, 10], zoom_start=1.5)
folium.Choropleth(geo_data='/kaggle/input/world-countries/world-countries.json', data=avg_values_immunization,
             columns=['Codes', 'Value'],
             key_on='feature.id',
             fill_color='PiYG', fill_opacity=0.7, line_opacity=1,
             legend_name='Immunization  DPT or mealses for children 12-23 months').add_to(map)
x = map.save('plot_immunization.html')
# This code perfectly works on Jupiter notebook on Kaggle, but unfortunately doesn't display properly
# Below I attach the printscreen of my picture. If anyone knows how to fix it, please write a comment. I would appreciate it.

#from IPython.display import IFrame
#IFrame(src= './plot_immunization.html', width=1000 ,height=450)

data_DPT_2014 = data_2014[data_2014['IndicatorName'].str.contains('DPT')]
data_measles_2014 = data_2014[data_2014['IndicatorName'].str.contains('measles')]

maximum_DPT_percentage = max(data_DPT_2014['Value'])
maximum_measles_percentage = max(data_measles_2014['Value'])
minimum_DPT_percentage = min(data_DPT_2014['Value'])
minimum_measles_percentage = min(data_measles_2014['Value'])

print('Maximum of {0} in 2014 is {1}'.format(data_DPT['IndicatorName'].iloc[0],maximum_DPT_percentage)) 
print('Maximum of {0} in 2014  is {1}'.format(data_measles['IndicatorName'].iloc[0],maximum_measles_percentage))
print('Minimum of {0} in 2014  is {1}'.format(data_DPT['IndicatorName'].iloc[0],minimum_DPT_percentage)) 
print('Minimum of {0} in 2014  is {1}'.format(data_measles['IndicatorName'].iloc[0],minimum_measles_percentage)) 
max_perc_immunization_countries = data_DPT_2014[data_DPT_2014['Value']==maximum_DPT_percentage]['CountryName'].unique().tolist()
min_perc_immunization_countries = data_DPT_2014[data_DPT_2014['Value']==minimum_DPT_percentage]['CountryName'].unique().tolist()

print('\n'+'\033[4m'+'\033[94m'+'\033[1m'+'Countries with the highest percentage of DPT immunization (children ages 12-23 months):\n'+'\033[0m')
for el in max_perc_immunization_countries:
    print(el)
    
print('\033[4m'+'\033[94m'+'\033[1m'+'\nCountries with the lowest percentage of DPT immunization (children ages 12-23 months):\n'+'\033[0m')
for el in min_perc_immunization_countries:
    print(el)
max_perc_immunization_countries = data_measles_2014[data_measles_2014['Value']==maximum_measles_percentage]['CountryName'].unique().tolist()
min_perc_immunization_countries = data_measles_2014[data_measles_2014['Value']==minimum_measles_percentage]['CountryName'].unique().tolist()

print('\n'+'\033[4m'+'\033[94m'+'\033[1m'+'Countries with the highest percentage of measles immunization (children ages 12-23 months):\n'+'\033[0m')
for el in max_perc_immunization_countries:
    print(el)
    
print('\033[4m'+'\033[94m'+'\033[1m'+'\nCountries with the lowest percentage of measles immunization (children ages 12-23 months):\n'+'\033[0m')
for el in min_perc_immunization_countries:
    print(el)

# DPT immunization indicator
lower_level = data_DPT_2014['Value'].mean() - 2*data_DPT_2014['Value'].std()
upper_level = data_DPT_2014['Value'].mean() + 2*data_DPT_2014['Value'].std()
hist_data = [x for x in data_DPT_2014['Value'] if x>lower_level and x<upper_level]
%matplotlib inline 

plt.figure(figsize=(10,6))
plt.hist(hist_data,12,density=False,facecolor='blue')
plt.xlabel('Percentage value for DPT indicator',fontsize=14)
plt.ylabel('Number of countries',fontsize = 14)
plt.title('Distribution for {}'.format(data_DPT_2014['IndicatorName'].iloc[0]),fontsize = 15)
plt.xlim(63, 100)
plt.grid(True)

plt.show()
# Measles immunization indicator
lower_level2 = data_measles_2014['Value'].mean() - 2*data_measles_2014['Value'].std()
upper_level2 = data_measles_2014['Value'].mean() + 2*data_measles_2014['Value'].std()
hist2_data = [x for x in data_measles_2014['Value'] if x>lower_level and x<upper_level]
%matplotlib inline 

plt.figure(figsize=(10,6))
plt.hist(hist2_data,12,density=False,facecolor='blue')
plt.xlabel('Percentage value for measles indicator',fontsize=14)
plt.ylabel('Number of countries',fontsize = 14)
plt.title('Distribution for {}'.format(data_measles_2014['IndicatorName'].iloc[0]),fontsize = 15)
plt.xlim(63, 100)
plt.grid(True)

plt.show()