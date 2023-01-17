# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
from matplotlib import pyplot as plt

import seaborn as sns

from datetime import datetime



%matplotlib inline

#Get data from online csv files on John Hopkins University repository

conf_url = 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv'

deaths_url = 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_global.csv'

recovered_url = 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_recovered_global.csv'
#Load data

raw_tot_confirmed = pd.read_csv(conf_url)

raw_tot_deaths = pd.read_csv(deaths_url)

raw_tot_recovered = pd.read_csv(recovered_url)



print("The Shape of Cornfirmed cases is: ", raw_tot_confirmed.shape)

print("The Shape of Cornfirmed deaths is: ", raw_tot_deaths.shape)

print("The Shape of Cornfirmed recoveries is: ", raw_tot_recovered.shape)



raw_tot_confirmed.head()
# Un-Pivoting the data (reorganize with the dates as a culumn/feature)



raw_tot_confirmed2 = pd.melt(raw_tot_confirmed, id_vars=['Province/State', 'Country/Region', 'Lat', 'Long'], var_name=['Date'])

raw_tot_deaths2 = pd.melt(raw_tot_deaths, id_vars=['Province/State', 'Country/Region', 'Lat', 'Long'], var_name=['Date'])

raw_tot_recovered2 = pd.melt(raw_tot_recovered, id_vars=['Province/State', 'Country/Region', 'Lat', 'Long'], var_name=['Date'])





print("Confirmed cases shape is: ", raw_tot_confirmed2.shape)

print("Confirmed deaths shape is: ", raw_tot_deaths2.shape)

print("Confirmed recoveries shape is: ", raw_tot_recovered2.shape)





raw_tot_confirmed2.head()
# Un-Pivoting the data (reorganize with the dates as a culumn/feature)



raw_tot_confirmed2 = pd.melt(raw_tot_confirmed, id_vars=['Province/State', 'Country/Region', 'Lat', 'Long'], var_name=['Date'])

raw_tot_deaths2 = pd.melt(raw_tot_deaths, id_vars=['Province/State', 'Country/Region', 'Lat', 'Long'], var_name=['Date'])

raw_tot_recovered2 = pd.melt(raw_tot_recovered, id_vars=['Province/State', 'Country/Region', 'Lat', 'Long'], var_name=['Date'])





print("Confirmed cases shape is: ", raw_tot_confirmed2.shape)

print("Confirmed deaths shape is: ", raw_tot_deaths2.shape)

print("Confirmed recoveries shape is: ", raw_tot_recovered2.shape)





raw_tot_confirmed2.head()
#Converting the new column to dates



raw_tot_confirmed2['Date'] = pd.to_datetime(raw_tot_confirmed2['Date'])

raw_tot_deaths2['Date'] = pd.to_datetime(raw_tot_deaths2['Date'])

raw_tot_recovered2['Date'] = pd.to_datetime(raw_tot_recovered2['Date'])
# Renaming the Values

raw_tot_confirmed2.columns = raw_tot_confirmed2.columns.str.replace('value', 'Confirmed')

raw_tot_deaths2.columns = raw_tot_deaths2.columns.str.replace('value', 'Deaths')

raw_tot_recovered2.columns = raw_tot_recovered2.columns.str.replace('value', 'Recovered')
# Investigating the NULL values

raw_tot_recovered2.isnull().sum()
raw_tot_confirmed2['Province/State'].fillna(raw_tot_confirmed2['Country/Region'], inplace=True)

raw_tot_deaths2['Province/State'].fillna(raw_tot_deaths2['Country/Region'], inplace=True)

raw_tot_recovered2['Province/State'].fillna(raw_tot_recovered2['Country/Region'], inplace=True)



raw_tot_confirmed2.isnull().sum()
# Full Joins



# Confirmed with Deaths

full_join = raw_tot_confirmed2.merge(raw_tot_deaths2[['Province/State','Country/Region','Date','Deaths']], 

                                      how = 'left', 

                                      left_on = ['Province/State','Country/Region','Date'], 

                                      right_on = ['Province/State', 'Country/Region','Date'])



print("Shape of first join: ", full_join.shape)



# full join with Recovered

full_join = full_join.merge(raw_tot_recovered2[['Province/State','Country/Region','Date','Recovered']], 

                                      how = 'left', 

                                      left_on = ['Province/State','Country/Region','Date'], 

                                      right_on = ['Province/State', 'Country/Region','Date'])



print("Shape of second join: ", full_join.shape)



full_join.tail()
full_join['Date'] = pd.to_datetime(full_join['Date'])

#Removing null values

full_join['Recovered'].fillna(0, inplace=True)

#Calculate active cases

full_join['Active Cases'] = full_join['Confirmed'] - (full_join['Deaths'] + full_join['Recovered']) 



# Adding Month and Year as a new Column

full_join['Month-Year'] = full_join['Date'].dt.strftime('%b-%Y')



full_join.head()
# Braking the numbers by Day 





## Applying it on all dataset



#creating a new df    

full_join2 = full_join.copy()



#creating a new date columns - 1

full_join2['Date - 1'] = full_join2['Date'] + pd.Timedelta(days=1)

full_join2.rename(columns={'Confirmed': 'Confirmed - 1', 'Deaths': 'Deaths - 1', 'Recovered': 'Recovered - 1',

                          'Date': 'Date Minus 1'}, inplace=True)



#Joing on the 2 DFs

full_join3 = full_join.merge(full_join2[['Province/State', 'Country/Region','Confirmed - 1', 'Deaths - 1', 

                            'Recovered - 1', 'Date - 1', 'Date Minus 1']], how = 'left',

                             left_on = ['Province/State','Country/Region','Date'], 

                             right_on = ['Province/State', 'Country/Region','Date - 1'])



#minus_onedf.rename(columns={'Confirmed': 'Confirmed - 1', 'Deaths': 'Deaths - 1', 'Recovered': 'Recovered - 1'}, inplace=True)



full_join3.head()



# Additional Calculations

full_join3['Confirmed Daily'] = full_join3['Confirmed'] - full_join3['Confirmed - 1']

full_join3['Deaths Daily'] = full_join3['Deaths'] - full_join3['Deaths - 1']

full_join3['Recovered Daily'] = full_join3['Recovered'] - full_join3['Recovered - 1']



print(full_join3.shape)
# Editing manually the numbers for first day



full_join3['Confirmed Daily'].loc[full_join3['Date'] == '2020-01-22'] = full_join3['Confirmed']

full_join3['Deaths Daily'].loc[full_join3['Date'] == '2020-01-22'] = full_join3['Deaths']

full_join3['Recovered Daily'].loc[full_join3['Date'] == '2020-01-22'] = full_join3['Recovered']



# deleting columns

del full_join3['Confirmed - 1']

del full_join3['Deaths - 1']

del full_join3['Recovered - 1']

del full_join3['Date - 1']

del full_join3['Date Minus 1']
#Calculate daily active cases

full_join3['Active Cases Daily'] = full_join3['Confirmed Daily'] - (full_join3['Deaths Daily'] + full_join3['Recovered Daily']) 

full_join3.shape
# removing all rows that have no useful data (dates that had no cases in countries)

new_join = full_join3.loc[full_join3['Confirmed'] * 1 != 0]

print(new_join.shape)
my_join = new_join

#Grouping the data by Continent

Africa = ['Kenya', 'Uganda','Tanzania', 'Rwanda', 'Burundi', 'Ethiopia', 'Mauritania', 'Sudan', 'Eritrea', 'South Sudan', 'Western Sahara', 'Somalia', 'Djibouti', 'Mozambique',  'Mauritius', 'Madagascar', 'Malawi', 'Zambia', 'Zimbabwe', 'Seychelles', 'Congo (Kinshasa)', 'Angola', 'Cameroon', 'Chad', 'Central African Republic', 'Congo (Brazzaville)', 'Gabon', 'Equatorial Guinea', 'Sao Tome and Principe', 'South Africa', 'Namibia', 'Botswana', 'Eswatini', 'Nigeria', 'Ghana', "Cote d'Ivoire", 'Niger', 'Burkina Faso', 'Mali', 'Cabo Verde', 'Togo', 'Sierra Leone', 'Liberia', 'Senegal', 'Guinea', 'Benin', 'Guinea-Bissau', 'Egypt', 'Algeria', 'Morocco', 'Tunisia', 'Libya', 'Gambia', 'Comoros', 'Lesotho']

Asia = ['China', 'Nepal', 'India', 'Pakistan', 'Bangladesh', 'Iran', 'Afghanistan', 'Sri Lanka', 'Maldives', 'Sri Lanka', 'Maldives', 'Japan', 'Korea, South', 'Taiwan*', 'Mongolia', 'Indonesia', 'Philippines', 'Vietnam', 'Thailand', 'Burma', 'Malaysia', 'Cambodia', 'Laos', 'Singapore', 'Timor-Leste', 'Brunei', 'Uzbekistan', 'Kazakhstan', 'Kyrgyzstan', 'Turkey', 'Iraq', 'Saudi Arabia', 'Yemen', 'Syria', 'Azerbaijan', 'Iraq', 'Saudi Arabia', 'Yemen', 'Syria', 'Azerbaijan', 'United Arab Emirates', 'Israel', 'Jordan', 'Lebanon', 'State of Palestine', 'Oman', 'Kuwait', 'Georgia', 'Armenia', 'Qatar', 'Bahrain', 'Cyprus', 'West Bank and Gaza', 'Bhutan', 'Tajikistan']

Europe = ['Germany', 'France', 'Netherlands', 'Belgium', 'Austria', 'Switzerland', 'Kosovo', 'Luxembourg', 'Monaco', 'Liechtenstein', 'Russia', 'Ukraine', 'Poland', 'Romania', 'Czechia', 'Belarus', 'Hungary', 'Bulgaria', 'Slovakia', 'Moldova', 'United Kingdom', 'Sweden', 'Denmark', 'Norway', 'Latvia', 'Estonia', 'Iceland', 'Finland', 'Lithuania', 'Italy', 'Spain', 'Greece', 'Portugal', 'Serbia', 'Croatia', 'Bosnia and Herzegovina', 'Albania', 'North Macedonia', 'Slovenia', 'Montenegro', 'Malta', 'Andorra', 'Gibraltar', 'San Marino', 'Holy See', 'Ireland']

N_America = ['US', 'Canada']

S_America = ['Brazil', 'Colombia', 'Argentina', 'Peru', 'Venezuela', 'Chile', 'Ecuador', 'Bolivia', 'Paraguay', 'Uruguay', 'Guyana', 'Suriname', 'Falkland Islands', 'Mexico', 'Guatemala', 'Honduras', 'Nicaragua', 'El Salvador', 'Costa Rica', 'Panama', 'Belize', 'Cuba', 'Haiti', 'Dominican Republic', 'Puerto Rico', 'Jamaica', 'Trinidad and Tobago', 'Guadeloupe', 'Martinique', 'Bahamas', 'Barbados', 'Saint Lucia', 'Curaçao', 'Saint Vincent - Grenadines', 'Grenada', 'Aruba', 'Antigua and Barbuda', 'Dominica', 'Saint Kitts and Nevis', 'Anguilla', 'Montserrat', 'Saint-Barthélemy', 'Saint Vincent and the Grenadines']

Australia = ['Australia', 'New Zealand', 'Papua New Guinea', 'Fiji', 'Solomon Islands', 'Vanuatu', 'Samoa', 'Tonga', 'Wallis and Futuna Islands', 'Tuvalu', 'Niue', 'Tokelau', 'Guam', 'Kiribati', 'Micronesia', 'Northern Mariana Islands', 'Marshall Islands', 'Palau', 'Nauru']

Other = ['MS Zaandam', 'Diamond Princess']



#Creating a condition to categorize countiries' rows into continents 

my_cond=[my_join['Country/Region'].isin(Africa),my_join['Country/Region'].isin(Asia),my_join['Country/Region'].isin(Europe),my_join['Country/Region'].isin(N_America),my_join['Country/Region'].isin(S_America),my_join['Country/Region'].isin(Australia),my_join['Country/Region'].isin(Other)]

continents = ['Africa', 'Asia', 'Europe', 'N_America', 'S_America', 'Australia', 'Other']



#Creating continents column based on the above condition

my_join['Continent'] = np.select(my_cond, continents)



#check to see if there is any new country that has not been assigned a country

my_join[my_join['Continent'] == '0']
#reorder columns

my_join = my_join[['Continent', 'Country/Region', 'Province/State', 'Lat', 'Long','Month-Year', 'Date', 'Confirmed', 'Deaths', 'Recovered','Active Cases', 'Confirmed Daily', 'Deaths Daily', 'Recovered Daily', 'Active Cases Daily']]

my_join.head()
#Organize data to continent summaries

cov_cont = my_join.groupby(['Continent','Date'], as_index=False).sum()

cov_cont.drop(columns=['Long', 'Lat'], inplace=True)

cov_cont
cov_cont['Death Rate']= cov_cont['Deaths']/cov_cont['Confirmed']

cov_cont['Recovery Rate'] = cov_cont['Recovered']/cov_cont['Confirmed']

cov_cont['A.Cases Proportion'] = cov_cont['Active Cases']/cov_cont['Confirmed']

cov_cont