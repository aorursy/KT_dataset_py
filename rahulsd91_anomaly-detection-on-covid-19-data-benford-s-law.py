# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# Most Significant Digit (MSD)



def msd(x):

    if x!=0:

        e = np.floor(np.log10(np.abs(x)))

        return int(np.abs(x)*10**-e)

    else:

        return 0
# Import data and calculate MSD for cumulative and daily cases/deaths 



data = pd.read_csv('/kaggle/input/corona-virus-report/covid_19_clean_complete.csv')

data = data[['Country/Region','Date','Confirmed','Deaths']]

data['Date'] = data['Date'].astype('datetime64')

data = data.groupby(['Country/Region','Date']).sum().reset_index()

data[['Daily Confirmed','Daily Deaths']] = (data[['Country/Region','Confirmed','Deaths']]).groupby('Country/Region').diff().dropna()

data = data.dropna()

data[['MSD Daily Cases','MSD Daily Deaths']] = data[['Daily Confirmed','Daily Deaths']].applymap(lambda x:msd(x))

data[['MSD Cases','MSD Deaths']] = data[['Confirmed','Deaths']].applymap(lambda x:msd(x))
# Global Daily Case Counts

plt.figure(figsize=(12,8))

counts = data.loc[data['MSD Daily Cases']!=0,'MSD Daily Cases'].value_counts()

counts = counts/counts.sum()

benford = np.log10(1 + 1/np.arange(1,10))

print('Correlation Coeffiecient for daily cases is '+ str(np.corrcoef(counts,benford)[0,1]))

plt.plot(counts,marker='o',ls='None')

# Global Daily Deaths



counts = data.loc[data['MSD Daily Deaths']!=0,'MSD Daily Deaths'].value_counts()

counts = counts/counts.sum()

benford = np.log10(1 + 1/np.arange(1,10))

print('Correlation Coeffiecient for daily deaths is '+ str(np.corrcoef(counts,benford)[0,1]))

plt.plot(counts,marker='o',ls='None')



# Global Case Counts



counts = data.loc[data['MSD Cases']!=0,'MSD Cases'].value_counts()

counts = counts/counts.sum()

benford = np.log10(1 + 1/np.arange(1,10))

print('Correlation Coeffiecient for total cases is '+ str(np.corrcoef(counts,benford)[0,1]))

plt.plot(counts,marker='o',ls='None')



# Global Death Counts



counts = data.loc[data['MSD Deaths']!=0,'MSD Deaths'].value_counts()

counts = counts/counts.sum()

benford = np.log10(1 + 1/np.arange(1,10))

print('Correlation Coeffiecient for total deaths is '+ str(np.corrcoef(counts,benford)[0,1]))

plt.plot(counts,marker='o',ls='None')

plt.xlabel('MSD (d)',size='xx-large')

plt.ylabel('P(d)',size='xx-large')

plt.legend(['Daily Cases','Daily Deaths','Cases','Deaths'])



plt.plot(np.arange(0.9,9.1,0.1),np.log10(1 + 1/np.arange(0.9,9.1,0.1)))
# Combination of all data



for param in ['MSD Daily Cases','MSD Daily Deaths','MSD Cases','MSD Deaths']:

    counts += data.loc[data[param]!=0,param].value_counts()

counts = counts/counts.sum()

benford = np.log10(1 + 1/np.arange(1,10))

print('Correlation Coeffiecient is '+ str(np.corrcoef(counts,benford)[0,1]))

plt.plot(counts,marker='o',ls='None')

plt.plot(np.arange(0.9,9.1,0.1),np.log10(1 + 1/np.arange(0.9,9.1,0.1)))
# Country-wise total data

pearsons = {}

for name,group in data.groupby('Country/Region'):

    counts = group.loc[group['MSD Daily Cases']!=0,'MSD Daily Cases'].value_counts()

    for param in ['MSD Daily Deaths','MSD Cases','MSD Deaths']:

        add_counts = group.loc[group[param]!=0,param].value_counts()

        counts = (counts+add_counts).fillna(add_counts).fillna(counts)

    idxs = [i-1 for i in counts.index.to_list()]

    if len(idxs)>0:

        counts = counts/counts.sum()

        p_c = np.corrcoef(counts,benford[idxs])[0,1]

        pearsons.update({name:p_c})

pearsons = pd.DataFrame.from_dict([pearsons]).T.reset_index()

pearsons = pearsons.rename(columns={'index':'Country/Region',0:'Pearson Coefficient'})

pearsons = pearsons.dropna().sort_values(by='Pearson Coefficient')
pearsons.plot(x='Country/Region',y='Pearson Coefficient',kind='scatter',rot=90,figsize=(16,4))
pearsons.loc[pearsons['Pearson Coefficient']<0.8].plot(x='Country/Region',y='Pearson Coefficient',kind='scatter',rot=90,figsize=(16,4))
data.loc[data['Country/Region']=='Barbados'].plot(x='Date',y='Confirmed',kind='scatter')

data.loc[data['Country/Region']=='Barbados'].plot(x='Date',y='Deaths',kind='scatter')

data.loc[data['Country/Region']=='Barbados'].plot(x='Date',y='Daily Confirmed',kind='scatter')

data.loc[data['Country/Region']=='Barbados'].plot(x='Date',y='Daily Deaths',kind='scatter')
# Country-wise correlations in daily cases/deaths



cd_pearsons = {}

dd_pearsons = {}

for name,group in data.groupby('Country/Region'):

    case_counts = group.loc[group['MSD Daily Cases']!=0,'MSD Daily Cases'].value_counts()

    c_idxs = [i-1 for i in case_counts.index.to_list()]

    death_counts = group.loc[group['MSD Daily Deaths']!=0,'MSD Daily Deaths'].value_counts()

    d_idxs = [i-1 for i in death_counts.index.to_list()]

    if len(c_idxs)>0:

        case_counts = case_counts/case_counts.sum()

        p_c = np.corrcoef(case_counts.values,benford[c_idxs])[0,1]

        cd_pearsons.update({name:p_c})

    if len(d_idxs)>0:

        death_counts = death_counts/death_counts.sum()

        p_d = np.corrcoef(death_counts.values,benford[d_idxs])[0,1]

        dd_pearsons.update({name:p_d})

        

cd_pearsons = pd.DataFrame.from_dict([cd_pearsons]).T.reset_index()

cd_pearsons = cd_pearsons.rename(columns={'index':'Country/Region',0:'Pearson Coefficient'})

cd_pearsons = cd_pearsons.dropna().sort_values(by='Pearson Coefficient')



dd_pearsons = pd.DataFrame.from_dict([dd_pearsons]).T.reset_index()

dd_pearsons = dd_pearsons.rename(columns={'index':'Country/Region',0:'Pearson Coefficient'})

dd_pearsons = dd_pearsons.dropna().sort_values(by='Pearson Coefficient')



# Country-wise correlations in total cases/deaths



c_pearsons = {}

d_pearsons = {}

for name,group in data.groupby('Country/Region'):

    case_counts = group.loc[group['MSD Cases']!=0,'MSD Cases'].value_counts()

    c_idxs = [i-1 for i in case_counts.index.to_list()]

    death_counts = group.loc[group['MSD Deaths']!=0,'MSD Deaths'].value_counts()

    d_idxs = [i-1 for i in death_counts.index.to_list()]

    if len(c_idxs)>0:

        case_counts = case_counts/case_counts.sum()

        p_c = np.corrcoef(case_counts.values,benford[c_idxs])[0,1]

        c_pearsons.update({name:p_c})

    if len(d_idxs)>0:

        death_counts = death_counts/death_counts.sum()

        p_d = np.corrcoef(death_counts.values,benford[d_idxs])[0,1]

        d_pearsons.update({name:p_d})

        

c_pearsons = pd.DataFrame.from_dict([c_pearsons]).T.reset_index()

c_pearsons = c_pearsons.rename(columns={'index':'Country/Region',0:'Pearson Coefficient'})

c_pearsons = c_pearsons.dropna().sort_values(by='Pearson Coefficient')



d_pearsons = pd.DataFrame.from_dict([d_pearsons]).T.reset_index()

d_pearsons = d_pearsons.rename(columns={'index':'Country/Region',0:'Pearson Coefficient'})

d_pearsons = d_pearsons.dropna().sort_values(by='Pearson Coefficient')
d_pearsons.plot(x='Country/Region',y='Pearson Coefficient',kind='scatter',rot=90,figsize=(16,4))

c_pearsons.plot(x='Country/Region',y='Pearson Coefficient',kind='scatter',

                     rot=90,figsize=(16,4))

cd_pearsons.plot(x='Country/Region',y='Pearson Coefficient',kind='scatter',

                     rot=90,figsize=(16,4))

dd_pearsons.plot(x='Country/Region',y='Pearson Coefficient',kind='scatter',

                     rot=90,figsize=(16,4))
threshold = 0.8

c_pearsons.loc[c_pearsons['Pearson Coefficient']<threshold

              ].plot(x='Country/Region',y='Pearson Coefficient',kind='scatter',

                     rot=90,figsize=(16,4))

d_pearsons.loc[d_pearsons['Pearson Coefficient']<threshold

              ].plot(x='Country/Region',y='Pearson Coefficient',kind='scatter',

                     rot=90,figsize=(16,4))

cd_pearsons.loc[cd_pearsons['Pearson Coefficient']<threshold

              ].plot(x='Country/Region',y='Pearson Coefficient',kind='scatter',

                     rot=90,figsize=(16,4))

dd_pearsons.loc[dd_pearsons['Pearson Coefficient']<threshold

              ].plot(x='Country/Region',y='Pearson Coefficient',kind='scatter',

                     rot=90,figsize=(16,4))
np.intersect1d(np.intersect1d(c_pearsons.loc[c_pearsons['Pearson Coefficient']<threshold,'Country/Region'].values,

               d_pearsons.loc[d_pearsons['Pearson Coefficient']<threshold,'Country/Region'].values),

               np.intersect1d(cd_pearsons.loc[cd_pearsons['Pearson Coefficient']<threshold,'Country/Region'].values,

               dd_pearsons.loc[dd_pearsons['Pearson Coefficient']<threshold,'Country/Region'].values))
data.loc[data['Country/Region']=='Belarus'].plot(x='Date',y='Confirmed',kind='scatter')

data.loc[data['Country/Region']=='Belarus'].plot(x='Date',y='Deaths',kind='scatter')

data.loc[data['Country/Region']=='Belarus'].plot(x='Date',y='Daily Confirmed',kind='scatter')

data.loc[data['Country/Region']=='Belarus'].plot(x='Date',y='Daily Deaths',kind='scatter')
threshold = 0.95

np.intersect1d(np.intersect1d(c_pearsons.loc[c_pearsons['Pearson Coefficient']<threshold,'Country/Region'].values,

               d_pearsons.loc[d_pearsons['Pearson Coefficient']<threshold,'Country/Region'].values),

               np.intersect1d(cd_pearsons.loc[cd_pearsons['Pearson Coefficient']<threshold,'Country/Region'].values,

               dd_pearsons.loc[dd_pearsons['Pearson Coefficient']<threshold,'Country/Region'].values))
# def country_to_continent(country_name):

#     if country_name == 'Congo (Brazzaville)':

#         country_continent_name = 'Africa' 

#     elif country_name=='Congo (Kinshasa)':

#         country_continent_name = 'Africa'

#     elif country_name == "Cote d'Ivoire":

#         country_continent_name = 'Africa'

#     elif country_name == "Holy See":

#         country_continent_name = 'Europe'

#     elif country_name == "Taiwan*":

#         country_continent_name = 'Asia'  

#     elif country_name == "Timor-Leste":

#         country_continent_name = 'Asia'

#     elif country_name == "West Bank and Gaza":

#         country_continent_name = 'Asia'

#     elif country_name == "Kosovo":

#         country_continent_name = 'Europe'        

#     elif country_name == "Burma":

#         country_continent_name = 'Asia'

#     elif country_name=='Western Sahara':

#         country_continent_name = 'Africa'

#     elif country_name in ['US','China','Russia','India','Brazil','Canada','United Kingdom']:

#         country_continent_name = country_name

        

#     else:

#         country_alpha2 = pc.country_name_to_country_alpha2(country_name)

#         country_continent_code = pc.country_alpha2_to_continent_code(country_alpha2)

#         country_continent_name = pc.convert_continent_code_to_continent_name(country_continent_code)

#     return country_continent_name
