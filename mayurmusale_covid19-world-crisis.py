# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import requests
import warnings
warnings.filterwarnings("ignore")

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
Confirmed = pd.read_csv('../input/time_series_covid_19_confirmed.csv')
Death = pd.read_csv('../input/time_series_covid_19_deaths.csv')
Recover = pd.read_csv('../input/time_series_covid_19_recovered.csv')
#Groupe by countries
Covid19_Confirmed = pd.DataFrame(Confirmed.groupby('Country/Region').sum())
Covid19_Death = pd.DataFrame(Death.groupby('Country/Region').sum())
Covid19_Recover = pd.DataFrame(Recover.groupby('Country/Region').sum())
# Reset indexs
Covid19_Confirmed.reset_index(inplace=True)
Covid19_Death.reset_index(inplace=True)
Covid19_Recover.reset_index(inplace=True)
# Selected required columns only
Confirmed_Cases = Covid19_Confirmed[['Country/Region','1/31/20','2/29/20','3/31/20','4/4/20']]
Death_Cases = Covid19_Death[['Country/Region','1/31/20','2/29/20','3/31/20','4/4/20']]
Recover_Cases = Covid19_Recover[['Country/Region','1/31/20','2/29/20','3/31/20','4/4/20']]
# Renameing column name
Confirmed_Cases.rename(columns={'1/31/20':'January','2/29/20':'February','3/31/20':'March','4/4/20':'MTD_Cases'},inplace=True)
Death_Cases.rename(columns={'1/31/20':'January','2/29/20':'February','3/31/20':'March','4/4/20':'MTD_Death'},inplace=True)
Recover_Cases.rename(columns={'1/31/20':'January','2/29/20':'February','3/31/20':'March','4/4/20':'MTD_Recover'},inplace=True)
Top_confirm_ctr = Confirmed_Cases.nlargest(5,['MTD_Cases']) # Top 5 confirmed cases countries
Top_death_ctr = Death_Cases.nlargest(5,['MTD_Death']) # Top 5 Death cases countries
Top_recover_ctr = Recover_Cases.nlargest(5,['MTD_Recover']) # Top 5 Recover cases countries
fig, ax = plt.subplots(1, 3, figsize=(20,4), dpi=300)

ax[0].set_title('Confirmed')
ax[0].set_facecolor('aliceblue')
Top_confirm_ctr.plot(ax = ax[0], x='Country/Region',y = 'MTD_Cases', color='c',  marker='o', linestyle='--', lw=3)
ax[0].grid(linestyle='--',color='gray')

ax[1].set_title('Death')
ax[1].set_facecolor('aliceblue')
Top_death_ctr.plot(ax = ax[1], x= 'Country/Region', y = 'MTD_Death', color='tomato',  marker='o', linestyle='--', lw=3)
ax[1].grid(linestyle='--',color='gray')

ax[2].set_title('Recover')
ax[2].set_facecolor('aliceblue')
Top_recover_ctr.plot(ax = ax[2],x = 'Country/Region',y='MTD_Recover' ,color='green',  marker='o', linestyle='--', lw=3)
ax[2].grid(linestyle='--',color='gray')

plt.show()
#made single DF merging by all DF for compare
MTD_Compare = pd.merge(pd.merge(Confirmed_Cases[['Country/Region','MTD_Cases']],
                      Death_Cases[['Country/Region','MTD_Death']], how='left',on='Country/Region'),
             Recover_Cases[['Country/Region','MTD_Recover']],how='left',on='Country/Region')
# Sort to choose TOP
MTD_Compare.sort_values(by=['MTD_Cases','MTD_Death','MTD_Recover'],inplace=True, ascending=False)
# add column to calculate Death and recovery % over total cases
MTD_Compare['Death%'] = MTD_Compare['MTD_Death']*100/MTD_Compare['MTD_Cases']
MTD_Compare['Recover%'] = MTD_Compare['MTD_Recover']*100/MTD_Compare['MTD_Cases']
# Top 10 countries who has max confirmed cases
Top10_DOR = MTD_Compare.nlargest(10,['MTD_Cases'])
fig, ax = plt.subplots(1, 1, figsize=(20,4))

ax.set_title('Top10_DOR')
ax.set_facecolor('honeydew')
x = Top10_DOR['Country/Region']
y = Top10_DOR['Death%']
z = Top10_DOR['Recover%']
ax.plot(x,y, 'red',linestyle='--',lw=2, marker='o')
ax.plot(x,z, 'lightseagreen',linestyle='--',lw=2, marker='o')
plt.grid(linestyle='--', color='gray')

#Top10_DOR.plot(ax = ax, x='Country/Region', y = ['Death%','Recover%'], color=['red','lightseagreen'], linestyle='--',lw=2, marker='o' )

for (i,j,k) in zip(Top10_DOR['Country/Region'],Top10_DOR['Death%'],Top10_DOR['Recover%']):
    plt.annotate(round(j, 2), xy=(i,j))
    plt.annotate(round(k, 2), xy=(i,k))

plt.show()
# Seperated data of those countries who has more than 500 cases
Top_Ctry = MTD_Compare[MTD_Compare['MTD_Cases']>500]
Top_Ctry.sort_values(by=['Recover%'],inplace=True, ascending=False)

# number of countries which has greater deth rate than recovery
Max_Death = Top_Ctry[Top_Ctry['Death%']>Top_Ctry['Recover%']]
Max_Death.sort_values(by=['Death%'],inplace=True, ascending=False)
# top countries recovery vs Death analysis

fig, ax = plt.subplots(1, 1, figsize=(20,4), dpi=200)

x = Top_Ctry['Country/Region']
y = Top_Ctry['Death%']
z = Top_Ctry['Recover%']
ax.set_title('Top_Ctry')
ax.set_facecolor('cornsilk')
ax.plot(x,y, 'red',linestyle='--',lw=2, marker='o')
ax.plot(x,z, 'blue',linestyle='--',lw=2, marker='o')
plt.grid(linestyle='--', color='gray')
plt.xticks(rotation=90)

#Top10_DOR.plot(ax = ax, x='Country/Region', y = ['Death%','Recover%'], color=['red','lightseagreen'], linestyle='--',lw=2, marker='o' )

plt.show()

# Analysis on those countries who has max death rate than recovery rate

fig, ax = plt.subplots(1, 1, figsize=(20,4), dpi=200)
x = Max_Death['Country/Region']
y = Max_Death['Death%']
z = Max_Death['Recover%']
ax.set_title('Max_Death')
ax.set_facecolor('cornsilk')
ax.plot(x,y,'red', linestyle='--',lw=2, marker='o')
ax.plot(x,z, 'blue', linestyle='--',lw=2, marker='o')
plt.grid(linestyle='--', color='gray')
plt.xticks(rotation=90)

for (i,j,k) in zip(Max_Death['Country/Region'],Max_Death['Death%'],Max_Death['Recover%']):
    plt.annotate(round(j, 2), xy=(i,j))
    plt.annotate(round(k, 2), xy=(i,k))


plt.show()
Max_Death
