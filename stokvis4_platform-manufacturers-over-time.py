# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt

%matplotlib inline

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
df = pd.read_csv('../input/vgsales.csv')
df.head()
df.info()
df.groupby('Platform')['NA_Sales','EU_Sales','JP_Sales','Other_Sales','Global_Sales']\
        .sum()\
        .sort_values('Global_Sales',ascending=False)
def platform_publisher(x):
    sony = re.compile(r'PS')
    microsoft = re.compile(r'(XB|X360|XOne)')
    nintendo = re.compile(r'(Wii|DS|GBA|GB|SNES|NES|WiiU|GC|3DS)')
    sega = re.compile(r'(GEN|SAT|DC|SCD)')
    computer = 'PC'
    atari = '2600'
    nec = re.compile(r'(TG16|PCFX)')
    if sony.match(x):
        return 'Sony'
    elif microsoft.match(x):
        return 'Microsoft'
    elif nintendo.match(x):
        return 'Nintendo'
    elif sega.match(x):
        return  'Sega'
    elif x == computer:
        return 'Computer'
    elif x == atari:
        return 'Atari'
    elif nec.match(x):
        return 'NEC'
    else:
        return 'Misc'
def mobile_platform(x):
    mobile = re.compile(r'(PS(V|P)|GBA?|3?DS|WS)')
    if mobile.match(x):
        return 'Handheld'
    elif x == 'PC':
        return 'Computer'
    else:
        return 'Console'
df['Platform_General'] = df['Platform'].apply(platform_publisher)
df['Mobile_Status'] = df['Platform'].apply(mobile_platform)
df.sort_values('NA_Sales', ascending=False).head()
fig, (ax1, ax2) = plt.subplots(2,1,figsize = (20,12))

plt.suptitle('Hardware Investigation', fontsize = 20)
ax1.set_title('Game Sales by Hardware Type')
ax1.plot(df[df.Mobile_Status == 'Console'].groupby('Year')['Global_Sales'].sum())
ax1.plot(df[df.Mobile_Status == 'Handheld'].groupby('Year')['Global_Sales'].sum())
ax1.plot(df[df.Mobile_Status == 'Computer'].groupby('Year')['Global_Sales'].sum())
ax1.legend(['Console','Handheld','Computer'],loc='best')
ax1.grid(b='On', axis = 'y', color = '#E0E0E0')
ax1.set_ylabel('Global Sales in millions of units')

ax2.set_title('Sales of Video Games for Consoles by Manufacturer')
ax2.plot(df[(df.Platform_General == 'Microsoft') & (df.Mobile_Status == 'Console')].groupby('Year')['Global_Sales'].sum())
ax2.plot(df[(df.Platform_General == 'Sony') & (df.Mobile_Status == 'Console')].groupby('Year')['Global_Sales'].sum())
ax2.plot(df[(df.Platform_General == 'Nintendo') & (df.Mobile_Status == 'Console')].groupby('Year')['Global_Sales'].sum())
ax2.plot(df[(df.Platform_General == 'Sega') & (df.Mobile_Status == 'Console')].groupby('Year')['Global_Sales'].sum())
ax2.plot(df[(df.Platform_General == 'Atari') & (df.Mobile_Status == 'Console')].groupby('Year')['Global_Sales'].sum())
ax2.legend(['Microsoft','Sony','Nintendo','Sega','Atari'],loc='best')
ax2.grid(b='On', axis = 'y', color = '#E0E0E0')
ax2.set_ylabel('Global Sales in millions of units');

fig, (ax1, ax2) = plt.subplots(2,1,figsize = (20,12))
plt.suptitle('Console Sales Investigation', fontsize = 16)
ax1.set_title('United States')
ax1.plot(df[(df.Platform_General == 'Microsoft') & (df.Mobile_Status == 'Console')].groupby('Year')['NA_Sales'].sum())
ax1.plot(df[(df.Platform_General == 'Sony') & (df.Mobile_Status == 'Console')].groupby('Year')['NA_Sales'].sum())
ax1.plot(df[(df.Platform_General == 'Nintendo') & (df.Mobile_Status == 'Console')].groupby('Year')['NA_Sales'].sum())
ax1.plot(df[(df.Platform_General == 'Sega') & (df.Mobile_Status == 'Console')].groupby('Year')['NA_Sales'].sum())
ax1.plot(df[(df.Platform_General == 'Atari') & (df.Mobile_Status == 'Console')].groupby('Year')['NA_Sales'].sum())
ax1.legend(['Microsoft','Sony','Nintendo','Sega','Atari'],loc='best')
ax1.grid(b='On', axis = 'y', color = '#E0E0E0')
ax1.set_ylabel('US Sales in millions of units');

ax2.set_title('Japan')
ax2.plot(df[(df.Platform_General == 'Microsoft') & (df.Mobile_Status == 'Console')].groupby('Year')['JP_Sales'].sum())
ax2.plot(df[(df.Platform_General == 'Sony') & (df.Mobile_Status == 'Console')].groupby('Year')['JP_Sales'].sum())
ax2.plot(df[(df.Platform_General == 'Nintendo') & (df.Mobile_Status == 'Console')].groupby('Year')['JP_Sales'].sum())
ax2.plot(df[(df.Platform_General == 'Sega') & (df.Mobile_Status == 'Console')].groupby('Year')['JP_Sales'].sum())
ax2.plot(df[(df.Platform_General == 'Atari') & (df.Mobile_Status == 'Console')].groupby('Year')['JP_Sales'].sum())
ax2.legend(['Microsoft','Sony','Nintendo','Sega','Atari'],loc='best')
ax2.grid(b='On', axis = 'y', color = '#E0E0E0')
ax2.set_ylabel('Japan Sales in millions of units');

df_nintendo_perc = (df[(df['Publisher'] == 'Nintendo') & (df['Platform_General'] == 'Nintendo')]\
                        .groupby('Year')['Global_Sales'].sum())\
                        /(df[(df['Platform_General'] == 'Nintendo')].groupby('Year')['Global_Sales'].sum())
df_microsoft_perc = (df[(df['Publisher'] == 'Microsoft Game Studios') & (df['Platform_General'] == 'Microsoft')]\
                            .groupby('Year')['Global_Sales'].sum())\
                            /(df[(df['Platform_General'] == 'Microsoft')].groupby('Year')['Global_Sales'].sum())
df_sony_perc = (df[(df['Publisher'] == 'Sony Computer Entertainment') & (df['Platform_General'] == 'Sony')]\
                    .groupby('Year')['Global_Sales'].sum())\
                    /(df[(df['Platform_General'] == 'Sony')].groupby('Year')['Global_Sales'].sum())
df_sega_perc = (df[(df['Publisher'] == 'Sega') & (df['Platform_General'] == 'Sega')]\
                    .groupby('Year')['Global_Sales'].sum())\
                    /(df[(df['Platform_General'] == 'Sega')].groupby('Year')['Global_Sales'].sum())
df_nintendo_perc = df_nintendo_perc.to_frame().reset_index().rename(columns= {'Global_Sales':'Nintendo'})
df_microsoft_perc = df_microsoft_perc.to_frame().reset_index().rename(columns= {'Global_Sales':'Microsoft'})
df_sony_perc = df_sony_perc.to_frame().reset_index().rename(columns= {'Global_Sales':'Sony'})
df_sega_perc = df_sega_perc.to_frame().reset_index().rename(columns= {'Global_Sales':'Sega'})
df_owned_perc = pd.merge(pd.merge(pd.merge(df_nintendo_perc,df_sega_perc, on='Year', how = 'outer'),\
                         df_sony_perc,on='Year', how = 'outer')\
                         ,df_microsoft_perc,on='Year', how = 'outer')
plt.figure(figsize = (20,8))
plt.plot(df_owned_perc.iloc[:,0],df_owned_perc.iloc[:,1:])
plt.xlabel('Year', fontsize = 14)
plt.ylabel('% of Total Sales where Publisher and Platform are the Same', fontsize = 14)
plt.legend(['Nintendo','Sega','Sony','Microsoft'])
plt.grid(b='On', axis = 'y', color = '#E0E0E0')
plt.yticks([x/100 for x in range(0,101,25)])
plt.ylim([0,1])
plt.title('% of Video Game Sales of Total Sales where Publisher and Platform are the Same \n', fontsize = 20);
