import pandas as pd

import numpy as np

from datetime import datetime, timedelta

import matplotlib.pyplot as plt
columns_names = df = pd.read_excel('../input/pracownicy/Pracownicy.xlsx', sheet_name= 0,header=None)

columns_names = [i[0] for i in columns_names.values]

columns_names

df = pd.read_excel('../input/pracownicy/Pracownicy.xlsx', sheet_name= 1,names = columns_names)

df
df = pd.DataFrame(df)
df.head(5)
df['Nazwisko'].head(5)
#loc - wybieranie podzioru za pomocą etykiet

df.loc[1,['Nazwisko','Pensja']]
#iloc - wybieranie podzioru za pomocą ineksów

df.iloc[1,[1,6]]
df.shape
df.dtypes
df.describe
df['Data Przyjecia'].astype('datetime64')
df = df.astype({'Data Przyjecia':'datetime64','Data Zwolnienia':'datetime64'})

df.dtypes
id_series = pd.Series([df.index])

df.insert(0,'id',df.index)

df
df['Imie'].str.lower()
df['Imie'].str.swapcase()
df['Stanowisko'] = df['Stanowisko'].fillna('Kierownik')
def Typ_Umowy(df):

    srednia_umowy = df[['Typ Umowy','Pensja']].groupby(by='Typ Umowy').mean()

    for i,j in enumerate(df['Typ Umowy']):

        if pd.isna(j):

            if df['Pensja'][i]< int(srednia_umowy.mean()):

                df.loc[i,['Typ Umowy']]= 'Zlecenie'

            else:

                df.loc[i,['Typ Umowy']]= 'UOP'    
Typ_Umowy(df)

df
df.sort_values(by = ['Data Przyjecia'])
srednia =  df[['Stanowisko','Pensja']].groupby(by = ['Stanowisko']).mean()

srednia = srednia.sort_values(by =['Pensja'], ascending = False)

srednia
liczba = df[['Stanowisko','Pensja']].groupby(by = ['Stanowisko']).count()

srednia.insert(0,'liczba_stanowisk',liczba)

srednia
odchylenie = df[['Stanowisko','Pensja']].groupby('Stanowisko').std()

odchylenie.insert(0,'liczba_stanowisk',liczba)
odchylenie.sort_values(by = 'Pensja',ascending = False)
df
df['Data Przyjecia'] + timedelta(hours = 8)
x = pd.DatetimeIndex(df['Data Przyjecia']).month

df['Miesiac'] = pd.DatetimeIndex(df['Data Przyjecia']).month

df['Rok'] = pd.DatetimeIndex(df['Data Przyjecia']).year

y = pd.DatetimeIndex(df['Data Przyjecia']).year

df['Rok/Miesiac'] = [str(i)[:7] for i in df['Data Przyjecia']]
df.sort_values(by='Data Przyjecia')
daty = pd.date_range(start='2016-06-01',end = '2019-12-01',freq='M',closed='left')

daty
slow = {}

for i,j in enumerate(daty):

    if i == 0:

        slow[str(daty[i])[:7]] = 0

    if i<len(daty)-1:

        slow[str(daty[i+1])[:7]] = slow[str(daty[i])[:7]]

    for k in range(len(df)):

        if j < df['Data Przyjecia'][k] <= daty[i+1]:            

            slow[str(daty[i+1])[:7]] = slow[str(daty[i])[:7]] + 1

        elif i<len(daty)-1 and j < df['Data Zwolnienia'][k] <= daty[i+1]:

            slow[str(daty[i+1])[:7]] -= 1



        
slow.items()
x,y = zip(*slow.items())
plt.figure(figsize=(30,8))

plt.xticks(rotation=45)

plt.plot(x,y)

plt.show()