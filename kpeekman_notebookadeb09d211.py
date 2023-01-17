# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

pd.options.display.float_format = '{:.2f}'.format



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



%matplotlib inline

pd.set_option('display.max_rows', 51)

# Any results you write to the current directory are saved as output.
### ANDMED TABELIST

df = pd.read_csv("../input/ToojoukuluIIIkv.csv", encoding="Latin-1", sep=";", decimal='.')

#df.rename(columns={'Riiklikud maksud': 'Riiklikud_maksud', 'Tööjõumaksud ja maksed': 'Tööjõumaksud_maksed', 'Töötajate arv':'Töötajate_arv'}, inplace=True)

df['Tööjõumaksud_maksed'] = df['Tööjõumaksud_maksed'].str.replace(',','.')

df['Kaive'] = df['Kaive'].str.replace(',','.')

df['Tööjoukulu_kuus'] = df['Tööjoukulu_kuus'].str.replace(',','.')

df['Käive_kuus'] = df['Käive_kuus'].str.replace(',','.')

df['Keskmine_töötasu_kuus(bruto)'] = df['Keskmine_töötasu_kuus(bruto)'].str.replace(',','.')

df['Keskmine_töötasu_kuus_neto'] = df['Keskmine_töötasu_kuus_neto'].str.replace(',','.')

df['Tööjoukulu_osakaal_kaibest'] = df['Tööjoukulu_osakaal_kaibest'].str.replace(',','.')

df['Kuu_käive_töötaja_kohta'] = df['Kuu_käive_töötaja_kohta'].str.replace(',','.')

df['Tööjõumaksud_maksed'] = df['Tööjõumaksud_maksed'].astype(float)

df['Kaive'] = df['Kaive'].astype(float)

df['Tööjoukulu_kuus'] = df['Tööjoukulu_kuus'].astype(float)

df['Käive_kuus'] = df['Käive_kuus'].astype(float)

df['Keskmine_töötasu_kuus(bruto)'] = df['Keskmine_töötasu_kuus(bruto)'].astype(float)

df['Keskmine_töötasu_kuus_neto'] = df['Keskmine_töötasu_kuus_neto'].astype(float)

df['Tööjoukulu_osakaal_kaibest'] = df['Tööjoukulu_osakaal_kaibest'].astype(float)

df['Kuu_käive_töötaja_kohta'] = df['Kuu_käive_töötaja_kohta'].astype(float)

df
# INFO TABELI ANDMETE KOHTA

df.info()
### EESTI TOP 50 käibe alusel

df5=df[["Registrikood", "Nimi", "Liik", "Maakond","Käive_kuus", "Tootajaid"]].sort_values("Käive_kuus", ascending=False).head(50)

df5[["Registrikood", "Nimi", "Liik", "Maakond","Käive_kuus", "Tootajaid"]]

### TARTU TOP 50 KÄIBE ALUSEL

df2=df[df['Maakond']== "Tartu maakond"].sort_values("Käive_kuus", ascending=False).head(50)

df2[["Registrikood", "Nimi", "Liik", "Maakond","Käive_kuus", "Tootajaid"]]
### EESTI TOP 50 KESKMISE BRUTOTASU ALUSEL

df4=df[["Registrikood", "Nimi", "Liik", "Maakond","Keskmine_töötasu_kuus(bruto)","Keskmine_töötasu_kuus_neto"]].sort_values("Keskmine_töötasu_kuus(bruto)", ascending=False).head(50)

df4
### TARTU TOP 50 KESKMISE BRUTOTASU ALUSEL

df1=df[df['Maakond']== "Tartu maakond"].sort_values("Keskmine_töötasu_kuus(bruto)", ascending=False).head(50)

df1[["Registrikood", "Nimi", "Liik", "Maakond","Keskmine_töötasu_kuus(bruto)", "Keskmine_töötasu_kuus_neto"]]
### KESKMISTE TÖÖTASUDE ESINEMISSAGEDUS VAHEMIKUS 0-6000 ÜLDTABEL

df['Keskmine_töötasu_kuus(bruto)'].plot.hist(bins=50, range=(0,6000));

#df['Keskmine_töötasu_kuus(bruto)'].plot.hist(bins=100, grid=False, rwidth=50); 
##KESKMISE TÖÖTASU ESINEMISSAGEDUS EESTI TOP 50

df4['Keskmine_töötasu_kuus(bruto)'].plot.hist(bins=50, range=(1000,10000));
##KESKMISE TÖÖTASU ESINEMISSAGEDUS TARTU TOP 50

df1['Keskmine_töötasu_kuus(bruto)'].plot.hist(bins=50, range=(0,7000));
### KUU TÖÖJÕUKULU OSAKAAL KUU KÄIBEST

(df['Tööjoukulu_osakaal_kaibest']*100).plot.hist(bins=70, range=(0,150));
### TÖÖTAJATE ARVU JA KÄIBE SEOS (ÜLDTABEL)

df.plot.scatter("Tootajaid","Käive_kuus", alpha=0.2, xlim=(0,500),ylim=(0,1000000));
###TÖÖTAJATE ARVU JA KÄIBE SEOS EESTI TOP 50

df5.plot.scatter("Tootajaid","Käive_kuus", alpha=0.4, xlim=(0,500),ylim=(0,1000000));
###TÖÖTAJATE ARVU JA KÄIBE SEOS TARTU TOP 50

df2.plot.scatter("Tootajaid","Käive_kuus", alpha=0.2, xlim=(0,500),ylim=(0,1000000));