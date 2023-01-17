import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns
co2=pd.read_csv('../input/co2-ghg-emissionsdata/co2_emission.csv')
co2.shape
co2.head()
co2.info()
co2.drop('Code',axis=1,inplace=True)
co2.info()
co2.rename(columns={'Annual CO₂ emissions (tonnes )':'A_Co2_emissions(ton)'},inplace=True)
co2.info()
co2.head()
co2.Entity.unique()
co2.rename(columns={'Entity':'Country'},inplace=True)
co2.info()
co2.Year.unique()
co2.Year.min()
co2.Year.max()
co2['A_Co2_emissions(ton)'].max()
co2['A_Co2_emissions(ton)'].min()
condition=co2['A_Co2_emissions(ton)']>=0

df=co2[condition]
df['A_Co2_emissions(ton)'].min()
df['A_Co2_emissions(ton)'].max()
df.info()
sns.distplot(df['A_Co2_emissions(ton)'],kde=False)

plt.show()
df.groupby('Country')['A_Co2_emissions(ton)'].sum().sort_values(ascending=False)
df.groupby('Country')['A_Co2_emissions(ton)'].sum().sort_values(ascending=False)[1:6]
plt.style.use('seaborn')

df.groupby('Country')['A_Co2_emissions(ton)'].sum().sort_values(ascending=False)[1:6].plot(kind='bar')

plt.title('Top 5 Countries in co2 emission from 1751 to 2017',fontweight='bold',fontsize=14)

plt.ylabel('CO2 Emissions in Ton')
p1_c1=df['Year']<=1800
period1=df[p1_c1]
p2_c1=df['Year']<=1850

p2_c2=df['Year']>1800
period2=df[p2_c1&p2_c2]
p3_c1=df['Year']<=1900

p3_c2=df['Year']>1850
period3=df[p3_c1&p3_c2]
p4_c1=df['Year']<=1950

p4_c2=df['Year']>1900
period4=df[p4_c1&p4_c2]
p5_c1=df['Year']<=2000

p5_c2=df['Year']>1950
period5=df[p5_c1&p5_c2]
p6_c1=df['Year']>2000
period6=df[p6_c1]
sns.distplot(period1['A_Co2_emissions(ton)'],kde=False).set_title('Co2 emissions from 1751 to 1800 Histogram')

plt.ylabel('CO2 emissions in Tonnes')

plt.show()
period1.groupby('Country')['A_Co2_emissions(ton)'].sum().sort_values(ascending=False)
plt.style.use('seaborn')

period1.groupby('Country')['A_Co2_emissions(ton)'].sum().sort_values(ascending=False)[1:11].plot(kind='bar')

plt.ylabel('CO2 Emissions in tonnes')

plt.title('Co2 emissions from year 1751 to 1800 by country',fontweight='bold')
co2_1751_to1800=period1['A_Co2_emissions(ton)'].sum()

period1['A_Co2_emissions(ton)'].sum()
sns.distplot(period2['A_Co2_emissions(ton)'],kde=False).set_title('Co2 emissions from 1801 to 1850 Histogram')

plt.ylabel('CO2 Emissions per tonnes')

plt.show()
period2.groupby('Country')['A_Co2_emissions(ton)'].sum().sort_values(ascending=False)
plt.style.use('seaborn')

period2.groupby('Country')['A_Co2_emissions(ton)'].sum().sort_values(ascending=False)[1:11].plot(kind='bar')

plt.title('Co2 emissions from year 1801 to 1850 by country',fontweight='bold')

plt.ylabel('CO2 Emissions in tonnes')

co2_1801_to1850=period2['A_Co2_emissions(ton)'].sum()

period2['A_Co2_emissions(ton)'].sum()
sns.distplot(period3['A_Co2_emissions(ton)'],kde=False).set_title('Co2 emissions from 1851 to 1900 Histogram')

plt.ylabel('CO2 Emissions per tonnes')

plt.show()
period3.groupby('Country')['A_Co2_emissions(ton)'].sum().sort_values(ascending=False)
plt.style.use('seaborn')

period3.groupby('Country')['A_Co2_emissions(ton)'].sum().sort_values(ascending=False)[1:11].plot(kind='bar')

plt.ylabel('Co2 emissions in ton')

plt.title('Co2 emissions from 1851 to 1900 per country',fontweight='bold')
co2_1851_to1900=period3['A_Co2_emissions(ton)'].sum()

period3['A_Co2_emissions(ton)'].sum()
sns.distplot(period4['A_Co2_emissions(ton)'],kde=False).set_title('Co2 emissions from 1901 to 1950 Histogram')

plt.show()
period4.groupby('Country')['A_Co2_emissions(ton)'].sum().sort_values(ascending=False)[:60]
plt.style.use('seaborn')

period4.groupby('Country')['A_Co2_emissions(ton)'].sum().sort_values(ascending=False)[1:11].plot(kind='bar')

plt.title('Top 10 coutries in emttiong co2 from 1901 to 1950',fontweight='bold')

plt.ylabel('CO2 Emissions in tonnes')
co2_1901_to1950=period4['A_Co2_emissions(ton)'].sum()

period3['A_Co2_emissions(ton)'].sum()
sns.distplot(period5['A_Co2_emissions(ton)'],kde=False).set_title('Co2 emissions from 1951 to 2000 Histogram')

plt.show()
period5.groupby('Country')['A_Co2_emissions(ton)'].sum().sort_values(ascending=False)[:60]
plt.style.use('seaborn')

period5.groupby('Country')['A_Co2_emissions(ton)'].sum().sort_values(ascending=False)[1:11].plot(kind='bar')

plt.title('Top 10 coutries in emttiong co2 from 1951 to 2000',fontweight='bold')

plt.ylabel('CO2 Emissions in tonnes')
co2_1951_to2000=period5['A_Co2_emissions(ton)'].sum()

period5['A_Co2_emissions(ton)'].sum()
sns.distplot(period6['A_Co2_emissions(ton)'],kde=False).set_title('Co2 emissions from 1951 to 2000 Histogram')

plt.show()
plt.style.use('seaborn')

period6.groupby('Country')['A_Co2_emissions(ton)'].sum().sort_values(ascending=False)[1:11].plot(kind='bar')

plt.title('Top 10 coutries in emttiong co2 from 2000 to 2017',fontweight='bold')

plt.ylabel('CO2 Emissions in tonnes')
co2_2000_to2017=period6['A_Co2_emissions(ton)'].sum()

period6['A_Co2_emissions(ton)'].sum()
plt.bar([1,2,3,4,5,6],[co2_2000_to2017,co2_1951_to2000,co2_1901_to1950,co2_1851_to1900,co2_1801_to1850,co2_1751_to1800]

       ,tick_label=['2000-2017','1951-2000','1950-1901','1851-1900','1801-1850','1751-1800'])

plt.xtitle=('Time Frames')

plt.ylabel=('CO2 Emissions in ton')

plt.title('CO2 Emissions over different periods of time',fontweight='bold')

plt.show()