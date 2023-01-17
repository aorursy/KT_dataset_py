# Usual library imports.... more later 

import numpy as np

import pandas as pd

import datetime

from datetime import date

import matplotlib.pyplot as plt

import matplotlib as mpl

import matplotlib.dates as mdates

mpl.rcdefaults()



plt.style.use('seaborn-whitegrid')



# Set precision to two decimals

pd.set_option("display.precision", 0)



# Define date format for charts like Apr 16 or Mar 8

my_date_fmt = mdates.DateFormatter('%b %e')

# Define date format for charts with only the day number

my_date_fmt_d = mdates.DateFormatter('%e')
!ls ../input
# Data loading

df = pd.read_csv("../input/covid19_casos_confirmados_paises_29052020.csv", sep=',')



df.head()
# Get last date for charts

last_date = str(df.iloc[-1,0])

print('Last date in the set: ' + last_date)
# Convert FECHA column to datetime format and set as index

df['FECHA'] = pd.to_datetime(df['FECHA'], format='%d/%m/%Y')

df.set_index('FECHA', inplace=True)
df.fillna(value=0, inplace=True)

df
# Plot total number of cases per country

plt.figure(figsize=(10,5))

plt.ylabel('Cases')

plt.title('Covid-19 confirmed cases per country as of ' + last_date, fontsize='large')

for x in df.columns:

  plt.bar(x, df[x].tail(1), alpha=0.5),
# Plot total number of cases per country, excluding USA

plt.figure(figsize=(10,5))

plt.ylabel('Cases')

plt.title('Covid-19 confirmed cases per country as of ' + last_date, fontsize='large')

for x in df.columns:

  if x != 'USA':

    plt.bar(x, df[x].tail(1), alpha=0.5)
# Countries populations

# Source: https://www.worldometers.info/world-population/population-by-country/



pop = {}



pop['ITALY'] = 60461826

pop['SPAIN'] = 46754778

pop['FRANCE'] = 65273511

pop['GERMANY'] = 83783942

pop['UK'] = 67886011

pop['AUSTRALIA'] = 25499884

pop['CANADA'] = 37600000

pop['BRAZIL'] = 212559417

pop['IRAN'] = 83992949

pop['USA'] = 331002651
# Containment measures dates in the European countries

# Source: https://www.dw.com/en/coronavirus-what-are-the-lockdown-measures-across-europe/a-52905137



cm = {}



cm['ITALY'] = datetime.date(2020, 3, 9)

cm['SPAIN'] = datetime.date(2020, 3, 14)

cm['FRANCE'] = datetime.date(2020, 3, 17)

cm['GERMANY'] = datetime.date(2020, 3, 22)

cm['UK'] = datetime.date(2020, 3, 23)
# Other significant dates



OchoM = datetime.date(2020, 3, 8) # International Woman's Day, 600K people gathered in the streets of Spain

MWC_cancelled = datetime.date(2020, 2, 12) # Date of cancellation of MWC Barcelona 2020

community_transmission = datetime.date(2020, 3, 2) # WHO declared covid-19 community transmission

paris_marathon_cancelled = datetime.date(2020, 3, 5)

cases_100K = datetime.date(2020, 3, 7) # 100K cases of covid-19 worldwide

pandemic_date = datetime.date(2020, 3, 11) # WHO declares covid-19 a pandemic
# Calculate nbr of cases per million people

df_per_million = pd.DataFrame(columns=df.columns)

for x in df_per_million.columns:

  df_per_million[x] = 1000000 * df[x].tail(1) // pop[x]



df_per_million
# Plot total number of cases per country, excluding USA

plt.figure(figsize=(10,5))

plt.ylabel('Cases per million')

plt.title('Covid-19 confirmed cases per million people as of ' + last_date, fontsize='large')

for x in df_per_million.columns:

  plt.bar(x, df_per_million[x], alpha=0.5)
# Plot confirmed cases for all countries

fig, ax = plt.subplots(figsize=(10,5))

plt.ylabel('Cases')

plt.title('Evolution of Covid-19 number of cases', fontsize='large')

ax.xaxis.set_major_formatter(my_date_fmt)



countries = df.columns

for x in countries:

  plt.plot(df.index, df[x], label=x, linewidth=1.5)



fig.autofmt_xdate(rotation=30, ha='right')

plt.legend(loc='upper left', fontsize='small')

plt.show()
# Plot confirmed cases for all countries

fig, ax = plt.subplots(figsize=(10,5))

plt.ylabel('Cases')

plt.title('Evolution of Covid-19 number of cases', fontsize='large')

ax.xaxis.set_major_formatter(my_date_fmt)



countries = df.columns

for x in countries:

  if x != 'USA':

    plt.plot(df.index, df[x], label=x, linewidth=1.5)



fig.autofmt_xdate(rotation=30, ha='right')

plt.legend(loc='upper left', fontsize='small')

plt.show()
# Split data into two sets

df_EU = df[['SPAIN', 'ITALY', 'FRANCE', 'UK', 'GERMANY']]

df_world = df[['AUSTRALIA', 'CANADA', 'BRAZIL', 'IRAN', 'USA']]
# Plot confirmed cases for the world countries set, excluding USA

fig, ax = plt.subplots(figsize=(10,5))

plt.ylabel('Cases')

plt.title('Evolution of covid-19 cases per country', fontsize='large')

ax.xaxis.set_major_formatter(my_date_fmt)



countries = df_world.columns

for x in countries:

  if x != 'USA':

    plt.plot(df_world.index, df_world[x], linewidth=1.5, label=x)



plt.legend(loc='upper left', fontsize='small')

fig.autofmt_xdate(rotation=30, ha='right')

plt.show()
# Plot confirmed cases for the EU set

fig, ax = plt.subplots(figsize=(10,5))

plt.ylabel('Cases')

plt.title('Evolution of covid-19 cases per country', fontsize='large')

ax.xaxis.set_major_formatter(my_date_fmt)



countries = df_EU.columns

for x in countries:

  plt.plot(df_EU.index, df_EU[x], linewidth=1.5, label=x)



plt.legend(loc='upper left', fontsize='small')

fig.autofmt_xdate(rotation=30, ha='right')

plt.show()
# Spain vs. Italy

fig, ax = plt.subplots(figsize=(10,5))

ax.xaxis.set_major_formatter(my_date_fmt_d)

ax.xaxis.set_major_locator(plt.MultipleLocator(2))



plt.plot(df['ITALY'].tail(100).head(50), color='lime', linewidth=5, alpha=0.75, label='ITALY')

plt.axvline(cm['ITALY'], color='lime', linestyle='-.', linewidth=2, label='Lockdown')

plt.plot(df['SPAIN'].tail(100).head(50), color='coral', linewidth=5, alpha=0.75, label='SPAIN')

plt.axvline(cm['SPAIN'], color='coral', linestyle='-.', linewidth=2, label='Lockdown')

for i in range(-8, 0, 2):

  plt.plot(df['SPAIN'].tail(100).head(50).shift(i), linewidth=1, marker='+', markersize=3, alpha=0.5, label=i)



plt.legend(loc='lower right', fontsize='small')

plt.show()
# Germany vs. Italy

fig, ax = plt.subplots(figsize=(10,5))

ax.xaxis.set_major_formatter(my_date_fmt_d)

ax.xaxis.set_major_locator(plt.MultipleLocator(2))



plt.plot(df['ITALY'].tail(100).head(50), color='lime', linewidth=5, alpha=0.75, label='ITALY')

plt.axvline(cm['ITALY'], color='lime', linestyle='-.', linewidth=2, label='Lockdown')

plt.plot(df['GERMANY'].tail(100).head(50), color='brown', linewidth=5,alpha=0.75, label='GERMANY')

plt.axvline(cm['GERMANY'], color='brown', linestyle='-.', linewidth=2, label='Social distancing')

for i in range(-10, -2, 2):

  plt.plot(df['GERMANY'].tail(100).head(50).shift(i), linewidth=1, marker='+', markersize=3, alpha=0.5, label=i)



plt.legend(loc='lower right', fontsize='small')

plt.show()
# France vs. Italy

fig, ax = plt.subplots(figsize=(10,5))

ax.xaxis.set_major_formatter(my_date_fmt_d)

ax.xaxis.set_major_locator(plt.MultipleLocator(2))



plt.plot(df['ITALY'].tail(100).head(50), color='lime', linewidth=5, alpha=0.75, label='ITALY')

plt.axvline(cm['ITALY'], color='lime', linestyle='-.', linewidth=2, label='Lockdown')

plt.plot(df['FRANCE'].tail(100).head(50), color='blue', linewidth=5, alpha=0.75, label='FRANCE')

plt.axvline(cm['FRANCE'], color='blue', linestyle='-.', linewidth=2, label='Lockdown')

for i in range(-14, -6, 2):

  plt.plot(df['FRANCE'].tail(100).head(50).shift(i), linewidth=1, marker='+', markersize=3, alpha=0.5, label=i)



plt.legend(loc='lower right', fontsize='small')

plt.show()
# UK vs. Italy

fig, ax = plt.subplots(figsize=(10,5))

ax.xaxis.set_major_formatter(my_date_fmt_d)

ax.xaxis.set_major_locator(plt.MultipleLocator(2))



plt.plot(df['ITALY'].tail(100).head(50), color='lime', linewidth=5, label='ITALY')

plt.axvline(cm['ITALY'], color='lime', linestyle='-.', linewidth=2, label='Lockdown')

plt.plot(df['UK'].tail(100).head(50), color='navy', linewidth=5, label='UK')

plt.axvline(cm['UK'], color='navy', linestyle='-.', linewidth=2, label='Lockdown')

for i in range(-18, -10, 2):

  plt.plot(df['UK'].tail(100).head(50).shift(i), linewidth=1, marker='x', markersize=3, alpha=0.5, label=i)



plt.legend(loc='lower right', fontsize='small')

plt.show()
fig, ax = plt.subplots(figsize=(10,5))

plt.title('Evolution of covid-19 cases per country (aligned to Italy´s start date)', fontsize='large')

plt.ylabel('Cases')

ax.xaxis.set_major_formatter(my_date_fmt_d)

ax.xaxis.set_major_locator(plt.MultipleLocator(2))



plt.plot(df['ITALY'].tail(100).head(60), color='lime', linewidth=1.5, label='ITALY')

plt.plot(df['SPAIN'].tail(100).head(60).shift(-6), color='coral', linewidth=1.5, label='SPAIN')

plt.plot(df['GERMANY'].tail(100).head(60).shift(-6), color='brown', linewidth=1.5, label='GERMANY')

plt.plot(df['FRANCE'].tail(100).head(60).shift(-10), color='blue', linewidth=1.5, label='FRANCE')

plt.plot(df['UK'].tail(100).head(60).shift(-16), color='navy', linewidth=1.5, label='UK')



plt.legend(loc='upper left', fontsize='small')

plt.show()
# Plots of cummulative confirmed cases

fig, ax = plt.subplots(5,1, figsize=(8, 10))

fig.subplots_adjust(top=0.93)

i = 0

for x in ['ITALY', 'GERMANY', 'SPAIN', 'FRANCE', 'UK']:

  ax[i].set_title(x, fontsize='medium')

  ax[i].xaxis.set_major_formatter(my_date_fmt)

  ax[i].xaxis.set_major_locator(plt.MultipleLocator(12))  

  ax[i].bar(df.index, df[x], alpha=0.25)

  ax[i].plot(df.index, df[x].rolling(window=7).mean(), linewidth=2, label='ma-7', color='orange')

  ax[i].legend(loc='upper left', fontsize='medium')

  i = i + 1



fig.suptitle('Evolution of covid-19 confirmed cases (1 of 2)', fontsize='large')  



fig.autofmt_xdate(rotation=30, ha='right')

plt.show()
# Plots of cummulative confirmed cases

fig, ax = plt.subplots(5,1, figsize=(8, 10))

fig.subplots_adjust(top=0.93)

i = 0

for x in ['IRAN', 'AUSTRALIA', 'CANADA', 'USA', 'BRAZIL']:

  ax[i].set_title(x, fontsize='medium')

  ax[i].xaxis.set_major_formatter(my_date_fmt)

  ax[i].xaxis.set_major_locator(plt.MultipleLocator(12))

  ax[i].bar(df.index, df[x], alpha=0.25)

  ax[i].plot(df.index, df[x].rolling(window=7).mean(), linewidth=2, label='ma-7', color='orange')

  ax[i].legend(loc='upper left', fontsize='medium')

  i = i + 1



fig.suptitle('Evolution of covid-19 confirmed cases (2 of 2)', fontsize='large')  

fig.autofmt_xdate(rotation=30, ha='right')

plt.show()
# Create new dataframe with confirmed cases daily variation, calculated as:

# COUNTRY_D[t] = COUNTRY[t] - COUNTRY[t-1]



ddf = pd.DataFrame(index=df.index, columns=df.columns)



for x in ddf.columns:

  ddf[x] = df[x].diff()



ddf.fillna(value=0, inplace=True)

ddf
# Plots of daily variation / grotwh rate of confirmed cases

fig, ax = plt.subplots(5,1, figsize=(8, 10))

fig.subplots_adjust(top=0.93)

i = 0

for x in ['ITALY', 'GERMANY', 'SPAIN', 'FRANCE', 'UK']:

  ax[i].set_title(x, fontsize='medium')

  ax[i].xaxis.set_major_formatter(my_date_fmt)

  ax[i].xaxis.set_major_locator(plt.MultipleLocator(12))

  ax[i].bar(ddf.index, ddf[x], alpha=0.25)

  ax[i].plot(ddf.index, ddf[x].rolling(window=7).mean(), linewidth=2, label='ma-7', color='orange')

  ax[i].legend(loc='upper left', fontsize='medium')

  i = i + 1



fig.suptitle('Daily variation of covid-19 confirmed cases (1 of 2)', fontsize='large')  

fig.autofmt_xdate(rotation=30, ha='right')

plt.show()
# Plots of daily variation / growth rate of confirmed cases

fig, ax = plt.subplots(5,1, figsize=(8, 10))

fig.subplots_adjust(top=0.94)

i = 0

for x in ['IRAN', 'AUSTRALIA', 'CANADA', 'USA', 'BRAZIL']:

  ax[i].set_title(x, fontsize='medium')

  ax[i].xaxis.set_major_formatter(my_date_fmt)

  ax[i].xaxis.set_major_locator(plt.MultipleLocator(12))

  ax[i].bar(ddf.index, ddf[x], alpha=0.25)

  ax[i].plot(ddf.index, ddf[x].rolling(window=7).mean(), linewidth=2, label='ma-7', color='orange')

  ax[i].legend(loc='upper left', fontsize='medium')

  i = i + 1



fig.suptitle('Daily variation of covid-19 confirmed cases (2 of 2)', fontsize='large')  

fig.autofmt_xdate(rotation=30, ha='right')

plt.show()
df.tail(84).head(21).T
# Spain vs. Italy: contextualized comparison of daily variation of cases

fig, ax = plt.subplots(figsize=(10,5))

ax.xaxis.set_major_formatter(my_date_fmt)

ax.xaxis.set_major_locator(plt.MultipleLocator(4))

plt.title('Daily variation of covid-19 confirmed cases', fontsize='large')



plt.plot(ddf['ITALY'].tail(100).head(55), color='lime', linewidth=2, label='ITALY')

plt.axvline(cm['ITALY'], color='lime', linestyle='-.', label='Lockdown March 9')

plt.axvspan(cm['ITALY'], cm['ITALY'] + datetime.timedelta(12) , ymin=0, ymax=1, color='lime', linewidth=2, alpha=0.2, label='High growth window')



plt.plot(ddf['SPAIN'].tail(100).head(55), color='coral', linewidth=2, label='SPAIN')

plt.axvline(cm['SPAIN'], color='coral', linestyle='-.', label='Lockdown March 14')

plt.axvspan(cm['SPAIN'], cm['SPAIN'] + datetime.timedelta(11) , ymin=0, ymax=1, color='coral', linewidth=2, alpha=0.2, label='High growth window')



plt.legend(loc='upper left', fontsize='small')

fig.autofmt_xdate(rotation=30, ha='right')

plt.show()
# Let's review once again the numbers 

df[['SPAIN', 'ITALY']].tail(84).head(21).T
# And the growth rate

ddf[['SPAIN', 'ITALY']].tail(84).head(21).T
# Plot of Italy's high infection window

fig, ax = plt.subplots(figsize=(10,5))

ax.xaxis.set_major_formatter(my_date_fmt)

ax.xaxis.set_major_locator(plt.MultipleLocator(4))

plt.title('Daily variation of covid-19 confirmed cases in Italy', fontsize='large')



plt.plot(ddf['ITALY'].tail(100).head(55), color='lime', linewidth=2)

plt.axvspan(cm['ITALY'] - datetime.timedelta(14), cm['ITALY'] , ymin=0, ymax=1, color='magenta', alpha=0.2, label='High infection window')

plt.axvspan(cm['ITALY'], cm['ITALY'] + datetime.timedelta(12) , ymin=0, ymax=1, color='lime', alpha=0.2, label='High growth window')



plt.axvline(community_transmission, color='red', linestyle=':', lw=3, label='WHO admits community transmission')

plt.axvline(paris_marathon_cancelled, color='yellow', linestyle=':', lw=3, label='Paris marathon cancelled')

plt.axvline(cases_100K, color='navy', linestyle='dotted', lw=3, label='100K cases worldwide')

plt.axvline(pandemic_date, color='blue', linestyle='-.', lw=3, label='Who declares pandemic')



plt.legend(loc='lower right', fontsize='small')

fig.autofmt_xdate(rotation=30, ha='right')

plt.show()
# Plot of Spain's high infection window

fig, ax = plt.subplots(figsize=(10,5))

ax.xaxis.set_major_formatter(my_date_fmt)

ax.xaxis.set_major_locator(plt.MultipleLocator(4))

plt.title('Daily variation of covid-19 confirmed cases in Spain', fontsize='large')



plt.plot(ddf['SPAIN'].tail(100).head(55), color='coral', linewidth=2)

plt.axvspan(cm['SPAIN']- datetime.timedelta(14), cm['SPAIN'], ymin=0, ymax=1, color='magenta', alpha=0.2, label='High infection window')

plt.axvspan(cm['SPAIN'], cm['SPAIN'] + datetime.timedelta(11) , ymin=0, ymax=1, color='coral', linewidth=2, alpha=0.2, label='High growth window')



plt.axvline(community_transmission, color='red', linestyle=':', lw=3, label='WHO admits community transmission')

plt.axvline(paris_marathon_cancelled, color='yellow', linestyle=':', lw=3, label='Paris marathon cancelled')

plt.axvline(cases_100K, color='navy', linestyle='dotted', lw=3, label='100K cases worldwide')

plt.axvline(pandemic_date, color='blue', linestyle='-.', lw=3, label='WHO declares pandemic')



plt.legend(loc='lower right', fontsize='small')

fig.autofmt_xdate(rotation=30, ha='right')

plt.show()