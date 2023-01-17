#!pip install fastai2 -q
import pandas as pd
import matplotlib.pyplot as plt
#from fastai2.tabular.all import *
from ipywidgets import interact, interact_manual , interactive

pd.set_option('display.max_rows', 200)
from matplotlib.pyplot import figure
from pylab import rcParams

# Data extracted from:
#URL = "https://www.kaggle.com/sudalairajkumar/novel-corona-virus-2019-dataset#covid_19_data.csv"
filename = "covid_19_data.csv"
df = pd.read_csv('/kaggle/input/novel-corona-virus-2019-dataset/covid_19_data.csv',index_col=0,parse_dates=[0])
country_list = df['Country/Region'].unique().tolist()
country_list.sort()
#  open date field
#df = add_datepart(df, 'ObservationDate',drop=False)
#df['ObservationDate'] = pd.to_datetime(df['ObservationDate'], format='%M/%d/%Y')   #  '%M/%d/%Y'
country_list = ['Brazil','US','France', 'Italy','UK','Belgium', 'Canada'] + country_list  ## put these countries first in the roll-down list

filename1= "population.csv"
dpop = pd.read_csv('/kaggle/input/worldpopulation/population.csv',index_col=0,sep=';')

### data on population by country by age
filename2 = '/kaggle/input/over65/population_by_age.csv'
d65 = pd.read_csv('/kaggle/input/over65/population_by_age.csv',index_col=0,sep=';')
# merge corona virus data with country population 
#df = pd.merge(df,dpop,left_on="Country/Region",right_on="Country")

##  last date with data
last_date=df['ObservationDate'].tail(1)
last_date =str(last_date.values)[2:12]
last_date
from matplotlib.pyplot import figure
figure(num=None, figsize=(10, 8), dpi=80, facecolor='w', edgecolor='k')
from pylab import rcParams
rcParams['figure.figsize'] = 8,6

def plot_by_country(country=country_list):
  df_filtered = df[df['Country/Region'] == country].copy()

  #  CONFIRMED - filter and calculate daily values 
  res_confirmed=df_filtered[['Confirmed','Deaths']].groupby([df_filtered['ObservationDate']]).sum()
  res_acc = res_confirmed[res_confirmed['Confirmed'] > 100]
  res_tot = res_acc.diff()
  
  ratio = res_acc['Deaths']/res_acc['Confirmed']   ### ratio of accumulated data,not daily
 
  res_tot['Deaths_mov']    = res_tot['Deaths'].rolling(5).mean()
  res_tot['Confirmed_mov'] = res_tot['Confirmed'].rolling(5).mean()
  res_tot['Ratio ']        = ratio    
  res_tot = res_tot.dropna().reset_index()
  res_tot = res_tot.reset_index()
  res_tot.columns = ['Day', 'Date', 'Confirmed','Deaths','Deaths_mov','Confirmed_mov', 'Ratio']
  
  ax= res_tot.plot('Day','Confirmed_mov',legend=False,color='b')
  ax.set_ylabel('Confirmed Cases')
  
  ax2 = ax.twinx()
  ax2=res_tot.plot('Day','Deaths_mov',kind='line',legend=False,ax=ax2,color='r') 
  ax2.set_ylabel('Deaths Moving Average ')
  
  ax.figure.legend(loc='upper right')
  plt.title(f' LAST DATE {last_date}  -DEATHS/CONFIRM. NEW CASES - MOVING AVERAGE - 5 DAYS   ',loc='right')
  plt.figtext(0.99, 0.01, "x-axis represent days since first time 100 cases were detected", horizontalalignment='right')
  plt.show()

  print(res_tot)
im = interact_manual(plot_by_country)  
display(im)
def extract_data(country):
  df_filtered = df[df['Country/Region'] == country].copy()
  population = dpop[dpop.index== country].values[0][0]
  ratio_65 =  d65[d65.index==country]['Ratio Old'].values[0]
  #  CONFIRMED - filter and calculate daily values 
  res_confirmed=df_filtered[['Confirmed','Deaths']].groupby([df_filtered['ObservationDate']]).sum()
  res_acc = res_confirmed[res_confirmed['Confirmed'] > 100]
  res_tot = res_acc.diff()
  
  ratio = res_acc['Deaths']/res_acc['Confirmed']   ### ratio of accumulated data,not daily
  res_tot['Deaths_acc']    =  res_acc['Deaths']
  res_tot['Deaths_mov']    = res_tot['Deaths'].rolling(7).mean()
  res_tot['Confirmed_mov'] = res_tot['Confirmed'].rolling(7).mean()
  res_tot['Ratio ']        = ratio
  res_tot['Deaths_mov_capita'] = res_tot['Deaths_mov'] * 1000000 / population
  res_tot['Deaths_mov_old_capita'] = res_tot['Deaths_mov'] * 1000000 / (population * ratio_65) 
  res_tot['Confirmed_mov_capita'] = res_tot['Confirmed_mov'] * 1000000 / population
  res_tot['Deaths_acc_capita'] = res_tot['Deaths_acc'] * 1000000 / population  
  res_tot = res_tot.dropna()
  return res_tot , population , ratio_65
  
### PLOT CHART OF DEATHS PER MILLION INHABITANTS - MOVING AVERAGE - 5 DAYS
figure(num=None, figsize=(12, 10), dpi=80, facecolor='w', edgecolor='k')
fig, ax = plt.subplots()
for country in  ['Brazil','US','France', 'Italy','UK','Belgium', 'Canada']:
  series,population,over65 = extract_data(country)
  x = range(len(series))
  ax.plot(x,series['Deaths_mov_capita'],label=country)
  #print(country, series[['Deaths','Deaths_mov_capita']].tail(1),population)
plt.legend(loc='upper left')
plt.title(f'ÚLTIMA DATA {last_date}  -MORTES POR MILHÃO DE HABITANTES - MÉDIA MÓVEL - 7 DAYS   ',loc='right')
#plt.title(f'DEATHS PER MILLION INHABITANTS - MOVING AVERAGE - 5 DAYS   ',loc='right')
plt.figtext(0.99, 0.01, "eixo x representa o número de dias desde que os primeiros 100 casos foram detectados", horizontalalignment='right')
plt.show()
### PLOT CHART OF DEATHS PER MILLION INHABITANTS  OVER 65 - MOVING AVERAGE - 5 DAYS
figure(num=None, figsize=(12, 10), dpi=80, facecolor='w', edgecolor='k')
fig, ax = plt.subplots()
for country in  ['Brazil','US','France', 'Italy','UK','Belgium', 'Canada']:
  series,population,over65 = extract_data(country)
  x = range(len(series))
  ax.plot(x,series['Deaths_mov_old_capita'],label=country)
  #print(country, series[['Deaths','Deaths_mov_capita']].tail(1),population)
plt.legend(loc='upper left')
plt.title(f' LAST DATE {last_date}- DEATHS PER MILLION INHABITANTS  OVER 65 - MOVING AVERAGE - 5 DAYS',loc='right')
#plt.title(f' DEATHS PER MILLION INHABITANTS  OVER 65 - MOVING AVERAGE - 5 DAYS',loc='right')
plt.figtext(0.99, 0.01, "x-axis represent days since first time 100 cases were detected", horizontalalignment='right')
plt.show()
### PLOT CHART OF ACCUMULATED DEATHS PER MILLION INHABITANTS
figure(num=None, figsize=(12, 10), dpi=80, facecolor='w', edgecolor='k')
fig, ax = plt.subplots()
for country in ['Brazil','US','France', 'Italy','UK','Belgium', 'Canada']:
  series,population,over65 = extract_data(country)
  x = range(len(series))
  ax.plot(x,series['Deaths_acc_capita'],label=country)

  #ax2 = ax.twinx()
  #ax2=res_tot.plot('Day','Deaths_mov',kind='line',legend=False,ax=ax2,color='r') 
  #plt.plot(x[-1],over65,ax=ax2,color='bo')
  #print(country, series[['Deaths','Deaths_mov_capita']].tail(1),population)
plt.legend(loc='upper left')
plt.title(f'ÚLTIMA DATA {last_date} - MORTES POR MILHÃO DE HABITANTES - MÉDIA MÓVEL - 7 DAYS   ',loc='right')
plt.figtext(0.99, 0.01, "eixo x representa o número de dias desde que os primeiros 100 casos foram detectados", horizontalalignment='right')
plt.show()

plt.title(f'ÚLTIMA DATA {last_date} - MORTES POR MILHÃO DE HABITANTES - MÉDIA MÓVEL - 7 DAYS   ',loc='right')
#plt.title(f'DEATHS PER MILLION INHABITANTS - MOVING AVERAGE - 5 DAYS   ',loc='right')
plt.figtext(0.99, 0.01, "eixo x representa o número de dias desde que os primeiros 100 casos foram detectados", horizontalalignment='right')
plt.show()
###  PLOT CHART OF ACCUMULATED DEATHS
figure(num=None, figsize=(12, 10), dpi=80, facecolor='w', edgecolor='k')
fig, ax = plt.subplots()
for country in  ['Belgium','Spain','Israel','Brazil','US','France', 'Italy', 'South Korea', 'Germany']:
  series,population,ove65 = extract_data(country)
  x = range(len(series))
  ax.plot(x,series['Deaths_acc'],label=country)
  #print(country, series[['Deaths','Deaths_mov_capita']].tail(1),population)
plt.legend(loc='upper left')
plt.title(f' LAST DATE {last_date}  -------ACCUMULATED DEATHS     ',loc='right')
plt.figtext(0.99, 0.01, "x-axis represent days since first time 100 cases were detected", horizontalalignment='right')
plt.show()
### standard deviation is not a good indicator for fat-tail distributions

def extract_std(country):
  df_filtered = df[df['Country/Region'] == country].copy()
  population = dpop[dpop.index== country].values[0][0]
  ratio_65 =  d65[d65.index==country]['Ratio Old'].values[0]
  #  CONFIRMED - filter and calculate daily values 
  res_confirmed=df_filtered[['Confirmed','Deaths']].groupby([df_filtered['ObservationDate']]).sum()
  res_acc = res_confirmed[res_confirmed['Confirmed'] > 100]
  res_tot = res_acc.diff()
  
  ratio = res_acc['Deaths']/res_acc['Confirmed']   ### ratio of accumulated data,not daily
  res_tot['Deaths_acc']    =  res_acc['Deaths']
  res_tot['Deaths_mov']    = res_tot['Deaths'].rolling(5).mean()
  
  res_tot['Confirmed_mov'] = res_tot['Confirmed'].rolling(5).mean()
  res_tot['Ratio ']        = ratio
  res_tot['Deaths_mov_capita'] = res_tot['Deaths_mov'] * 1000000 / population
  std = res_tot['Deaths_mov_capita'].std()
  res_tot['Deaths_mov_old_capita'] = res_tot['Deaths_mov'] * 1000000 / (population * ratio_65) 
  res_tot['Confirmed_mov_capita'] = res_tot['Confirmed_mov'] * 1000000 / population
  res_tot['Deaths_acc_capita'] = res_tot['Deaths_acc'] * 1000000 / population  
  res_tot = res_tot.dropna()
  return res_tot , population , ratio_65 , std
### PLOT CHART OF DEATHS PER MILLION INHABITANTS - MOVING AVERAGE - 5 DAYS
figure(num=None, figsize=(12, 10), dpi=80, facecolor='w', edgecolor='k')
fig, ax = plt.subplots()
#for country in  ['Belgium','Spain','Israel','Brazil','US','France', 'Italy','France', 'Germany']:
for country in  ['Sweden','Belgium','Spain','US','France', 'Italy','France', 'Germany','US']: 
#for country in  ['Belgium', 'Italy']:      
  series,population,over65,std = extract_std(country)
  x = range(len(series[:76]))
  ax.plot(x,series['Deaths_mov_capita'][:76],label=country)
  print(country, std,len(series))
plt.legend(loc='upper left')
plt.title(f'LAST DATE {last_date}  -------DEATHS PER MILLION INHABITANTS - MOVING AVERAGE - 5 DAYS   ',loc='right')
#plt.title(f'DEATHS PER MILLION INHABITANTS - MOVING AVERAGE - 5 DAYS   ',loc='right')
plt.figtext(0.99, 0.01, "x-axis represent days since first time 100 cases were detected", horizontalalignment='right')
plt.show()