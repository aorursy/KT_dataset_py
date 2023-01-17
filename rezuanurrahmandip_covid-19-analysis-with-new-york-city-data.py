import numpy as np 
import pandas as pd # library for data analsysis
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
import requests # library to handle requests
from pandas.io.json import json_normalize # tranform JSON file into a pandas dataframe

from matplotlib import pyplot
import matplotlib.cm as cm
import matplotlib.colors as colors

us_data=pd.read_csv('../input/uncover/UNCOVER/USAFacts/confirmed-covid-19-cases-in-us-by-state-and-county.csv')
us_data_ny=us_data[us_data['state_name']=='NY']
us_data_ny.drop(us_data_ny[us_data_ny.county_name=='Statewide Unallocated'].index, inplace=True)
us_data_ny.drop(us_data_ny[us_data_ny.county_name=='New York City Unallocated/Probable'].index, inplace=True)
columns=['state_fips','lat','long','geometry','county_fips']
us_data_ny.drop(columns, inplace=True, axis=1)
us_data_ny['date']=pd.to_datetime(us_data_ny['date'], format='%Y-%m-%d')
us_data_ny.set_index('date',inplace=True)
us_data_ny.head()
us_data_ny_albany=us_data_ny[us_data_ny['county_name']=='Albany County']
us_data_ny_allegany=us_data_ny[us_data_ny['county_name']=='Allegany County']
us_data_ny_bronx=us_data_ny[us_data_ny['county_name']=='Bronx County']
us_data_ny_cattaraugugus=us_data_ny[us_data_ny['county_name']=='Cattaraugus County']
us_data_ny_ontario=us_data_ny[us_data_ny['county_name']=='Ontario County']
us_data_ny_queens=us_data_ny[us_data_ny['county_name']=='Queens County']
us_data_ny_rockland=us_data_ny[us_data_ny['county_name']=='Rockland County']
us_data_ny_west=us_data_ny[us_data_ny['county_name']=='Westchester County']
us_data_ny_albany.plot.bar(figsize=(30,10))

pyplot.show()

us_data_ny_allegany.plot.bar(figsize=(30,10))

pyplot.show()
us_data_ny_bronx.plot.bar(figsize=(30,10))

pyplot.show()
us_data_ny_cattaraugugus.plot.bar(figsize=(30,10))

pyplot.show()
us_data_ny_ontario.plot.bar(figsize=(30,10))

pyplot.show()
us_data_ny_queens.plot.bar(figsize=(30,10))

pyplot.show()
us_data_ny_rockland.plot.bar(figsize=(30,10))

pyplot.show()
us_data_ny_west.plot.bar(figsize=(30,10))

pyplot.show()
scaler=us_data_ny_albany['confirmed']
scaler=scaler.shift(1,axis=0)
scaler=pd.DataFrame(scaler)
scaler=scaler.reset_index()
albany_n=us_data_ny_albany
albany_n=albany_n.reset_index()
albany_n['ratio']=albany_n['confirmed']/scaler['confirmed']
columns=['county_name','state_name','confirmed']
albany_n.drop(columns, inplace=True, axis=1)
albany_n.set_index('date',inplace=True)
albany_n['ratio']=albany_n['ratio'].fillna(0)
albany_n.plot.bar(figsize=(30,10))
pyplot.show()
scaler=us_data_ny_allegany['confirmed']
scaler=scaler.shift(1,axis=0)
scaler=pd.DataFrame(scaler)
scaler=scaler.reset_index()
allegany_n=us_data_ny_allegany
allegany_n=allegany_n.reset_index()
allegany_n['ratio']=allegany_n['confirmed']/scaler['confirmed']
columns=['county_name','state_name','confirmed']
allegany_n.drop(columns, inplace=True, axis=1)
allegany_n.set_index('date',inplace=True)
allegany_n['ratio']=allegany_n['ratio'].fillna(0)
allegany_n.plot.bar(figsize=(30,10))
pyplot.show()
scaler=us_data_ny_bronx['confirmed']
scaler=scaler.shift(1,axis=0)
scaler=pd.DataFrame(scaler)
scaler=scaler.reset_index()
bronx_n=us_data_ny_bronx
bronx_n=bronx_n.reset_index()
bronx_n['ratio']=bronx_n['confirmed']/scaler['confirmed']
columns=['county_name','state_name','confirmed']
bronx_n.drop(columns, inplace=True, axis=1)
bronx_n.set_index('date',inplace=True)
bronx_n['ratio']=bronx_n['ratio'].fillna(0)
bronx_n.plot.bar(figsize=(30,10))
pyplot.show()
scaler=us_data_ny_cattaraugugus['confirmed']
scaler=scaler.shift(1,axis=0)
scaler=pd.DataFrame(scaler)
scaler=scaler.reset_index()
cattaraugugus_n=us_data_ny_cattaraugugus
cattaraugugus_n=cattaraugugus_n.reset_index()
cattaraugugus_n['ratio']=cattaraugugus_n['confirmed']/scaler['confirmed']
columns=['county_name','state_name','confirmed']
cattaraugugus_n.drop(columns, inplace=True, axis=1)
cattaraugugus_n.set_index('date',inplace=True)
cattaraugugus_n['ratio']=cattaraugugus_n['ratio'].fillna(0)
cattaraugugus_n.plot.bar(figsize=(30,10))
pyplot.show()
scaler=us_data_ny_ontario['confirmed']
scaler=scaler.shift(1,axis=0)
scaler=pd.DataFrame(scaler)
scaler=scaler.reset_index()
ontario_n=us_data_ny_ontario
ontario_n=ontario_n.reset_index()
ontario_n['ratio']=ontario_n['confirmed']/scaler['confirmed']
columns=['county_name','state_name','confirmed']
ontario_n.drop(columns, inplace=True, axis=1)
ontario_n.set_index('date',inplace=True)
ontario_n['ratio']=ontario_n['ratio'].fillna(0)
ontario_n.plot.bar(figsize=(30,10))
pyplot.show()
scaler=us_data_ny_ontario['confirmed']
scaler=scaler.shift(1,axis=0)
scaler=pd.DataFrame(scaler)
scaler=scaler.reset_index()
ontario_n=us_data_ny_ontario
ontario_n=ontario_n.reset_index()
ontario_n['ratio']=ontario_n['confirmed']/scaler['confirmed']
columns=['county_name','state_name','confirmed']
ontario_n.drop(columns, inplace=True, axis=1)
ontario_n.set_index('date',inplace=True)
ontario_n['ratio']=ontario_n['ratio'].fillna(0)
ontario_n.plot.bar(figsize=(30,10))
pyplot.show()
scaler=us_data_ny_queens['confirmed']
scaler=scaler.shift(1,axis=0)
scaler=pd.DataFrame(scaler)
scaler=scaler.reset_index()
queens_n=us_data_ny_queens
queens_n=queens_n.reset_index()
queens_n['ratio']=queens_n['confirmed']/scaler['confirmed']
columns=['county_name','state_name','confirmed']
queens_n.drop(columns, inplace=True, axis=1)
queens_n.set_index('date',inplace=True)
queens_n['ratio']=queens_n['ratio'].fillna(0)
queens_n.plot.bar(figsize=(30,10))
pyplot.show()
scaler=us_data_ny_rockland['confirmed']
scaler=scaler.shift(1,axis=0)
scaler=pd.DataFrame(scaler)
scaler=scaler.reset_index()
rockland_n=us_data_ny_rockland
rockland_n=rockland_n.reset_index()
rockland_n['ratio']=rockland_n['confirmed']/scaler['confirmed']
columns=['county_name','state_name','confirmed']
rockland_n.drop(columns, inplace=True, axis=1)
rockland_n.set_index('date',inplace=True)
rockland_n['ratio']=rockland_n['ratio'].fillna(0)
rockland_n.plot.bar(figsize=(30,10))
pyplot.show()
scaler=us_data_ny_west['confirmed']
scaler=scaler.shift(1,axis=0)
scaler=pd.DataFrame(scaler)
scaler=scaler.reset_index()
west_n=us_data_ny_west
west_n=west_n.reset_index()
west_n['ratio']=west_n['confirmed']/scaler['confirmed']
columns=['county_name','state_name','confirmed']
west_n.drop(columns, inplace=True, axis=1)
west_n.set_index('date',inplace=True)
west_n['ratio']=west_n['ratio'].fillna(0)
west_n.plot.bar(figsize=(30,10))
pyplot.show()