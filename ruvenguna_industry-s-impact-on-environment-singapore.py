import matplotlib.pyplot as plt 
import pandas as pd
import numpy as np
from scipy.stats import ttest_ind

# Width = 16, Height = 6
DIMS=(16, 6)
#Import dataframes
sulphur_dioxide_df = pd.read_csv('../input/air-pollutant-sulphur-dioxide.csv')
pm10_df = pd.read_csv("../input/air-pollutant-particulate-matter-pm10.csv")
pm25_df = pd.read_csv("../input/air-pollutant-particulate-matter-pm2-5.csv")
ozone_df = pd.read_csv("../input/air-pollutant-ozone.csv")
nitrogen_dioxide_df = pd.read_csv("../input/air-pollutant-nitrogen-dioxide.csv")
carbon_monoxide_df = pd.read_csv("../input/air-pollutant-carbon-monoxide-2nd-maximum-8-hour-mean.csv")

#Combine all dataframes into 1 for ease of analysis
pollution_df_1 = pd.merge(carbon_monoxide_df, nitrogen_dioxide_df, on='year', how='outer')
pollution_df_2 = pd.merge(pollution_df_1, ozone_df, on='year', how='outer')
pollution_df_3 = pd.merge(pollution_df_2, pm25_df, on='year', how='outer')
pollution_df_4 = pd.merge(pollution_df_3, pm10_df, on='year', how='outer')
pollution_df = pd.merge(pollution_df_4, sulphur_dioxide_df, on='year', how='outer')

pollution_df.head(10)
pollution_df.describe()
maufacture_df =  pd.read_csv("../input/total-output-in-manufacturing-by-industry-annual.csv")
maufacture_df.head(10)
maufacture_df.describe()
flats_df =  pd.read_csv("../input/completion-status-of-hdb-residential-developments.csv")
flats_df.head(10)
flats_df.describe()
commercial_df = pd.read_csv("../input/completion-status-of-hdb-commercial-developments.csv")
commercial_df.head(10)
commercial_df.describe()
veh_df = pd.read_csv("../input/annual-motor-vehicle-population-by-vehicle-type.csv")
veh_df.head(10)
veh_df.describe()
#Remove NaN values
pol_df = pollution_df.dropna().sort_values('year', ascending=True)

#Select data from 2008 to 2014
year = range(2008, 2015)
pol_df = pol_df[pol_df['year'].isin(year)]

#Remove the year so that it isnt normalized
pol_df = pol_df.drop('year', axis=1)

#Mean normalization
pol_df=(pol_df-pol_df.mean())/pol_df.std()
pol_df['year'] = year

pol_df
#Variables to plot
Var_to_plot = ['carbon_monoxide_2nd_maximum_8hourly_mean','nitrogen_dioxide_mean', 'pm2.5_mean','ozone_4th_maximum_8hourly_mean',
               'pm10_2nd_maximum_24hourly_mean','sulphur_dioxide_mean']

#Draw plot
Indi_pol_plot = pol_df.plot(x='year', y = Var_to_plot, kind = 'line', grid = True, figsize=DIMS,
                        title = 'Individual Pollution in Singapore from 2008 to 2014')

#Graph formatting
Indi_pol_plot.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.show()
#Selecting years required
year = range(2008, 2015)
maufacture_df = maufacture_df[maufacture_df['year'].isin(year)]

#Selecting the industries
listtofind = ['Chemicals & Chemical Products', 'Computer, Electronic & Optical Products' ]
maufacture_df = maufacture_df[maufacture_df['level_2'].isin(listtofind)]

#Reformat the dataframe
maufacture_df =  maufacture_df.set_index(['year', 'level_2'])['value'].unstack()

#Mean normalization
maufacture_df=(maufacture_df-maufacture_df.mean())/maufacture_df.std()
maufacture_df['year'] = year
maufacture_df
#Draw plot
manu_graph = maufacture_df.plot(x='year', y=listtofind, kind = 'line', grid = True, figsize=DIMS, 
                         title = 'Manufacturing of different industries from 2008 to 2014')
Indi_pol_plot = pol_df.plot(x='year', y = Var_to_plot, kind = 'line', grid = True, figsize=DIMS,
                        title = 'Individual Pollution in Singapore from 2008 to 2014')

#Graph formatting
Indi_pol_plot.legend(loc='center left', bbox_to_anchor=(1, 0.5))
manu_graph.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.tight_layout()
plt.show()
#Create new dataframe
com_corr = pol_df.copy()

#Add in the electronics data
com_corr['Elec Corr'] = maufacture_df['Computer, Electronic & Optical Products'].tolist()

#Product correlation dataframe
com_corr.corr(method = 'spearman')
#Draw plot
manu_graph = maufacture_df.plot(x='year', y='Computer, Electronic & Optical Products', kind = 'line', grid = True)
pol_df.plot(x='year', y = 'carbon_monoxide_2nd_maximum_8hourly_mean', kind = 'line', grid = True, figsize=DIMS, ax=manu_graph,
                        title = 'Electronics Manufacturing Industry VS Carbon Monoxide Pollution')

#Graph formatting
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.tight_layout()
plt.show()
#Create new dataframe
chem_corr = pol_df.copy()

#Add in the chemical data
chem_corr['Chem Corr'] = maufacture_df['Chemicals & Chemical Products'].tolist()

#Product correlation dataframe
chem_corr.corr(method = 'spearman')
#Draw plot
manu_graph = maufacture_df.plot(x='year', y='Chemicals & Chemical Products', kind = 'line', grid = True)
pol_df.plot(x='year', y = 'ozone_4th_maximum_8hourly_mean', kind = 'line', grid = True, figsize=DIMS, ax=manu_graph,
                        title = 'Chemicals Manufacturing Industry VS Ozone Pollution')

#Graph formatting
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.tight_layout()
plt.show()
#Select the data we want
flats_df = flats_df[flats_df['financial_year'].isin(year)]
flats_df = flats_df[(flats_df["type"] != "DBSS") & (flats_df["status"] == 'Under Construction')]

#Change the data type
flats_df['no_of_units'] = flats_df['no_of_units'].apply(np.float)

#Mean normalization
flats_df['no_of_units']=(flats_df['no_of_units']-flats_df['no_of_units'].mean())/flats_df['no_of_units'].std()

flats_df
#Select the data we want
commercial_df = commercial_df[commercial_df['financial_year'].isin(year)]
commercial_df = commercial_df[(commercial_df["status"] == 'Under Construction') & (commercial_df['no_of_units'] != 0) & (commercial_df['type'].isin(['Shops, Lock-Up Shops and Eating Houses', 'Emporiums and Supermarkets']))]

#Reformat dataframe
comm_df =  commercial_df.set_index(['financial_year', 'type'])['no_of_units'].unstack().reset_index()

#Mean normalization
comm_df=(comm_df-comm_df.mean())/comm_df.std()

comm_df
property_df = pd.DataFrame()
property_df['year'] = year
property_df['HDB'] = flats_df['no_of_units'].tolist()
property_df['Emporiums and Supermarkets'] = comm_df['Emporiums and Supermarkets'].tolist()
property_df['Shops, Lock-Up Shops and Eating Houses'] = comm_df['Shops, Lock-Up Shops and Eating Houses'].tolist()

property_df
#Create new dataframe
housing_corr = pol_df.copy()

#Add in the housing data
housing_corr['HDB Corr'] = property_df['HDB'].tolist()
housing_corr['E&S Corr'] = property_df['Emporiums and Supermarkets'].tolist()
housing_corr['Shops Corr'] = property_df['Shops, Lock-Up Shops and Eating Houses'].tolist()

#Product correlation dataframe
housing_corr.corr(method = 'spearman')
#Draw plot
property_df_plot = property_df.plot(x='year', y=['HDB', 
                              'Emporiums and Supermarkets', 
                              'Shops, Lock-Up Shops and Eating Houses'], 
                              kind = 'line', grid = True, figsize = DIMS,
                              title = 'Under construction HDB from 2008 to 2014')
pol_df.plot(x='year', y = 'ozone_4th_maximum_8hourly_mean', kind = 'line', grid = True, figsize=DIMS, ax=property_df_plot,
                        title = 'Residencial and Commercial Development VS Ozone Pollution')

#Graph formatting
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.tight_layout()
plt.show()
#Select the data we need
veh_df = veh_df[veh_df['year'].isin(year)]

#Perform groupby
veh_df = veh_df.groupby('year').sum().reset_index()

#Mean normalization
veh_df['number']=(veh_df['number']-veh_df['number'].mean())/veh_df['number'].std()
veh_df.rename(columns = {'number':'Number of Vehicles'}, inplace = True)
veh_df
#Create new dataframe
veh_corr = pol_df.copy()

#Add in the housing data
veh_corr['Veh Corr'] = veh_df['Number of Vehicles'].tolist()

#Product correlation dataframe
veh_corr.corr(method = 'spearman')
veh_graph = veh_df.plot(x='year', y='Number of Vehicles', kind = 'line', grid = True,
                    title = 'Vehicle Population from 2008 to 2014')
pol_df.plot(x='year', y = ['nitrogen_dioxide_mean', 'ozone_4th_maximum_8hourly_mean'], kind = 'line', grid = True, figsize=DIMS, ax=veh_graph,
                        title = 'Vehicle Population VS Ground-Level Ozone and Nitrogen Dioxide Pollution')

plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.show()
