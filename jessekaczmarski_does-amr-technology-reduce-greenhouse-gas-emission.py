#Initial Packages

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns



#Data import

df_base = pd.read_csv('../input/ny-energy-and-water-data-disclosure-local-law-84/energy-and-water-data-disclosure-for-local-law-84-2014-data-for-calendar-year-2013.csv')

df_base.info()
#Determining how to subsample the data

df_base['Automatic Water Benchmarking Eligible'].value_counts()
#Renaming columns for easier coding

df_base.rename(columns = {'Total GHG Emissions(MtCO2e)':'t_ghg','Indirect GHG Emissions(MtCO2e)':'i_ghg','ENERGY STAR Score':'ess',

                         'Weather Normalized Source EUI(kBtu/ft2)':'weui','Reported Property Floor Area (Building(s)) (ft²)':'area',

                         '':''}, inplace=True)



#Subsampling, but removing the 657 observations that are "See Primary BBL", and removing extra variables

df_amr = df_base[df_base['Automatic Water Benchmarking Eligible'] == "Yes"]

df_amr = df_amr[['ess','t_ghg','i_ghg','weui','area','Borough','Latitude','Longitude']]

df_noamr = df_base[df_base['Automatic Water Benchmarking Eligible'] == "No"]

df_noamr = df_noamr[['ess','t_ghg','i_ghg','weui','area','Borough','Latitude','Longitude']]



#Validation of subsample

print('AMR subsample has',len(df_amr),'observations')

print('Non-AMR subsample has', len(df_noamr), 'observations')
#Missing observations

print('AMR Subsample:\n Total GHG Emissions has',df_amr['t_ghg'].isnull().sum(), 'missing obs.\n Energy STAR score has',

    df_amr['ess'].isnull().sum(), 'missing obs. \n Property area has',

      df_amr['area'].isnull().sum(), 'missing obs. \n Weather normalizes EUI has',

      df_amr['weui'].isnull().sum(), 'missing obs.\n \n','Non-AMR Subsample:\n Total GHG Emissions has',df_noamr['t_ghg'].isnull().sum(), 'missing obs.\n Energy STAR score has',

    df_noamr['ess'].isnull().sum(), 'missing obs. \n Property area has',

      df_noamr['area'].isnull().sum(), 'missing obs. \n Weather normalizes EUI has',

      df_noamr['weui'].isnull().sum(), 'missing obs.' 

)
#Removing the observations with missing data

df_amr = df_amr.dropna()

df_noamr = df_noamr.dropna()



df_amr = df_amr[df_amr['area'] != "Not Available"]

df_noamr = df_noamr[df_noamr['area'] != "Not Available"]

df_amr = df_amr[df_amr['t_ghg'] != "Not Available"]

df_noamr = df_noamr[df_noamr['t_ghg'] != "Not Available"]

df_amr = df_amr[df_amr['weui'] != "Not Available"]

df_noamr = df_noamr[df_noamr['weui'] != "Not Available"]

df_amr = df_amr[df_amr['ess'] != "Not Available"]

df_noamr = df_noamr[df_noamr['ess'] != "Not Available"]

df_amr = df_amr[df_amr['i_ghg'] != "Not Available"]

df_noamr = df_noamr[df_noamr['i_ghg'] != "Not Available"]





print('AMR Subsample:\n Total GHG Emissions has',df_amr['t_ghg'].isnull().sum(), 'missing obs.\n Energy STAR score has',

    df_amr['ess'].isnull().sum(), 'missing obs. \n Property area has',

      df_amr['area'].isnull().sum(), 'missing obs. \n Weather normalizes EUI has',

      df_amr['weui'].isnull().sum(), 'missing obs.\n There are',len(df_amr),'obs left.\n\n','Non-AMR Subsample:\n Total GHG Emissions has',df_noamr['t_ghg'].isnull().sum(), 'missing obs.\n Energy STAR score has',

    df_noamr['ess'].isnull().sum(), 'missing obs. \n Property area has',

      df_noamr['area'].isnull().sum(), 'missing obs. \n Weather normalizes EUI has',

      df_noamr['weui'].isnull().sum(), 'missing obs.\n There are',len(df_noamr),'obs left.' 

)
#Converting objects to floats

df_amr['t_ghg'] = df_amr['t_ghg'].astype(float)

df_amr['area'] = df_amr['area'].astype(float)

df_amr['ess'] = df_amr['ess'].astype(float)

df_amr['weui'] = df_amr['weui'].astype(float)

df_amr['i_ghg'] = df_amr['i_ghg'].astype(float)



df_noamr['t_ghg'] = df_noamr['t_ghg'].astype(float)

df_noamr['area'] = df_noamr['area'].astype(float)

df_noamr['ess'] = df_noamr['ess'].astype(float)

df_noamr['weui'] = df_noamr['weui'].astype(float)

df_noamr['i_ghg'] = df_noamr['i_ghg'].astype(float)
print('AMR Compliant Sample:\n\n',df_amr.describe(percentiles=[]).transpose())
print('Non-AMR Compliant Sample:\n\n',df_noamr.describe(percentiles=[]).transpose())
#Where are our buildings?

print('AMR Compliant Sample:\n\n', df_amr['Borough'].value_counts(), '\n\n Non-AMR Compliant Sample:\n\n',df_noamr['Borough'].value_counts())
#Looking at the distribution of each variable before assigning the type of regression, AMR Compliant

plt.subplot(2,2,1)

df_amr['t_ghg'].plot.hist()

plt.xlabel('GHG Emissions (MtCO2e)')

plt.subplot(2,2,2)

df_amr['area'].plot.hist()

plt.xlabel('Floor Space (ft^2)')

plt.xticks(rotation=45)

plt.subplot(2,2,3)

df_amr['ess'].plot.hist()

plt.xlabel('ENERGY STAR Score')

plt.subplot(2,2,4)

df_amr['weui'].plot.hist()

plt.xlabel('Weather Standardized EUI')

plt.tight_layout()

print('Figure 1: AMR Compliant Sample')

plt.show()
#Looking at the distribution of each variable before assigning the type of regression, AMR Compliant

plt.subplot(2,2,1)

df_noamr['t_ghg'].plot.hist()

plt.xlabel('GHG Emissions (MtCO2e)')

plt.subplot(2,2,2)

df_noamr['area'].plot.hist()

plt.xlabel('Floor Space (ft^2)')

plt.xticks(rotation=45)

plt.subplot(2,2,3)

df_noamr['ess'].plot.hist()

plt.xlabel('ENERGY STAR Score')

plt.subplot(2,2,4)

df_noamr['weui'].plot.hist()

plt.xlabel('Weather Standardized EUI')

plt.tight_layout()

print('Figure 2: Non-AMR Compliant Sample')

plt.show()
#Where are these buildings?

import folium

map = folium.Map([40.730610, -73.935242],

                zoom_start = 11)



for row in df_amr.itertuples():

    map.add_child(folium.Marker([row.Latitude,row.Longitude]))



for row in df_noamr.itertuples():

    map.add_child(folium.Marker([row.Latitude,row.Longitude], icon=folium.Icon(color='red')))

map



## BLUE = AMR Compliant, RED = Non-AMR Compliant
df_amr['amr'] = 1 #Subsample where 1=has amr tech, 0=does not

df_noamr['amr'] = 0

df_reg = df_amr.append(df_noamr, ignore_index=True) #Appending the data together

print('For the combined data,',df_reg['amr'].mean(),'% of the observations have AMR technology.')
import statsmodels.api as sm

regamr_p = sm.OLS(df_reg['t_ghg'],df_reg[['amr']])

type(regamr_p)

results = regamr_p.fit()

type(results)

print(results.summary())
regamr_p = sm.OLS(df_reg['t_ghg'],df_reg[['amr', 'ess','area','weui']])

type(regamr_p)

results = regamr_p.fit()

type(results)

print(results.summary())
#Testing sum of squared residuals.

y_hat = results.predict() #predicting y_hat from the fitted model

resids = y_hat - df_reg['t_ghg'] #predicted - actual for residuals

resids_d = resids/1000000 #transformation to better understand scale

resids_dsq = resids_d*resids_d

print('The sum of squared residals is', resids_dsq.sum())
residplot = resids_d.plot.kde()

plt.title('Distribution of residuals')
df_reg['y'] = df_reg['t_ghg'] / 1000000

df_reg['y_hat'] = y_hat

df_reg['y_hatsq'] = df_reg['y_hat']*df_reg['y_hat']
regamr_p = sm.OLS(df_reg['y'],df_reg[['y_hat','y_hatsq']])

type(regamr_p)

results = regamr_p.fit()

type(results)

print(results.summary())
corr = df_reg[['amr', 'ess','area','weui']].corr()

sns.heatmap(corr,xticklabels=corr.columns, yticklabels=corr.columns,annot=True, fmt="f")

plt.title('Correlation index (independent variables)')