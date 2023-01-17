import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt
df_gdp_path = "../input/gdp-population-years-1-to-2008/GDP_Data_Year_1_To_2008.csv"

df_pop_path = "../input/gdp-population-years-1-to-2008/Population_Data_Year_1_To_2008.csv"



df_gdp = pd.read_csv(df_gdp_path, delimiter=";",index_col='Country')

df_pop = pd.read_csv(df_pop_path, delimiter=";",index_col='Country')



df_gdp.index = df_gdp.index.str.strip()

df_pop.index = df_pop.index.str.strip()
df_gdp
df_pop
df_pop = df_pop.replace(',','', regex=True)

df_gdp = df_gdp.replace(',','', regex=True)



df_gdp = df_gdp.apply(pd.to_numeric, errors='coerce')

df_pop = df_pop.apply(pd.to_numeric, errors='coerce')
df_pop = df_pop.drop(['2009'], axis=1)
df_pop_T = df_pop.T

df_gdp_T = df_gdp.T

df_pop_T
df_pop_T.iloc[0] = df_pop_T.iloc[0].fillna(0)
def nan_helper(y):

    return np.isnan(y), lambda z: z.nonzero()[0]



def interpolate_with_nan(name,dataset):

    y = np.array(dataset[name])

    nans, x = nan_helper(y)

    y[nans]= np.interp(x(nans), x(~nans), y[~nans])

    return y



for country in df_pop_T.columns:

    df_pop_T[country] = interpolate_with_nan(country,df_pop_T)



for country in df_gdp_T.columns:

    df_gdp_T[country] = interpolate_with_nan(country,df_gdp_T)
df_gdp_T
fig, ax = plt.subplots(2, 2, figsize=(10, 8))



start_1, end_1 = '1', '1820'



ax[0,0].set_title('Scandinavian GDP: 1 - 1820')

ax[0,0].plot(df_gdp_T.loc[start_1:end_1, 'Finland'],

marker='.', linestyle='-', linewidth=0.5, label='Finland')

ax[0,0].plot(df_gdp_T.loc[start_1:end_1, 'Norway'],

marker='.', linestyle='-', linewidth=0.5, label='Norway')

ax[0,0].plot(df_gdp_T.loc[start_1:end_1, 'Sweden'],

marker='.', linestyle='-', linewidth=0.5, label='Sweden')

ax[0,0].plot(df_gdp_T.loc[start_1:end_1, 'Denmark'],

marker='.', linestyle='-', linewidth=0.5, label='Denmark')

ax[0,0].set_ylabel('GDP')

ax[0,0].legend()



start_2, end_2 = '1820', '1880'



ax[0,1].set_title('Scandinavian GDP: 1820 - 1880')

ax[0,1].plot(df_gdp_T.loc[start_2:end_2, 'Finland'],

marker='.', linestyle='-', linewidth=0.5, label='Finland')

ax[0,1].plot(df_gdp_T.loc[start_2:end_2, 'Norway'],

marker='.', linestyle='-', linewidth=0.5, label='Norway')

ax[0,1].plot(df_gdp_T.loc[start_2:end_2, 'Sweden'],

marker='.', linestyle='-', linewidth=0.5, label='Sweden')

ax[0,1].plot(df_gdp_T.loc[start_2:end_2, 'Denmark'],

marker='.', linestyle='-', linewidth=0.5, label='Denmark')

ax[0,1].set_ylabel('GDP')

ax[0,1].legend()



start_3, end_3 = '1880', '1945'



ax[1,0].set_title('Scandinavian GDP: 1880 - 1945')

ax[1,0].plot(df_gdp_T.loc[start_3:end_3, 'Finland'],

marker='.', linestyle='-', linewidth=0.5, label='Finland')

ax[1,0].plot(df_gdp_T.loc[start_3:end_3, 'Norway'],

marker='.', linestyle='-', linewidth=0.5, label='Norway')

ax[1,0].plot(df_gdp_T.loc[start_3:end_3, 'Sweden'],

marker='.', linestyle='-', linewidth=0.5, label='Sweden')

ax[1,0].plot(df_gdp_T.loc[start_3:end_3, 'Denmark'],

marker='.', linestyle='-', linewidth=0.5, label='Denmark')

ax[1,0].set_ylabel('GDP')

ax[1,0].legend()



start_4, end_4 = '1945', '2008'



ax[1,1].set_title('Scandinavian GDP: 1945 - 2008')

ax[1,1].plot(df_gdp_T.loc[start_4:end_4, 'Finland'],

marker='.', linestyle='-', linewidth=0.5, label='Finland')

ax[1,1].plot(df_gdp_T.loc[start_4:end_4, 'Norway'],

marker='.', linestyle='-', linewidth=0.5, label='Norway')

ax[1,1].plot(df_gdp_T.loc[start_4:end_4, 'Sweden'],

marker='.', linestyle='-', linewidth=0.5, label='Sweden')

ax[1,1].plot(df_gdp_T.loc[start_4:end_4, 'Denmark'],

marker='.', linestyle='-', linewidth=0.5, label='Denmark')

ax[1,1].set_ylabel('GDP')

ax[1,1].legend()
fig, av = plt.subplots(2, 2, figsize=(10, 8))



av[0,0].set_title('Scandinavian population: 1 - 1820')

av[0,0].plot(df_pop_T.loc[start_1:end_1, 'Finland'],

marker='.', linestyle='-', linewidth=0.5, label='Finland')

av[0,0].plot(df_pop_T.loc[start_1:end_1, 'Norway'],

marker='.', linestyle='-', linewidth=0.5, label='Norway')

av[0,0].plot(df_pop_T.loc[start_1:end_1, 'Sweden'],

marker='.', linestyle='-', linewidth=0.5, label='Sweden')

av[0,0].plot(df_pop_T.loc[start_1:end_1, 'Denmark'],

marker='.', linestyle='-', linewidth=0.5, label='Denmark')

av[0,0].set_ylabel('GDP')

av[0,0].legend()



av[0,1].set_title('Scandinavian population: 1820 - 1880')

av[0,1].plot(df_pop_T.loc[start_2:end_2, 'Finland'],

marker='.', linestyle='-', linewidth=0.5, label='Finland')

av[0,1].plot(df_pop_T.loc[start_2:end_2, 'Norway'],

marker='.', linestyle='-', linewidth=0.5, label='Norway')

av[0,1].plot(df_pop_T.loc[start_2:end_2, 'Sweden'],

marker='.', linestyle='-', linewidth=0.5, label='Sweden')

av[0,1].plot(df_pop_T.loc[start_2:end_2, 'Denmark'],

marker='.', linestyle='-', linewidth=0.5, label='Denmark')

av[0,1].set_ylabel('GDP')

av[0,1].legend()



av[1,0].set_title('Scandinavian population: 1880 - 1945')

av[1,0].plot(df_pop_T.loc[start_3:end_3, 'Finland'],

marker='.', linestyle='-', linewidth=0.5, label='Finland')

av[1,0].plot(df_pop_T.loc[start_3:end_3, 'Norway'],

marker='.', linestyle='-', linewidth=0.5, label='Norway')

av[1,0].plot(df_pop_T.loc[start_3:end_3, 'Sweden'],

marker='.', linestyle='-', linewidth=0.5, label='Sweden')

av[1,0].plot(df_pop_T.loc[start_3:end_3, 'Denmark'],

marker='.', linestyle='-', linewidth=0.5, label='Denmark')

av[1,0].set_ylabel('GDP')

av[1,0].legend()



av[1,1].set_title('Scandinavian population: 1945 - 2008')

av[1,1].plot(df_pop_T.loc[start_4:end_4, 'Finland'],

marker='.', linestyle='-', linewidth=0.5, label='Finland')

av[1,1].plot(df_pop_T.loc[start_4:end_4, 'Norway'],

marker='.', linestyle='-', linewidth=0.5, label='Norway')

av[1,1].plot(df_pop_T.loc[start_4:end_4, 'Sweden'],

marker='.', linestyle='-', linewidth=0.5, label='Sweden')

av[1,1].plot(df_pop_T.loc[start_4:end_4, 'Denmark'],

marker='.', linestyle='-', linewidth=0.5, label='Denmark')

av[1,1].set_ylabel('GDP')

av[1,1].legend()