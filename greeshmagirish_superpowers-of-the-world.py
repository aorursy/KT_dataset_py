import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

sns.set_style('whitegrid')

sns.set_context('talk')
armedforces_tot = pd.read_csv('/kaggle/input/worldbank-data-on-gdp-population-and-military/API_MS.MIL.TOTL.P1_DS2_en_csv_v2_513199.csv')

milexp_gdp = pd.read_csv('/kaggle/input/worldbank-data-on-gdp-population-and-military/API_MS.MIL.XPND.GD.ZS_DS2_en_csv_v2_511529.csv')

milexp_govt = pd.read_csv('/kaggle/input/worldbank-data-on-gdp-population-and-military/API_MS.MIL.XPND.ZS_DS2_en_csv_v2_514803.csv')

milexp_usd = pd.read_csv('/kaggle/input/military-expenditure-of-countries-19602019/Military Expenditure.csv')

land_area = pd.read_csv('/kaggle/input/worldbank-data-on-gdp-population-and-military/API_AG.LND.TOTL.K2_DS2_en_csv_v2_511817.csv')

pop = pd.read_csv('/kaggle/input/worldbank-data-on-gdp-population-and-military/API_SP.POP.TOTL_DS2_en_csv_v2_511378.csv')

gdp = pd.read_csv('/kaggle/input/worldbank-data-on-gdp-population-and-military/API_NY.GDP.MKTP.CD_DS2_en_csv_v2_559588.csv')
top_six = ['United States', 'Russian Federation', 'China', 'India', 'European Union', 'Brazil']
milexp_usd_six = milexp_usd[milexp_usd['Name'].isin(top_six)]

milexp_usd_six = milexp_usd_six.T.reset_index()

milexp_usd_six.columns = milexp_usd_six.iloc[0]

milexp_usd_six = milexp_usd_six.drop([0,1,2,3]).reset_index()

milexp_usd_six = milexp_usd_six.drop('index', axis=1)

milexp_usd_six = milexp_usd_six.rename(columns = {'Name': 'Mil Exp in USD'})

milexp_usd_six['Mil Exp in USD'] = milexp_usd_six['Mil Exp in USD'].astype(int)

milexp_usd_six.China = milexp_usd_six.China.astype(float)

milexp_usd_six.India = milexp_usd_six.India.astype(float)

milexp_usd_six.Brazil = milexp_usd_six.Brazil.astype(float)

milexp_usd_six['United States'] = milexp_usd_six['United States'].astype(float)

milexp_usd_six['Russian Federation'] = milexp_usd_six['Russian Federation'].astype(float)

milexp_usd_six['European Union'] = milexp_usd_six['European Union'].astype(float)

milexp_usd_six = milexp_usd_six[milexp_usd_six['Mil Exp in USD'] > 1992]



armedforces_six = armedforces_tot[armedforces_tot['Country Name'].isin(top_six)]

armedforces_six = armedforces_six.T.reset_index()

armedforces_six.columns = armedforces_six.iloc[0]

armedforces_six = armedforces_six.drop([0,1,2,3]).reset_index()

armedforces_six = armedforces_six.drop('index', axis=1)

armedforces_six = armedforces_six.rename(columns = {'Country Name': 'Armed Forces Total'})

armedforces_six = armedforces_six[armedforces_six['Armed Forces Total'] != 'Unnamed: 64']

armedforces_six['Armed Forces Total'] = armedforces_six['Armed Forces Total'].astype(int)

armedforces_six = armedforces_six[armedforces_six['Armed Forces Total'] > 1992]

armedforces_six.China = armedforces_six.China.astype(float)

armedforces_six.India = armedforces_six.India.astype(float)

armedforces_six.Brazil = armedforces_six.Brazil.astype(float)

armedforces_six['United States'] = armedforces_six['United States'].astype(float)

armedforces_six['Russian Federation'] = armedforces_six['Russian Federation'].astype(float)

armedforces_six['Russian Federation'] = armedforces_six['Russian Federation'].astype(float)





milexp_gdp_six = milexp_gdp[milexp_gdp['Country Name'].isin(top_six)]

milexp_gdp_six = milexp_gdp_six.T.reset_index()

milexp_gdp_six.columns = milexp_gdp_six.iloc[0]

milexp_gdp_six = milexp_gdp_six.drop([0,1,2,3]).reset_index()

milexp_gdp_six = milexp_gdp_six.drop('index', axis=1)

milexp_gdp_six = milexp_gdp_six.rename(columns = {'Country Name': 'Mil Exp % GDP'})

milexp_gdp_six = milexp_gdp_six[milexp_gdp_six['Mil Exp % GDP'] != 'Unnamed: 64']

milexp_gdp_six['Mil Exp % GDP'] = milexp_gdp_six['Mil Exp % GDP'].astype(int)

milexp_gdp_six = milexp_gdp_six[milexp_gdp_six['Mil Exp % GDP'] > 1992]

milexp_gdp_six.China = milexp_gdp_six.China.astype(float)

milexp_gdp_six.India = milexp_gdp_six.India.astype(float)

milexp_gdp_six.Brazil = milexp_gdp_six.Brazil.astype(float)

milexp_gdp_six['United States'] = milexp_gdp_six['United States'].astype(float)

milexp_gdp_six['European Union'] = milexp_gdp_six['European Union'].astype(float)



milexp_govt_six = milexp_govt[milexp_govt['Country Name'].isin(top_six)]

milexp_govt_six = milexp_govt_six.T.reset_index()

milexp_govt_six.columns = milexp_govt_six.iloc[0]

milexp_govt_six = milexp_govt_six.drop([0,1,2,3]).reset_index()

milexp_govt_six = milexp_govt_six.drop('index', axis=1)

milexp_govt_six = milexp_govt_six.rename(columns = {'Country Name': 'Mil Exp % Govt Exp'})

milexp_govt_six = milexp_govt_six[milexp_govt_six['Mil Exp % Govt Exp'] != 'Unnamed: 64']

milexp_govt_six['Mil Exp % Govt Exp'] = milexp_govt_six['Mil Exp % Govt Exp'].astype(int)

milexp_govt_six = milexp_govt_six[milexp_govt_six['Mil Exp % Govt Exp'] > 1992]

milexp_govt_six.China = milexp_govt_six.China.astype(float)

milexp_govt_six.India = milexp_govt_six.India.astype(float)

milexp_govt_six.Brazil = milexp_govt_six.Brazil.astype(float)

milexp_govt_six['United States'] = milexp_govt_six['United States'].astype(float)

milexp_govt_six['Russian Federation'] = milexp_govt_six['Russian Federation'].astype(float)

milexp_govt_six['European Union'] = milexp_govt_six['European Union'].astype(float)





land_area_six = land_area[land_area['Country Name'].isin(top_six)]

land_area_six = land_area_six.T.reset_index()

land_area_six.columns = land_area_six.iloc[0]

land_area_six = land_area_six.drop([0,1,2,3]).reset_index()

land_area_six = land_area_six.drop('index', axis=1)

land_area_six = land_area_six.rename(columns = {'Country Name': 'Land Area'})

land_area_six = land_area_six[land_area_six['Land Area'] != 'Unnamed: 64']

land_area_six['Land Area'] = land_area_six['Land Area'].astype(int)

land_area_six.China = land_area_six.China.astype(float)

land_area_six.India = land_area_six.India.astype(float)

land_area_six.Brazil = land_area_six.Brazil.astype(float)

land_area_six['United States'] = land_area_six['United States'].astype(float)

land_area_six['Russian Federation'] = land_area_six['Russian Federation'].astype(float)

land_area_six['European Union'] = land_area_six['European Union'].astype(float)

land_area_six = land_area_six[land_area_six['Land Area'] > 1992]



pop_six = pop[pop['Country Name'].isin(top_six)]

pop_six = pop_six.T.reset_index()

pop_six.columns = pop_six.iloc[0]

pop_six = pop_six.drop([0,1,2,3]).reset_index()

pop_six = pop_six.drop('index', axis=1)

pop_six = pop_six.rename(columns = {'Country Name': 'Population'})

pop_six = pop_six[pop_six['Population'] != 'Unnamed: 64']

pop_six['Population'] = pop_six['Population'].astype(int)

pop_six.China = pop_six.China.astype(float)

pop_six.India = pop_six.India.astype(float)

pop_six.Brazil = pop_six.Brazil.astype(float)

pop_six['United States'] = pop_six['United States'].astype(float)

pop_six['Russian Federation'] = pop_six['Russian Federation'].astype(float)

pop_six['European Union'] = pop_six['European Union'].astype(float)

pop_six = pop_six[pop_six['Population'] > 1992]



gdp_six = gdp[gdp['Country Name'].isin(top_six)]

gdp_six = gdp_six.T.reset_index()

gdp_six.columns = gdp_six.iloc[0]

gdp_six = gdp_six.drop([0,1,2,3]).reset_index()

gdp_six = gdp_six.drop('index', axis=1)

gdp_six = gdp_six.rename(columns = {'Country Name': 'GDP'})

gdp_six = gdp_six[gdp_six['GDP'] != 'Unnamed: 64']

gdp_six['GDP'] = gdp_six['GDP'].astype(int)

gdp_six.China = gdp_six.China.astype(float)

gdp_six.India = gdp_six.India.astype(float)

gdp_six.Brazil = gdp_six.Brazil.astype(float)

gdp_six['United States'] = gdp_six['United States'].astype(float)

gdp_six['Russian Federation'] = gdp_six['Russian Federation'].astype(float)

gdp_six['European Union'] = gdp_six['European Union'].astype(float)

gdp_six = gdp_six[gdp_six['GDP'] > 1992]
land_area_six['United States'].max()
fig,ax = plt.subplots(1,1,figsize=(20,15))

ax = plt.gca()

pop_six.plot(kind='line',x='Population',y='United States',ax=ax)

plt.title('Population of United States - 1992 Onwards')
fig,ax = plt.subplots(1,1,figsize=(20,15))

ax = plt.gca()

gdp_six.plot(kind='line',x='GDP',y='United States',ax=ax)

plt.title('GDP of United States - 1992 Onwards')
fig,ax = plt.subplots(1,1,figsize=(20,15))

ax = plt.gca()

armedforces_six.plot(kind='line',x='Armed Forces Total',y='United States',ax=ax)

plt.title('Armed Forces Strength of United States - 1992 Onwards')
fig,(ax1,ax2) = plt.subplots(1,2,figsize=(20,15))

ax = plt.gca()

gdp_six.plot(kind='line',x='GDP',y='United States',ax=ax1, color='green')

milexp_gdp_six.plot(kind='line',x='Mil Exp % GDP',y='United States',ax=ax2)

ax1.set_title('GDP United States - 1992 Onwards')

ax2.set_title('Military Expenditure as a % of GDP United States - 1992 Onwards')
fig,(ax1,ax2) = plt.subplots(1,2,figsize=(25,15))

ax = plt.gca()

milexp_govt_six.plot(kind='line',x='Mil Exp % Govt Exp',y='United States',ax=ax1, color='green')

milexp_usd_six.plot(kind='line',x='Mil Exp in USD',y='United States',ax=ax2)

ax1.set_title('Military Expenditure as a % of General Govt Expenditure United States - 1992 Onwards')

ax2.set_title('Military Expenditure in USD United States - 1992 Onwards')
land_area_six['China'].max()
fig,ax = plt.subplots(1,1,figsize=(20,15))

ax = plt.gca()

pop_six.plot(kind='line',x='Population',y='China',ax=ax)

plt.title('Population of China - 1992 Onwards')
fig,ax = plt.subplots(1,1,figsize=(20,15))

ax = plt.gca()

gdp_six.plot(kind='line',x='GDP',y='China',ax=ax)

plt.title('GDP of China - 1992 Onwards')
fig,ax = plt.subplots(1,1,figsize=(20,15))

ax = plt.gca()

armedforces_six.plot(kind='line',x='Armed Forces Total',y='China',ax=ax)

plt.title('Armed Forces Strength of China - 1992 Onwards')
fig,(ax1,ax2) = plt.subplots(1,2,figsize=(20,15))

ax = plt.gca()

gdp_six.plot(kind='line',x='GDP',y='China',ax=ax1, color='green')

milexp_gdp_six.plot(kind='line',x='Mil Exp % GDP',y='China',ax=ax2)

ax1.set_title('GDP China - 1992 Onwards')

ax2.set_title('Military Expenditure as a % of GDP China - 1992 Onwards')
fig,(ax1,ax2) = plt.subplots(1,2,figsize=(25,15))

ax = plt.gca()

milexp_govt_six.plot(kind='line',x='Mil Exp % Govt Exp',y='China',ax=ax1, color='green')

milexp_usd_six.plot(kind='line',x='Mil Exp in USD',y='China',ax=ax2)

ax1.set_title('Military Expenditure as a % of General Govt Expenditure China - 1992 Onwards')

ax2.set_title('Military Expenditure in USD China - 1992 Onwards')
fig,ax = plt.subplots(1,1,figsize=(25,15))

land_area_six.plot(kind='line',x='Land Area',y='India',ax=ax, color='green')

land_area_six.plot(kind='line',x='Land Area',y='Brazil',ax=ax, color='blue')

land_area_six.plot(kind='line',x='Land Area',y='Russian Federation',ax=ax, color='red')

land_area_six.plot(kind='line',x='Land Area',y='European Union',ax=ax, color='purple')

ax.set_title('Land Area of Potential Superpowers - 1992 onwards')
fig,ax = plt.subplots(1,1,figsize=(25,15))

pop_six.plot(kind='line',x='Population',y='India',ax=ax, color='green')

pop_six.plot(kind='line',x='Population',y='Brazil',ax=ax, color='blue')

pop_six.plot(kind='line',x='Population',y='Russian Federation',ax=ax, color='red')

pop_six.plot(kind='line',x='Population',y='European Union',ax=ax, color='purple')

ax.set_title('Population of Potential Superpowers - 1992 onwards')
fig,ax = plt.subplots(1,1,figsize=(25,15))

gdp_six.plot(kind='line',x='GDP',y='India',ax=ax, color='green')

gdp_six.plot(kind='line',x='GDP',y='Brazil',ax=ax, color='blue')

gdp_six.plot(kind='line',x='GDP',y='Russian Federation',ax=ax, color='red')

gdp_six.plot(kind='line',x='GDP',y='European Union',ax=ax, color='purple')

ax.set_title('GDP of Potential Superpowers - 1992 onwards')
fig, ax = plt.subplots(1,1,figsize=(25,15))

ax = plt.gca()

armedforces_six.plot(kind='line',x='Armed Forces Total',y='India',ax=ax, color='green')

armedforces_six.plot(kind='line',x='Armed Forces Total',y='Brazil',ax=ax, color='blue')

armedforces_six.plot(kind='line',x='Armed Forces Total',y='Russian Federation',ax=ax, color='red')

armedforces_six.plot(kind='line',x='Armed Forces Total',y='European Union',ax=ax, color='purple')

ax.set_title('Armed Forces Total Potential Superpowers - 1992 Onwards')
fig,(ax1,ax2) = plt.subplots(1,2,figsize=(20,15))

ax = plt.gca()

gdp_six.plot(kind='line',x='GDP',y='India',ax=ax1, color='green')

gdp_six.plot(kind='line',x='GDP',y='Brazil',ax=ax1, color='blue')

gdp_six.plot(kind='line',x='GDP',y='Russian Federation',ax=ax1, color='red')

gdp_six.plot(kind='line',x='GDP',y='European Union',ax=ax1, color='purple')

milexp_gdp_six.plot(kind='line',x='Mil Exp % GDP',y='India',ax=ax2, color='green')

milexp_gdp_six.plot(kind='line',x='Mil Exp % GDP',y='Brazil',ax=ax2, color='blue')

milexp_gdp_six.plot(kind='line',x='Mil Exp % GDP',y='Russian Federation',ax=ax2, color='red')

milexp_gdp_six.plot(kind='line',x='Mil Exp % GDP',y='European Union',ax=ax2, color='purple')

ax1.set_title('GDP Potential Superpowers - 1992 Onwards')

ax2.set_title('Military Expenditure as a % of GDP Potential Superpowers - 1992 Onwards')
fig,(ax1,ax2) = plt.subplots(1,2,figsize=(30,15))

ax = plt.gca()

milexp_govt_six.plot(kind='line',x='Mil Exp % Govt Exp',y='India',ax=ax1, color='green')

milexp_govt_six.plot(kind='line',x='Mil Exp % Govt Exp',y='Brazil',ax=ax1, color='blue')

milexp_govt_six.plot(kind='line',x='Mil Exp % Govt Exp',y='Russian Federation',ax=ax1, color='red')

milexp_govt_six.plot(kind='line',x='Mil Exp % Govt Exp',y='European Union',ax=ax1, color='purple')

milexp_usd_six.plot(kind='line',x='Mil Exp in USD',y='India',ax=ax2, color='green')

milexp_usd_six.plot(kind='line',x='Mil Exp in USD',y='Brazil',ax=ax2, color='blue')

milexp_usd_six.plot(kind='line',x='Mil Exp in USD',y='Russian Federation',ax=ax2, color='red')

milexp_usd_six.plot(kind='line',x='Mil Exp in USD',y='European Union',ax=ax2, color='purple')

ax1.set_title('Military Expenditure as a % of General Govt Expenditure Potential Superpowers - 1992 Onwards')

ax2.set_title('Military Expenditure in USD Potential Superpowers - 1992 Onwards')