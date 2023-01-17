import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import re # regular expressions

import matplotlib.pyplot as plt # plotting

import seaborn as sns # plotting
wdi_full_data = pd.read_csv('../input/wdi-indicators-data/WDIData.csv')

indicator_list = wdi_full_data[['Indicator Name','Indicator Code']].drop_duplicates().values



# This part is inspired from here: https://www.kaggle.com/kmravikumar/choosing-topics-to-explore

modified_indicators = []

unique_indicator_codes = []

for ele in indicator_list:

    indicator = ele[0]

    indicator_code = ele[1].strip()

    if indicator_code not in unique_indicator_codes:

        new_indicator = re.sub('[,()]',"",indicator).lower()

        new_indicator = re.sub('-'," to ",new_indicator).lower()

        modified_indicators.append([new_indicator,indicator_code])

        unique_indicator_codes.append(indicator_code)

indicators = pd.DataFrame(modified_indicators,columns=['Indicator Name','Indicator Code']).drop_duplicates()
selected_health_indicators = [('life expectancy at birth total years', 'SP.DYN.LE00.IN'),

                              ('mortality rate adult female per 1000 female adults', 'SP.DYN.AMRT.FE'),

                              ('mortality rate adult male per 1000 male adults', 'SP.DYN.AMRT.MA'),

                              ('mortality rate infant per 1000 live births', 'SP.DYN.IMRT.IN'),

                              ('current health expenditure % of gdp', 'SH.XPD.CHEX.GD.ZS')]



selected_economy_indicators = [('gdp per capita ppp constant 2011 international $', 'NY.GDP.PCAP.PP.KD'),

                               ('employment to population ratio 15+ total % national estimate', 'SL.EMP.TOTL.SP.NE.ZS'),

                               ('income share held by third 20%', 'SI.DST.03RD.20')]



selected_indicators = selected_health_indicators + selected_economy_indicators

selected_countries = ['Belgium', 'Denmark', 'France', 'Germany', 'Switzerland', 'Spain', 'Netherlands', 'United States', 

                      'Portugal', 'Romania', 'United Kingdom', 'Austria', 'Norway']



wdi_data = wdi_full_data[wdi_full_data['Country Name'].isin(selected_countries) & 

                         wdi_full_data['Indicator Code'].isin([ind[1] for ind in selected_indicators])].copy()

wdi_data = wdi_data.drop(['Country Code', 'Indicator Name', 'Unnamed: 64'], axis=1)
# Reshape the data so that later imputation of missing values is easy.

wdi_data = wdi_data.melt(id_vars=['Country Name', 'Indicator Code'], 

                    value_vars=wdi_data.columns.difference(['Country Name', 'Indicator Code']), 

                                var_name='Year', value_name='Indicator Value')

wdi_data = wdi_data.set_index(['Country Name', 'Year', 'Indicator Code'])['Indicator Value'].unstack().reset_index().rename_axis(None, axis=1)



# Fill in missing values.

for country in set(wdi_data['Country Name']):

    country_slice = wdi_data[wdi_data['Country Name'] == country].copy()

    country_slice.iloc[:, 2:] = country_slice.iloc[:, 2:].interpolate(method='linear', limit_direction='both', axis=0)

    wdi_data[wdi_data['Country Name'] == country] = country_slice
selected_years = ['2012', '2013', '2014', '2015', '2016', '2017', '2018', '2019']

wdi_data = wdi_data[wdi_data['Year'].isin(selected_years)]
numbeo_data = pd.read_csv('../input/numbeo-data/numbeo_qol.csv')

numbeo_data = numbeo_data[numbeo_data['Country Name'].isin(selected_countries)].copy().reset_index()
for country in set(numbeo_data['Country Name']):

    country_slice = numbeo_data[numbeo_data['Country Name'] == country].copy()

    country_slice.iloc[:, 2:] = country_slice.iloc[:, 2:].interpolate(method='linear', limit_direction='both', axis=0)

    numbeo_data[numbeo_data['Country Name'] == country] = country_slice

numbeo_data = numbeo_data.drop(['index'], axis=1)
pop_data = pd.read_csv('../input/population-by-country-2020/population_by_country_2020.csv').iloc[:, 0:2]

pop_data.rename(columns={'Country (or dependency)': 'Country Name', 'Population (2020)': 'Population'}, inplace=True)
# Select the relevant data

df = wdi_data[['Country Name', 'SH.XPD.CHEX.GD.ZS', 'SP.DYN.LE00.IN', 'NY.GDP.PCAP.PP.KD']]



# Group WDI data by the country

health_exp_life_exp = df.groupby(['Country Name']).mean().reset_index()



# Create the plot

plt.figure(dpi=150)

colors = ["#DA2C43", "#FFAA1D", "#FFF700", "#299617", "#A7F432", "#2243B6", "#5DADEC", "#00468C", "#A83731", "#353839",

         "#FF007C", "#6F2DA8", "#D98695"]

ax = sns.scatterplot(health_exp_life_exp['SH.XPD.CHEX.GD.ZS'], health_exp_life_exp['SP.DYN.LE00.IN'], 

            hue = health_exp_life_exp['Country Name'], size = health_exp_life_exp['NY.GDP.PCAP.PP.KD'].to_numpy() * 2, 

                sizes=(100,900), alpha = 0.8, palette = colors)

plt.xlabel('Avg health expenditure % of GDP', fontsize = 14)

plt.ylabel('Avg life expectancy [in years]', fontsize = 14)

plt.title('Health expenditure vs. Life expectancy (2012-2019)', fontsize = 16)

plt.ylim(74, 84)

plt.grid(True)

h,l = ax.get_legend_handles_labels()

plt.legend(h[:health_exp_life_exp.shape[0]+1], l[:health_exp_life_exp.shape[0]+1], bbox_to_anchor=(1.03, 1), loc=2, borderaxespad=0., fancybox=True)

plt.show()
# Select the relevant data

df = numbeo_data[['Country Name', 'Health Care', 'Pollution']].groupby(['Country Name']).mean().reset_index()



# Reshape the data to a format consistent with the graph style

health_pollution = df.melt('Country Name', var_name='Indices', value_name='Value')



# # Create the plot

plt.figure(dpi=150)

ax = sns.barplot(y="Country Name", x="Value", hue="Indices", data=health_pollution, 

                 palette = sns.color_palette("bright"))

plt.xlabel('Avg value', fontsize = 14)

plt.ylabel('Country', fontsize = 14)

plt.title('Avg health care and pollution indices (2012-2019)', fontsize = 16)

plt.grid(True)

plt.legend(bbox_to_anchor=(1.03, 1), loc=2, borderaxespad=0., fancybox=True)

plt.show()
# Select the relevant data

df = wdi_data[['Country Name', 'NY.GDP.PCAP.PP.KD', 'SI.DST.03RD.20']]



# Join the WDI data and the population data

employment_gdp = pd.merge(df.groupby(['Country Name']).mean().reset_index(), pop_data, on='Country Name')



# Create the plot

plt.figure(dpi=150)

colors = ["#DA2C43", "#FFAA1D", "#FFF700", "#299617", "#A7F432", "#2243B6", "#5DADEC", "#00468C", "#A83731", "#353839",

         "#FF007C", "#6F2DA8", "#D98695"]

ax = sns.scatterplot(employment_gdp['NY.GDP.PCAP.PP.KD'], employment_gdp['SI.DST.03RD.20'], 

            hue = employment_gdp['Country Name'], size = employment_gdp['Population'].to_numpy() * 2, 

                sizes=(100,900), alpha = 1, palette = colors)

plt.xlabel('Avg GDP per capita', fontsize = 14)

plt.ylabel('Avg income % held by middle class', fontsize = 14)

plt.title('GDP vs. Income share held (2012-2019)', fontsize = 16)

plt.grid(True)

plt.ylim(15, 19)

h,l = ax.get_legend_handles_labels()

plt.legend(h[:employment_gdp.shape[0]+1], l[:employment_gdp.shape[0]+1], bbox_to_anchor=(1.03, 1), loc=2, borderaxespad=0., fancybox=True)

plt.show()
# Select the relevant data

df = numbeo_data[['Country Name', 'Purchasing Power', 'Cost of Living']]



# Reshape the data to a format consistent with the graph style

pp_col = pd.merge(df.groupby(['Country Name']).mean().reset_index(), pop_data, on='Country Name')



# Create the plot

plt.figure(dpi=150)

colors = ["#DA2C43", "#FFAA1D", "#FFF700", "#299617", "#A7F432", "#2243B6", "#5DADEC", "#00468C", "#A83731", "#353839",

         "#FF007C", "#6F2DA8", "#D98695"]

ax = sns.scatterplot(pp_col['Purchasing Power'], pp_col['Cost of Living'], 

            hue = pp_col['Country Name'], size = pp_col['Population'].to_numpy() * 2, 

                sizes=(100,900), alpha = 0.9, palette = colors)

plt.xlabel('Avg purchasing power index', fontsize = 14)

plt.ylabel('Avg cost of living index', fontsize = 14)

plt.title('Purchase power vs. Cost of living (2012-2019)', fontsize = 16)

plt.grid(True)

h,l = ax.get_legend_handles_labels()

plt.legend(h[:pp_col.shape[0]+1], l[:pp_col.shape[0]+1], bbox_to_anchor=(1.03, 1), loc=2, borderaxespad=0., fancybox=True)

plt.show()
# Select the relevant data

df = numbeo_data[['Country Name', 'Quality of Life', 'Safety']].groupby(['Country Name']).mean().reset_index()



# Reshape the data to a format consistent with the graph style

qol_safety = df.melt('Country Name', var_name='Indices', value_name='Value')



# Create the plot

plt.figure(dpi=150)

ax = sns.barplot(y="Country Name", x="Value", hue="Indices", data=qol_safety, 

                 palette = sns.color_palette("bright"))

plt.xlabel('Avg value', fontsize = 14)

plt.ylabel('Country', fontsize = 14)

plt.title('Avg quality of life and safety indices (2012-2019)', fontsize = 16)

plt.grid(True)

plt.legend(bbox_to_anchor=(1.03, 1), loc=2, borderaxespad=0., fancybox=True)

plt.show()