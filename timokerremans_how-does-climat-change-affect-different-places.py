# import used libraries



import numpy as np    # linear algebra

import pandas as pd   # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns # data visualization

import matplotlib.pyplot as plt



# import libraries for map visualization

import plotly.offline as py

py.init_notebook_mode(connected=True)

import plotly.graph_objs as go

import plotly.tools as tls



# read the data files.



global_temp_df = pd.read_csv("../input/GlobalTemperatures.csv", index_col = ['dt'], parse_dates=['dt'])

Temp_by_country = pd.read_csv('../input/GlobalLandTemperaturesByCountry.csv',index_col = ['dt'], parse_dates=['dt'])
# structure data to multi-index by country and year, resampled by min, max and mean



df_max = Temp_by_country.groupby('Country').resample('Y').max()

df_min = Temp_by_country.groupby('Country').resample('Y').min()

df_mean = Temp_by_country.groupby('Country').resample('Y').mean()



# rename the column names



df_max.rename(columns={'AverageTemperature' : 'MaxTemp','AverageTemperatureUncertainty' : 'MaxTempUnc'}, inplace=True)

df_min.rename(columns={'AverageTemperature' : 'MinTemp','AverageTemperatureUncertainty' : 'MinTempUnc'}, inplace=True)

df_mean.rename(columns={'AverageTemperature' : 'MeanTemp','AverageTemperatureUncertainty' : 'MeanTempUnc'}, inplace=True)



# Get all in one dataframe

df_temp = df_mean

df_temp['MaxTemp'] = df_max.MaxTemp

df_temp['MaxTempUnc'] = df_max.MaxTempUnc

df_temp['MinTemp'] = df_min.MinTemp

df_temp['MinTempUnc'] = df_min.MinTempUnc



# Rename the datetime index column



df_temp.index.rename(('Country','Year'), inplace=True)



# Change Date format of index level 1 to just year (very round-a-boud way!)



Index_year = pd.to_datetime(df_temp.index.get_level_values(1), format = '%Y').strftime('%Y') # change the format of the datatime index



df_temp['year'] = Index_year                             # make a column with year

df_temp['Country'] = df_temp.index.get_level_values(0)   # make a column with country



df_temp.set_index(['Country', 'year'], inplace=True)                   # use those two columns to define the new multi index





# get spread

df_temp['Spread'] = df_temp['MaxTemp'] - df_temp['MinTemp']



# delete abundant dataframes for memory

del df_max

del df_mean

del df_min
# Let's look how our dataframe looks like at this point.

df_temp.head(9)
# indexing with multi-indices

print(df_temp.loc[('Belgium'),:])                         # Slicing data for one country

print(df_temp.loc[('Belgium','1994')])                    # Slicing data for one country, in one year

print(df_temp.loc[('Belgium','1994'),'MaxTemp'])          # Slicing data for one country, in one year, one specific column

print(df_temp.loc[('Belgium',slice('2004','2006')),:])    # Slicing a range of years

# df_temp.loc[('Belgium',slice('2004','2005')),:][['MeanTemp','MaxTemp']]

# df_temp.loc[(slice(None),slice('2004','2005')),:][['MeanTemp','MaxTemp']]
# Getting rolling averages since data can be very eratic.

rolling_years = 10

country = 'Europe'

metric = 'MeanTemp'



# Looking at the spread between max and min temperature over time, for different rolling averages.



fig, ax = plt.subplots()

df_temp.loc[country, metric].rolling(5).mean().plot(figsize=(25,7.5), ax=ax);

df_temp.loc[country, metric].rolling(10).mean().plot(ax=ax);

df_temp.loc[country, metric].rolling(30).mean().plot(ax=ax);

df_temp.loc[country, metric].rolling(50).mean().plot(ax=ax);

plt.grid()

plt.title(metric + ' in ' + country + ' for different rolling averages.', fontsize=24)

plt.legend(title=(metric, country), loc='upper left', labels=['5 year RA', '10 year RA', '20 year RA', '50 year RA'], fontsize=20);
# We first need a column with all the country names.



df_temp['Country'] = df_temp.index.get_level_values(0)



# To avoid confusion with the naming of the countries, we will disregard countries with external land:



df_temp_unique = df_temp[~df_temp['Country'].isin(

    ['Denmark', 'Antarctica', 'France', 'Europe', 'Netherlands',

     'United Kingdom', 'Africa', 'South America'])]



# Replace the countries we know with only their name, not specifying that they are in Europe to contrast them with their wordly teratories.



df_temp_unique = df_temp_unique.replace(

   ['Denmark (Europe)', 'France (Europe)', 'Netherlands (Europe)', 'United Kingdom (Europe)'],

   ['Denmark', 'France', 'Netherlands', 'United Kingdom'])



# now we create an array of all the unique countries:

countries = np.unique(df_temp_unique['Country'])
# For each country we calculate a metric that we want to visualize on a graph



Metric = [] # Here you can choose any metric you want, just use column name. It would be nice to get some metric corresponding to climat change



# STATIC METRIC:

#for country in countries:

#    metric.append(df_temp_unique[df_temp_unique['Country'] == country]['MeanTemp'].mean())

    #metric.append((df_temp_unique[df_temp_unique['Country'] == country]['MaxTemp']-df_temp_unique[df_temp_unique['Country'] == country]['MinTemp']).mean()) 

    #metric.append(df_temp_unique[df_temp_unique['Country'] == country]['MeanTemp'].loc[('1994-12-31':'2015-12-31')].mean())

    



# DYNAMICAL METRIC

# Here we look at the CHANGE in MeanTemp (or Spread) for each country over a period of 100 years. To avoid outlier years, we take a 20 year rolling average for each point.

for country in countries:

    df_now  = df_temp_unique[df_temp_unique['Country'] == country].loc[(slice(country),slice('1990','2010')),:]['MeanTemp'].mean(); # metric 20YRA representing metric NOW

    df_then = df_temp_unique[df_temp_unique['Country'] == country].loc[(slice(country),slice('1890','1910')),:]['MeanTemp'].mean(); # metric 20YRA representing metric THEN

    df_mean = df_temp_unique[df_temp_unique['Country'] == country]['MeanTemp'].mean(); 

    Metric.append(((df_now - df_then))) # Calculate the difference between then and now, and divide by then to get percentage change.
df_now = df_temp_unique.loc[('Belgium',slice('1990-12-31','2010-12-31')),:][['MeanTemp','Spread']].mean() 

df_then = df_temp_unique.loc[('Belgium',slice('1890-12-31','1910-12-31')),:][['MeanTemp','Spread']].mean()

#df_temp_unique.loc[(slice(None), slice('2010-12-31','2012-12-31')),:]



# Alternitively

df_now_alt = df_temp_unique[df_temp_unique['Country'] == country].loc[(slice(country),slice('1990-12-31','2010-12-31')),:]['MeanTemp'].mean()

df_then_alt = df_temp_unique[df_temp_unique['Country'] == country].loc[(slice(country),slice('1890-12-31','1910-12-31')),:]['MeanTemp'].mean()
data = [ dict(

        type = 'choropleth',

        locations = countries,

        z = Metric,

        locationmode = 'country names',

        text = countries,

        marker = dict(

            line = dict(color = 'rgb(0,0,0)', width = 1)),

            colorbar = dict(autotick = True, tickprefix = '', 

            title = '# \nTemperature,\n°C')

            )

       ]



''' # THIS IS FOR GLOBE REPRESENTATION

layout = dict(

    title = 'Average land temperature in countries',

    geo = dict(

        showframe = False,

        showocean = True,

        oceancolor = 'rgb(0,255,255)',

        projection = dict(

        type = 'orthographic',

            rotation = dict(

                    lon = 60,

                    lat = 10),

        ),

        lonaxis =  dict(

                showgrid = True,

                gridcolor = 'rgb(102, 102, 102)'

            ),

        lataxis = dict(

                showgrid = True,

                gridcolor = 'rgb(102, 102, 102)'

                )

            ),

        )

'''

# This is for a flat map representation

layout = dict(

    title = 'Change in Mean temperature over 100 years (20YRA)',

    geo = dict(

        showframe = False,

        showocean = True,

        oceancolor = 'rgb(0,225,255)',

        type = 'equirectangular'

    ),

)



fig = dict(data=data, layout=layout)

py.iplot(fig, validate=False, filename='world_temp_map')
global_temp_df.tail(5)
# We perform the same types of data manipulation as before to get a dataframe that has the right structure to make some nice plots.



df_mean_global = global_temp_df.resample('Y').mean()

df_max_global = global_temp_df.resample('Y').max()

df_min_global = global_temp_df.resample('Y').min()



df_temp_global = df_mean_global

df_temp_global['LandMaxTemperature'] = df_max_global.LandMaxTemperature

df_temp_global['LandMaxTemperatureUncertainty'] = df_max_global.LandMaxTemperatureUncertainty

df_temp_global['LandMinTemperature'] = df_min_global.LandMinTemperature

df_temp_global['LandMinTemperatureUncertainty'] = df_min_global.LandMinTemperatureUncertainty



# make the conf interval columns with measurement uncertainties. Nore that this will increase the number of columns significantly

df_temp_global['LandAverageTemperature+'] = df_temp_global.LandAverageTemperature + df_temp_global.LandAverageTemperatureUncertainty

df_temp_global['LandAverageTemperature-'] = df_temp_global.LandAverageTemperature - df_temp_global.LandAverageTemperatureUncertainty



df_temp_global['LandMaxTemperature+'] = df_temp_global.LandMaxTemperature + df_temp_global.LandMaxTemperatureUncertainty

df_temp_global['LandMaxTemperature-'] = df_temp_global.LandMaxTemperature - df_temp_global.LandMaxTemperatureUncertainty



df_temp_global['LandMinTemperature+'] = df_temp_global.LandMinTemperature + df_temp_global.LandMinTemperatureUncertainty

df_temp_global['LandMinTemperature-'] = df_temp_global.LandMinTemperature - df_temp_global.LandMinTemperatureUncertainty
# how does the dataframe look like.

df_temp_global.tail(10)
rolling_years = 10



plt.figure();

df_temp_global[['LandAverageTemperature+', 'LandAverageTemperature', 'LandAverageTemperature-',

               'LandMaxTemperature+', 'LandMaxTemperature', 'LandMaxTemperature-',

               'LandMinTemperature+', 'LandMinTemperature', 'LandMinTemperature-']].rolling(rolling_years).mean().plot(figsize=(25,7.5));

plt.grid();

plt.xlabel('Year', fontsize=20);

plt.ylabel('Temperature in Celcius', fontsize=20);

plt.title('10 year rolling average for global temperatures', fontsize=24);





plt.figure();

df_temp_global[['LandAverageTemperature+', 'LandAverageTemperature', 'LandAverageTemperature-']].rolling(rolling_years).mean().plot(figsize=(25,7.5));

plt.title('10 year rolling average for average global temperature', fontsize=24)

plt.xlabel('Year', fontsize=20)

plt.ylabel('Temperature in Celcius', fontsize=20)

plt.grid()



plt.figure();

df_temp_global[['LandMaxTemperature+', 'LandMaxTemperature', 'LandMaxTemperature-']].rolling(rolling_years).mean().plot(figsize=(25,7.5));

plt.title('10 year rolling average for maximum global temperature', fontsize=24)

plt.xlabel('Year', fontsize=20)

plt.ylabel('Temperature in Celcius', fontsize=20)

plt.grid()



plt.figure();

df_temp_global[['LandMinTemperature+', 'LandMinTemperature', 'LandMinTemperature-']].rolling(rolling_years).mean().plot(figsize=(25,7.5));

plt.title('10 year rolling average for minimum global temperature', fontsize=24)

plt.xlabel('Year', fontsize=20)

plt.ylabel('Temperature in Celcius', fontsize=20)

plt.grid()
