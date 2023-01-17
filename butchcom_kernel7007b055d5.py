import math

from os.path import join

import requests

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import plotly.express as px

from sklearn.linear_model import LinearRegression

from bs4 import BeautifulSoup      

from lxml import html

from lxml.etree import tostring, parse

from scipy.stats.mstats import kruskalwallis

from scipy.stats.stats import pearsonr



sns.set(style="whitegrid")

%matplotlib inline
plt.rcParams['figure.figsize'] = (14, 8)
def simpleLinearReg(values, numberOfValues, useNumberOfValues, startYear, stopYear, stepsize):

    """

    Helper function for extrapolating the values for 2018 and 2019 in WorldBank_Data and Eurostat_Data.

    values is a 2D numpy array with year and value

    """

    npNAN = np.empty([1, stopYear - startYear])

    npNAN[:] = np.nan

    

    if(useNumberOfValues < 0): # Only if it is specified to take last X numbers ; 0 indicates to take all available values

        values = values[useNumberOfValues:]

        

    if(len(values) >= numberOfValues):

        x = values[:,0].reshape((-1, 1))

        y = values[:,1]

        x_pred = np.array(np.arange(startYear, stopYear, stepsize)).reshape((-1, 1))

        model = LinearRegression().fit(x, y)

        y_new = model.predict(x_pred)

        return np.array(y_new)

    return npNAN[0]





def loadCSVFile(filename, delimiter, dec, skip, naValues, dataTypes):

    """

    Helper function to read csv

    """

    data = pd.read_csv(

        csvFolder + filename,             

        sep = delimiter,

        decimal = dec,

        skiprows = skip,                         

        na_values = naValues,                    

        #quotechar="'",                      

        dtype = dataTypes,           

        #usecols=['column_names'],

        #parse_dates=['column_names'],           

        encoding = 'latin-1'

    )

    return data



    

class DatasetWrapper:

    """

    Utility class wrapper to query a given dataset. 

    Most datasets about life expectancy have a similar structure.

    

    E.g. very often we want to query for specific countries. 

    This Wrapper class aims to avoid duplicate code.

    

    Args:

      data: Pandas dataframe.

      country_col: String. Name of country column.

      

    """

    def __init__(self, data, country_col):

        self.data = data

        self.country_col = country_col

        

    def get_countries(self, countries):

        if not isinstance(countries, list) or len(countries) == 1:

            return self.data[self.data[self.country_col] == countries]

        return pd.concat([self.data[self.data[self.country_col] == c] for c in countries]) 

    

    def get_col_instances(self, col, instances, countries=None):

        if not isinstance(instances, list):

            instances = [instances]

        if countries and not isinstance(countries, list):

            countries = [countries]



        dfs = []

        if countries:

            for country in countries:

                country_stats = self.get_countries(country)

                for instance in instances:

                    dfs.append(country_stats[country_stats[col] == instance])

        else:

            for instance in instances:

                dfs.append(self.data[self.data[col] == instance])

        return pd.concat(dfs)
pd.set_option('display.max_columns',None)

pd.set_option('display.max_rows', None)

csvFolder = '../input/data/comparison/'
# https://countrycode.org/

Country_Codes = loadCSVFile('Country_Codes.csv', 

                                           ';',

                                           '.',

                                           0, 

                                           [''], 

                                           {'Country': str, 'ISO2': str, 'ISO3': str, 'ISONumeric': int})



# https://datahelpdesk.worldbank.org/knowledgebase/articles/906519

WB_IncomeLevel = loadCSVFile('WorldBank_IncomeLevel.csv', 

                                           ';',

                                           '.',

                                           0, 

                                           [], 

                                           {'Code': str, 'Country': str, 'Income_Level': str})



# https://datahelpdesk.worldbank.org/knowledgebase/articles/906519

WB_GDP = loadCSVFile('WorldBank_GDP.csv', 

                                           ';',

                                           ',',

                                           0, 

                                           [], 

                                           {})



# https://data.oecd.org/healthstat/life-expectancy-at-birth.htm

OECD_Data = loadCSVFile('OECD_Data.csv', 

                        ',',

                         '.',

                         0, 

                         [], 

                         {'LOCATION': str, 'INDICATOR': str, 'SUBJECT': str, 'MEASURE': str, 'FREQUENCY': str, 'TIME': int, 'Value': float, 'Flag': str, 'Codes': str})



# https://population.un.org/wpp/Download/Standard/Mortality/

UN_Data = loadCSVFile('UN_Data.csv', 

                        ';',

                         ',',

                         0, 

                         [], 

                         {'Country': str, 'Country code': int, 'Sex': str})



# http://apps.who.int/gho/data/node.main.688?lang=en

WHO_Data = loadCSVFile('WHO_Data.csv', 

                        ';',

                         ',',

                         0, 

                         [''], 

                         {})



# https://data.worldbank.org/indicator/SP.DYN.LE00.FE.IN

WorldBank_Data = loadCSVFile('WorldBank_Data.csv', 

                        ';',

                         ',',

                         0, 

                         [''], 

                         {})



# https://ec.europa.eu/eurostat/web/population-demography-migration-projections/data/database

Eurostat_Data = loadCSVFile('Eurostat_Data.csv', 

                        ';',

                         '.',

                         0, 

                         [':'], 

                         {})





# Drop unneeded columns

OECD_Data.drop(OECD_Data.columns[[1,3,4,7]], axis = 1, inplace = True)

# 2018 and 2019 no values, only NA

WorldBank_Data.drop(['2018','2019'], axis = 1, inplace = True)

#WB_GDP.drop(WB_GDP.columns[[-1]], axis = 1, inplace = True)     # Altough there is a column for 2019, it contains no values



# Replace values so it is equal in all data frames

OECD_Data['SUBJECT'] = OECD_Data['SUBJECT'].map({'MEN': 'M', 'WOMEN': 'F', 'TOT': 'T'})



# For Eurostat data set

# Take the life expectancy values at year 1 because we are analysing life expactancy at birth (there is no year 0 in the data set)

Eurostat_Data = Eurostat_Data.loc[Eurostat_Data['Age'] == 'Y1']

# We are working with Female and male samples

Eurostat_Data = Eurostat_Data.loc[(Eurostat_Data['Sex'] == 'F') | (Eurostat_Data['Sex'] == 'M')]

# Eurostat data has to filtered more because there are a lot of specific rows including old European territories and aggregated life expectancis for different European areas

# For Germany (DE) there are two rows, DE (until 1990 former territory of the FRG) and DE_TOT (including former GDR).

# We keep DE_TOT and map it as DE because the values until 1992 are the same between DE and DE_TOT. Furthermore, DE_TOT includes values after 1990 which gives a more complete view of Germany despite including former GDR after 1990.

Eurostat_Data = Eurostat_Data.loc[(Eurostat_Data['Code'] != 'DE')]

Eurostat_Data = Eurostat_Data.replace(to_replace = 'DE_TOT', value = 'DE') 

                                  

Eurostat_Data = Eurostat_Data.loc[(Eurostat_Data['Code'] != 'EU28')]   # European Union 28 countries

Eurostat_Data = Eurostat_Data.loc[(Eurostat_Data['Code'] != 'EU27')]   # European Union 27 countries

Eurostat_Data = Eurostat_Data.loc[(Eurostat_Data['Code'] != 'EA19')]   # Euro area 19 countries

Eurostat_Data = Eurostat_Data.loc[(Eurostat_Data['Code'] != 'EA18')]   # Euro area 18 countries

Eurostat_Data = Eurostat_Data.loc[(Eurostat_Data['Code'] != 'FX')]     # Includes only France metropolitan

Eurostat_Data = Eurostat_Data.loc[(Eurostat_Data['Code'] != 'EEA31')]  # European Economic Area with 31 countries

Eurostat_Data = Eurostat_Data.loc[(Eurostat_Data['Code'] != 'EEA30')]  # European Economic Area with 30 countries

Eurostat_Data = Eurostat_Data.loc[(Eurostat_Data['Code'] != 'EFTA')]   # European Free Trade Association



# The Eurostat data set contains Great Britian and Greece with ISO2 Country Codes as UK and EL.

# According to iso.org UK is 'exceptionally reserved' and in other data sets the more common country code GB is used.

# Therefore, UK is replaced by GB.

# It is interesting that the country code for Greece is EL which should be GR according to iso.org - is also replaced.

# See: https://www.iso.org/obp/ui/#iso:code:3166:UK

# See: https://www.iso.org/obp/ui/#iso:code:3166:GB

# See: https://www.iso.org/obp/ui/#iso:code:3166:GR



# Furthermore, for Kosovo the country code XK is used but there is no ISO-Code for Kosovo according to iso.org

Eurostat_Data = Eurostat_Data.replace(to_replace = 'UK', value = 'GB')

Eurostat_Data = Eurostat_Data.replace(to_replace = 'EL', value = 'GR')



# WHO Data

# Replace values so it is equal in all data frames

WHO_Data['Sex'] = WHO_Data['Sex'].map({'Male': 'M', 'Female': 'F', 'Both sexes': 'T'})



#Rename Columns such as Country names and codes

OECD_Data.columns = ['ISO3', 'Sex', 'Year', 'Value']

UN_Data.rename(columns={'Country code': 'ISONumeric'}, inplace=True)

WorldBank_Data.rename(columns={'Country Name': 'Country', 'Country Code': 'ISO3'}, inplace=True)

Eurostat_Data.rename(columns={'Code': 'ISO2'}, inplace=True)



# Reset all indexes to get a clean index. Problems arrise when drooping rows with loc

OECD_Data.reset_index(drop=True, inplace=True)

UN_Data.reset_index(drop=True, inplace=True)

WHO_Data.reset_index(drop=True, inplace=True)

WorldBank_Data.reset_index(drop=True, inplace=True)

Eurostat_Data.reset_index(drop=True, inplace=True)
# World Bank Data

# Extrapolate the data for World Bank data set



worldBank_years = np.arange(1960,2018,1)

worldBank_values = [row[3:] for row in WorldBank_Data[:].values]

worldBank_preds = []

for row in worldBank_values:

    dummy_values = pd.DataFrame({'Year': worldBank_years[:], 'Value': row[:]}) # concat years and values

    dummy_values = np.array(dummy_values.dropna()) # Drop NAs

    worldBank_preds.append(simpleLinearReg(dummy_values, 10, -15, 2018, 2020, 1)) # use only the last 15 values to get more accurate extrapolations for 2018 and 2019



worldBank_preds = pd.DataFrame(worldBank_preds)

worldBank_preds.rename(columns={0: '2018', 1: '2019'}, inplace=True)

WorldBank_Data = pd.concat([WorldBank_Data, worldBank_preds], axis=1, sort=False)

#WorldBank_Data



# Eurostat Data

# Extrapolate the data for Eurostat data set



cols = Eurostat_Data.columns.tolist()

sameOrder = cols[0:3]

years = cols[::-1][:-3]    # reverse all columns so it starts with 1960. Get rid of columns in sameOrder with [:-3]

cols = sameOrder + years

Eurostat_Data = Eurostat_Data[cols]



eurostat_years = np.arange(1960,2018,1)

eurostat_values = [row[3:] for row in Eurostat_Data[:].values]

eurostat_preds = []

for row in eurostat_values:

    dummy_values = pd.DataFrame({'Year': eurostat_years[:], 'Value': row[:]}) # concat years and values

    dummy_values = np.array(dummy_values.dropna()) # Drop NAs

    

    eurostat_preds.append(simpleLinearReg(dummy_values, 5, -5, 2018, 2020, 1)) # use only the last 5 values. (last 5 because Eurostat does not have alwys that much values for every country)



eurostat_preds = pd.DataFrame(eurostat_preds)

eurostat_preds.rename(columns={0: '2018', 1: '2019'}, inplace=True)

Eurostat_Data = pd.concat([Eurostat_Data, eurostat_preds], axis=1, sort=False)



#Eurostat_Data



# Reshape OECD Data to be able to extrapolate values for 2018 and 2019



dummy1 = OECD_Data.loc[OECD_Data['Sex'] == 'F']

dummy2 = OECD_Data.loc[OECD_Data['Sex'] == 'M']

dummy3 = OECD_Data.loc[OECD_Data['Sex'] == 'T']



dummy1 = dummy1.pivot(index='ISO3', columns='Year', values='Value')

dummy2 = dummy2.pivot(index='ISO3', columns='Year', values='Value')

dummy3 = dummy3.pivot(index='ISO3', columns='Year', values='Value')



dummy1['Sex'] = 'F'

dummy2['Sex'] = 'M'

dummy3['Sex'] = 'T'



dummycolumns = dummy1.columns.tolist()

dummycolumns = ['Sex'] + dummycolumns[:-1]

dummy1 = dummy1[dummycolumns]

dummy2 = dummy2[dummycolumns]

dummy3 = dummy3[dummycolumns]



OECD_Data_reshaped = pd.concat([dummy1, dummy2, dummy3])

OECD_Data_reshaped = OECD_Data_reshaped.reset_index()

OECD_Data_reshaped.columns.name = None



# Extrapolate OECD Data for 2018 and 2019



OECD_Data_reshaped_years = np.arange(1960,2018,1)

OECD_Data_reshaped_values = [row[2:] for row in OECD_Data_reshaped[:].values]

OECD_Data_reshaped_preds = []

counter = 0

for row in OECD_Data_reshaped_values:

    dummy_values = pd.DataFrame({'Year': OECD_Data_reshaped_years[:], 'Value': row[:]}) # concat years and values

    dummy_values = np.array(dummy_values.dropna()) # Drop NAs

    

    OECD_Data_reshaped_preds.append(simpleLinearReg(dummy_values, 10, -15, 2018, 2020, 1))



OECD_Data_reshaped_preds = pd.DataFrame(OECD_Data_reshaped_preds)

OECD_Data_reshaped_preds.rename(columns={0: '2018', 1: '2019'}, inplace=True)

OECD_Data_reshaped = pd.concat([OECD_Data_reshaped, OECD_Data_reshaped_preds], axis=1, sort=False)



#OECD_Data_reshaped



# WHO Data

# Reshape WHO Data so we can extrapolate the values



dummy1 = WHO_Data.loc[WHO_Data['Sex'] == 'T']

dummy2 = WHO_Data.loc[WHO_Data['Sex'] == 'F']

dummy3 = WHO_Data.loc[WHO_Data['Sex'] == 'M']



dummy1 = dummy1.pivot(index='ISO3', columns='Year', values='Values')

dummy2 = dummy2.pivot(index='ISO3', columns='Year', values='Values')

dummy3 = dummy3.pivot(index='ISO3', columns='Year', values='Values')



dummy1['Sex'] = 'T'

dummy2['Sex'] = 'F'

dummy3['Sex'] = 'M'



dummycolumns = dummy1.columns.tolist()

dummycolumns = ['Sex'] + dummycolumns[:-1]

dummy1 = dummy1[dummycolumns]

dummy2 = dummy2[dummycolumns]

dummy3 = dummy3[dummycolumns]



WHO_Data_reshaped = pd.concat([dummy1, dummy2, dummy3])

WHO_Data_reshaped = WHO_Data_reshaped.reset_index()

WHO_Data_reshaped.columns.name = None



# Extrapolate WHO Data for 2017,2018, and 2019



WHO_Data_reshaped_years = np.arange(2000,2017,1)

WHO_Data_reshaped_values = [row[2:] for row in WHO_Data_reshaped[:].values]

WHO_Data_reshaped_preds = []

counter = 0

for row in WHO_Data_reshaped_values:

    dummy_values = pd.DataFrame({'Year': WHO_Data_reshaped_years[:], 'Value': row[:]}) # concat years and values

    dummy_values = np.array(dummy_values.dropna()) # Take all available values.

    

    WHO_Data_reshaped_preds.append(simpleLinearReg(dummy_values, 5, 0, 2017, 2020, 1))



WHO_Data_reshaped_preds = pd.DataFrame(WHO_Data_reshaped_preds)

WHO_Data_reshaped_preds.rename(columns={0: '2017', 1: '2018', 2: '2019'}, inplace=True)

WHO_Data_reshaped = pd.concat([WHO_Data_reshaped, WHO_Data_reshaped_preds], axis=1, sort=False)



#WHO_Data_reshaped
OECD_Data_reshaped = pd.merge(OECD_Data_reshaped, Country_Codes, on='ISO3', how='left')

UN_Data = pd.merge(UN_Data, Country_Codes, on='ISONumeric', how='left')

WHO_Data_reshaped = pd.merge(WHO_Data_reshaped, Country_Codes, on='ISO3', how='left')

WorldBank_Data = pd.merge(WorldBank_Data, Country_Codes, on='ISO3', how='left')

Eurostat_Data = pd.merge(Eurostat_Data, Country_Codes, on='ISO2', how='left')
Vis_LifeExpectancy = OECD_Data_reshaped.drop(OECD_Data_reshaped.columns.difference(['ISONumeric','Sex','2019']), 1)

Vis_LifeExpectancy = Vis_LifeExpectancy.loc[Vis_LifeExpectancy['Sex'] != 'T'].reset_index(drop=True)

Vis_LifeExpectancy.rename(columns = {'2019': 'OECD'}, inplace=True)



mergerUN = UN_Data.drop(UN_Data.columns.difference(['ISONumeric','Sex','2015-2020']), 1).reset_index(drop=True)

mergerUN.rename(columns = {'2015-2020': 'UN'}, inplace=True)



mergerWHO = WHO_Data_reshaped.drop(WHO_Data_reshaped.columns.difference(['ISONumeric','Sex','2019']), 1)

mergerWHO = mergerWHO.loc[mergerWHO['Sex'] != 'T'].reset_index(drop=True)

mergerWHO.rename(columns = {'2019': 'WHO'}, inplace=True)



mergerWorldBank = WorldBank_Data.drop(WorldBank_Data.columns.difference(['ISONumeric','Sex','2019']), 1).reset_index(drop=True)

mergerWorldBank.rename(columns = {'2019': 'WorldBank'}, inplace=True)



mergerEuroStat = Eurostat_Data.drop(Eurostat_Data.columns.difference(['ISONumeric','Sex','2019']), 1).reset_index(drop=True)

mergerEuroStat.rename(columns = {'2019': 'Eurostat'}, inplace=True)



Vis_LifeExpectancy = pd.merge(Vis_LifeExpectancy, mergerUN, on=['ISONumeric','Sex'], how='outer')

Vis_LifeExpectancy = pd.merge(Vis_LifeExpectancy, mergerWHO, on=['ISONumeric','Sex'], how='outer')

Vis_LifeExpectancy = pd.merge(Vis_LifeExpectancy, mergerWorldBank, on=['ISONumeric','Sex'], how='outer')

Vis_LifeExpectancy = pd.merge(Vis_LifeExpectancy, mergerEuroStat, on=['ISONumeric','Sex'], how='outer')



# Merge all country information

Vis_LifeExpectancy = pd.merge(Vis_LifeExpectancy, Country_Codes, on='ISONumeric', how='left')



# Calculate mean of values gruped by 'ISONumeric'.

# So we get mean values of male and female for all countries

meanValues = Vis_LifeExpectancy.groupby(['ISONumeric']).mean()

meanValues['Sex'] = 'A' # Name the new averaged category as A (Average)

meanValues['ISONumeric'] = meanValues.index

meanValues.reset_index(drop=True, inplace = True)

meanValues = pd.merge(meanValues, Country_Codes, on='ISONumeric', how='left')

frames = [Vis_LifeExpectancy, meanValues]

Vis_LifeExpectancy = pd.concat(frames, sort=False).reset_index(drop=True)
# Average all life expectancy values of all sources row wise

# To be able to do this correct, drop column ISONumeric

LifeExpectancy_Avg = Vis_LifeExpectancy.drop(Vis_LifeExpectancy.columns[[2]], 1)

LifeExpectancy_Avg = LifeExpectancy_Avg[LifeExpectancy_Avg['Sex'] == 'A'].reset_index(drop=True)

LifeExpectancy_Avg['LifeExpectancy_Average'] = LifeExpectancy_Avg.mean(axis=1)



# Get column ISONumeric back

cc = Country_Codes.drop(Country_Codes.columns[[0,1]], axis=1)

LifeExpectancy_Avg = pd.merge(LifeExpectancy_Avg, cc, on='ISO3', how='outer')



# Merge Income levels and values

IncomeLevel2018 = WB_GDP.drop(WB_GDP.columns.difference(['ISO3','2018']), 1)

IncomeLevel2018.columns = ['ISO3', 'WB_GDP_2018']

Vis_LifeExpectancy = pd.merge(Vis_LifeExpectancy, IncomeLevel2018, on=['ISO3'], how='outer')

Vis_LifeExpectancy = pd.merge(Vis_LifeExpectancy, WB_IncomeLevel, on=['ISO3'], how='outer')



# At this point we have 223 entries in the data frame Vis_LifeExpectancy.

# For better visualizations we drop the columns OECD and Eurostat because these sources cointain few life expectancy values compared to UN, WHO, and WorldBank

LineChart = Vis_LifeExpectancy.drop(Vis_LifeExpectancy.columns[[1,6]], axis = 1)



# To get more clear line chart we drop also countries without values and take randomly a subset of countries including Austria

LineChart = Vis_LifeExpectancy[Vis_LifeExpectancy['Sex'] == 'A'].reset_index(drop=True)

LineChart = LineChart.drop(LineChart.columns.difference(['UN','WHO','WorldBank','Country']), axis = 1)

LineChart = LineChart.dropna().reset_index(drop=True)

rowAustria = LineChart[LineChart['Country'] == 'Austria']

LineChart = LineChart.sample(n = 45, random_state = 4711).reset_index(drop=True)

LineChart = pd.concat([rowAustria,LineChart])



# Sort values according to WorldBank and plot a line chart

LineChart.sort_values('WorldBank', ascending=True, inplace = True)

LineChart = LineChart.reset_index(drop=True)

ax = LineChart.plot(title='Life expectancy - Average values of female and male 2019', rot=90, legend='best', figsize=(20, 10))



# Create X-Ticks

ticklabels = LineChart['Country'].values

tickindex = np.arange(0,len(ticklabels),1)

ax.set_xticks(tickindex)

ax.set_xticklabels(ticklabels)



mylabels = ['UN', 'WHO', 'World Bank']

ax.legend(labels=mylabels, title='Datasets')



# Plot vertical line at the position of Austria

plt.axvline(x=np.where(ticklabels == 'Austria'), color='orange',linestyle='--')
import plotly.graph_objects as go



WorldBank_Animation = WorldBank_Data.drop(['Country_y', 'ISO2', 'ISONumeric'], axis = 1, inplace = False)

WorldBank_Animation_Men = WorldBank_Animation.loc[WorldBank_Animation['Sex'] == 'M']

WorldBank_Animation_Women = WorldBank_Animation.loc[WorldBank_Animation['Sex'] == 'F']



years = list(range(1960, 2020))

years = [str(i) for i in years]



WorldBank_Animation_Men = WorldBank_Animation_Men.melt(

    id_vars=['Country_x', 'ISO3', 'Sex'],

    value_vars=years,

    var_name='Year', value_name='LifeExpectancy'

)



WorldBank_Animation_Women = WorldBank_Animation_Women.melt(

    id_vars=['Country_x', 'ISO3', 'Sex'],

    value_vars=years,

    var_name='Year', value_name='LifeExpectancy'

)



animation_Men = px.choropleth(WorldBank_Animation_Men, locations="ISO3",

                    color="LifeExpectancy", # LifeExpectancy is a column of gapminder

                    hover_name="Country_x", # Country_x to add to hover information

                    color_continuous_scale='Inferno',

                    animation_frame="Year",

                    animation_group="Country_x",

                    range_color=[20,90]

                   )



animation_Men = animation_Men.update_layout(

    title_text = 'Life expectancy from 1960 to 2019 for Men'

)



animation_Women = px.choropleth(WorldBank_Animation_Women, locations="ISO3",

                    color="LifeExpectancy", # LifeExpectancy is a column of gapminder

                    hover_name="Country_x", # Country_x to add to hover information

                    color_continuous_scale='Inferno',

                    animation_frame="Year",

                    animation_group="Country_x",

                    range_color=[20,90]

                   )

animation_Women = animation_Women.update_layout(

    title_text = 'Life expectancy from 1960 to 2019 for Women'

)
animation_Men.show()
animation_Women.show()
differences = Vis_LifeExpectancy.drop(Vis_LifeExpectancy.columns.difference(['UN','WHO','WorldBank','Sex','Country']), axis = 1) 

sns.pairplot(differences, hue = 'Sex')
plt.figure(figsize=(5,3))

correlations = differences.corr(method = "spearman").abs()

sns.heatmap(correlations,cmap='BrBG',annot=True)

plt.title('Correlation plot')
# Print Box-Plots

# Visually the WHO and WorldBank data sets are very similar for both females and males

differences.groupby('Sex').boxplot(fontsize=20,rot=90,figsize=(20,15),patch_artist=True)
import statsmodels.api as sm

from statsmodels.formula.api import ols



stackedValues = differences.drop(differences.columns[[0]], axis = 1)

keys = ['UN','WHO','WorldBank']

stackedValues = pd.melt(stackedValues, id_vars='Country', value_vars=keys, value_name='lifeexpectancy')

stackedValues = stackedValues.drop(stackedValues.columns[[0]], axis = 1)



mod = ols('lifeexpectancy ~ variable', data=stackedValues).fit()

                

aov_table = sm.stats.anova_lm(mod, typ=2)

print(aov_table)
IncomeLevel2018 = WB_GDP.drop(WB_GDP.columns.difference(['ISO3','2018']), 1)

IncomeLevel2018.columns = ['ISO3', 'WB_GDP_2018']



LifeExpectancy_IncomeLevel = pd.merge(LifeExpectancy_Avg, IncomeLevel2018, on=['ISO3'], how='outer')

LifeExpectancy_IncomeLevel = pd.merge(LifeExpectancy_IncomeLevel, WB_IncomeLevel, on=['ISO3'], how='outer')



plt.figure(figsize = (18, 8))

sns.scatterplot(data = LifeExpectancy_IncomeLevel,

               x = 'WB_GDP_2018', 

               y = 'LifeExpectancy_Average',

               hue = 'Income_Level', 

               style = 'Income_Level') 

plt.title('Average life expectancy to GDP')

plt.xlabel('Per Capita GDP at constant 2010 prices in US Dollars')

plt.ylabel('Life expectancy in years')



gdpAustria = LifeExpectancy_IncomeLevel.loc[LifeExpectancy_IncomeLevel['Country'] == 'Austria', 'WB_GDP_2018'].iloc[0]

lfAusria = LifeExpectancy_IncomeLevel.loc[LifeExpectancy_IncomeLevel['Country'] == 'Austria', 'LifeExpectancy_Average'].iloc[0]



plt.annotate('Austria', # this is the text

                 (gdpAustria,lfAusria), # this is the point to label

                 textcoords="offset points", # how to position the text

                 xytext=(0,0), # distance from text to points (x,y)

                 ha='left',

                 color='orange', fontsize=20) # horizontal alignment can be left, right or center
LifeExpectancy_IncomeLevelHI = LifeExpectancy_IncomeLevel[LifeExpectancy_IncomeLevel['Income_Level'] == 'High income']

LifeExpectancy_IncomeLevelUM = LifeExpectancy_IncomeLevel[LifeExpectancy_IncomeLevel['Income_Level'] == 'Upper middle income']

LifeExpectancy_IncomeLevelLM = LifeExpectancy_IncomeLevel[LifeExpectancy_IncomeLevel['Income_Level'] == 'Lower middle income']

LifeExpectancy_IncomeLevelLI = LifeExpectancy_IncomeLevel[LifeExpectancy_IncomeLevel['Income_Level'] == 'Low income']



a = sns.jointplot(x=LifeExpectancy_IncomeLevelHI['WB_GDP_2018'], y=LifeExpectancy_IncomeLevelHI['LifeExpectancy_Average'],data=LifeExpectancy_IncomeLevelHI,kind='kde')

a.fig.suptitle('High Income')



b = sns.jointplot(x=LifeExpectancy_IncomeLevelUM['WB_GDP_2018'], y=LifeExpectancy_IncomeLevelUM['LifeExpectancy_Average'],data=LifeExpectancy_IncomeLevelUM,kind='kde')

b.fig.suptitle('Upper middle income')



c = sns.jointplot(x=LifeExpectancy_IncomeLevelLM['WB_GDP_2018'], y=LifeExpectancy_IncomeLevelLM['LifeExpectancy_Average'],data=LifeExpectancy_IncomeLevelLM,kind='kde')

c.fig.suptitle('Lower middle income')



d = sns.jointplot(x=LifeExpectancy_IncomeLevelLI['WB_GDP_2018'], y=LifeExpectancy_IncomeLevelLI['LifeExpectancy_Average'],data=LifeExpectancy_IncomeLevelLI,kind='kde')

d.fig.suptitle('Low Income')



#sns.jointplot(x=LifeExpectancy_IncomeLevelHI['WB_GDP_2018'], y=LifeExpectancy_IncomeLevelHI['LifeExpectancy_Average'],data=LifeExpectancy_IncomeLevelHI,kind='reg')
# Main Source https://stackoverflow.com/questions/51210955/seaborn-jointplot-add-colors-for-each-class



g = sns.jointplot(LifeExpectancy_IncomeLevel['WB_GDP_2018'], LifeExpectancy_IncomeLevel['LifeExpectancy_Average'], height=7, kind='reg', scatter = False)

for i, subdata in LifeExpectancy_IncomeLevel.groupby('Income_Level'):

    sns.kdeplot(subdata.iloc[:,11], ax=g.ax_marg_x, legend=False)

    sns.kdeplot(subdata.iloc[:,9], ax=g.ax_marg_y, vertical=True, legend=False)

    g.ax_joint.plot(subdata.iloc[:,11], subdata.iloc[:,9], "o", ms = 8)



plt.tight_layout()

plt.xlim(-10000, 130000)

plt.ylim(50, 90)



# Plot vertical line at the position of Austria

gdpAustria = LifeExpectancy_IncomeLevel.loc[LifeExpectancy_IncomeLevel['Country'] == 'Austria', 'WB_GDP_2018'].iloc[0]

lfAusria = LifeExpectancy_IncomeLevel.loc[LifeExpectancy_IncomeLevel['Country'] == 'Austria', 'LifeExpectancy_Average'].iloc[0]

plt.axvline(x=gdpAustria, color='orange',linestyle='--')

plt.axhline(y=lfAusria, color='orange',linestyle='--')



plt.show()
# Worldbank population density - https://datacatalog.worldbank.org/search/indicators?search_api_views_fulltext_op=AND&query=en.pop.dnst

wb_pd_data = pd.read_excel("../input/worldbank_popDensity.xlsx", index_col=None)



#wb_pd_data.columns
wb_pd_data.head()
wb_gdp = pd.read_csv(join(csvFolder, 'WorldBank_GDP.csv'), sep=';', decimal=',', skiprows=0, encoding = 'latin-1')

wb_gdp['Indicator Name'] = 'GDP'

## Rearrange columns for wb_gdp

wb_gdp.rename(columns={'ISO3': 'Country Code'}, inplace=True)

cols = list(wb_gdp.columns.values)

cols_new = cols[:2] + cols[-1:] + cols[2:-1]

wb_gdp = wb_gdp.reindex(columns=cols_new)



## Convert data-type of columns for wb_pd_data

cols = list(wb_pd_data.columns.values)

cols_new = cols[:3] + [str(c) for c in cols[3:]]

cols_new

wb_pd_data.columns = cols_new



## Append GDP-data to popDensity-data 

wb_data = wb_pd_data.append(wb_gdp, sort=False, ignore_index=True)



wb_data.shape

## wb_data with popDens and GDP ready
hnp_stats = pd.read_csv("../input/data/health-nutrition-and-population-stats-worldbank/HNP_StatsData.csv")

hnp_stats.drop(columns=['Indicator Code', 'Unnamed: 64'], inplace=True)

hnp_stats_wrapper = DatasetWrapper(hnp_stats, "Country Name")
col = "Indicator Name"

indicator = ["Life expectancy at birth, total (years)", 

            'Population, total',

            'Mortality rate, under-5 (per 1,000)',

            'Mortality rate, adult, female (per 1,000 female adults)',

            'Mortality rate, adult, male (per 1,000 male adults)',

            'Immunization, BCG (% of one-year-old children)',

            'Immunization, DPT (% of children ages 12-23 months)',

            'Immunization, HepB3 (% of one-year-old children)',

            'Immunization, Hib3 (% of children ages 12-23 months)',

            'Immunization, measles (% of children ages 12-23 months)',

            'Immunization, Pol3 (% of one-year-old children)']

wb_indicators_pop = hnp_stats_wrapper.get_col_instances(col, indicator)

wb_indicators_pop.columns = wb_pd_data.columns

wb_indicators_pop.shape

wb_stats = wb_data.append(wb_indicators_pop, sort=False, ignore_index=True)

wb_stats_wrapper = DatasetWrapper(wb_stats, "Country Name")



# wb_stats.to_csv('worldbank_stats.csv')

# wb_stats.shape

# wb_stats['Country Name'].unique().size

# wb_stats['Indicator Name'].unique()
wb_long = pd.melt(wb_stats, id_vars=wb_stats.columns[:3], value_vars =wb_stats.columns[3:])

wb_long.reset_index(inplace=True, drop=True)

wb_long.rename(columns={'variable': 'year',

                        'Country Name': 'country',

                        'Indicator Name': 'indicator'}, inplace=True)

print(wb_long.shape)

wb_long.head()
un_wpp_metadata = pd.read_excel("../input/data/un_health_population_prospects/WPP2019_F01_LOCATIONS.XLSX", skiprows=15, header=[0, 1]) 

un_wpp_metadata.columns = [' '.join(col).strip() if not col[0].startswith("Unnamed:") else col[1] for col in un_wpp_metadata.columns.values]



un_wpp_reduced = un_wpp_metadata.loc[:,['ISO3 Alpha-code', 'Geographic region Name']]

un_wpp_reduced.rename(columns={'ISO3 Alpha-code': 'Country Code',

                              'Geographic region Name': 'region'}, inplace=True)



wb_long = wb_long.merge(un_wpp_reduced, on='Country Code', how='left')
wb_income_level = pd.read_csv(join(csvFolder, 'WorldBank_IncomeLevel.csv'), sep=';', decimal='.',

                              dtype={'Code': str, 'Country': str, 'Income_Level': str}, encoding = 'latin-1')

wb_income_level.rename(columns={'ISO3': 'Country Code',

                               'Income_Level': 'income_level'}, inplace=True)



wb_long = wb_long.merge(wb_income_level, on='Country Code', how='left')                            
wb_long.rename(columns={'Country Code': 'country_code'}, inplace=True)

wb_long = wb_long[['country', 'country_code', 'region', 'income_level', 'year', 'indicator', 'value']]

wb_long.year = wb_long.year.astype('int32')



wb_long_wrapper = DatasetWrapper(wb_long, "country")



print(wb_long.shape)

print(wb_long.dtypes)

print('\nNumber of countries: {}'.format(len(wb_long.iloc[:,0].unique())))

wb_long.head()



## wb_long ready
indicator_map = {'Life expectancy at birth, total (years)': 'LEx',

            'Population density': 'pop_density',

            'Population, total': 'pop_total',

            'Mortality rate, under-5 (per 1,000)': 'mort_under_5',

            'Mortality rate, adult, female (per 1,000 female adults)': 'mort_adult_f',

            'Mortality rate, adult, male (per 1,000 male adults)': 'mort_adult_m',

            'Immunization, BCG (% of one-year-old children)': 'immun_BCG',

            'Immunization, DPT (% of children ages 12-23 months)': 'immun_DPT',

            'Immunization, HepB3 (% of one-year-old children)': 'immun_HepB3',

            'Immunization, Hib3 (% of children ages 12-23 months)': 'immun_Hib3',

            'Immunization, measles (% of children ages 12-23 months)': 'immun_Measles',

            'Immunization, Pol3 (% of one-year-old children)': 'immun_Pol3',

            'GDP': 'GDP'}



#wb_wide.rename(columns=indicator_map, inplace=True)

wb_long['indicator'] = wb_long['indicator'].map(indicator_map)

wb_wide = wb_long.pivot_table(index= ['country', 'year', 'region', 'income_level'], columns='indicator', values='value')

wb_wide.reset_index(inplace=True)

wb_wide_wrapper = DatasetWrapper(wb_wide, "country")



print(wb_wide.shape)

wb_wide.head()



## wide format available
print('Number of countries in wb_wide: {}'.format(len(wb_wide.iloc[:,0].unique())))
mean_years = wb_wide.groupby('year').mean()

mean_countries = wb_wide.groupby('country').mean()

mean_years.head()
mean_countries.head()
df_descr = wb_wide.describe().T

df_descr['median'] = wb_wide.median()



df_descr
corr_matrix = wb_wide.corr().abs().sort_values(by='LEx', ascending=False).T.sort_values(by='LEx', ascending=False)

f, ax = plt.subplots(figsize=(10, 9))

sns.heatmap(corr_matrix)

corr_matrix
## How is distribution of LEx in different regions



g = sns.FacetGrid(wb_wide[wb_wide.year == 2016], col="region")

g.map(plt.hist, 'LEx')
## Basic Scatterplots for every region



g = sns.FacetGrid(wb_wide[wb_wide.year == 2016], col='region')

g.map(plt.scatter, 'pop_density', 'LEx')
## Deciding outliers for population density



wb_wide['pop_density'].plot(kind='box', figsize=(2,4))

print(wb_wide['pop_density'].describe())

print(wb_wide[['country', 'pop_density']].sort_values(by='pop_density', ascending = False).head(290).tail())

wb_wide[['country', 'pop_density']].sort_values(by='pop_density', ascending = False).head(290).iloc[:,0].unique()
## Show historical development of LEx and pop_density

## pop_density < 2500 because only some Asian countries are higher



plt.figure(figsize=(17,12))

plot_data = wb_wide.loc[:,['country', 'region', 'income_level', 'year', 'pop_density', 'LEx']].dropna()

g = sns.scatterplot(x="pop_density", y='LEx', hue='region', data=plot_data[plot_data["pop_density"] < 2500])

plt.xlabel('Population density')

plt.ylabel('Life expectancy')



g.figure.savefig("popDens_LEx.png")
## Make Scatterplot interactive including hover



# fig = px.scatter(plot_data[plot_data["pop_density"] < 2500], x="pop_density", y='LEx', color="region",

#                  hover_name='country', hover_data=['year'], opacity=0.8, height=600, width=1000)

# fig.show()
plt.figure(figsize=(10,7))

plot_data = wb_wide[wb_wide.pop_density < 2500]

plot_data = plot_data[plot_data.year == 2017]

ax = sns.scatterplot(x='pop_total', y='pop_density', data=plot_data, size='pop_total')



plot_data.loc[:, ['pop_density', 'pop_total']].corr()
plot_data = wb_long_wrapper.get_col_instances('indicator', ["pop_density",

                                                            'LEx'], )

plot_data = plot_data[plot_data.year == 2017].pivot('indicator', 'country', 'value').T.dropna()



sns.regplot(x="pop_density", y='LEx',

            data=plot_data[plot_data["pop_density"] < 2500])



plot_data.corr()
plt.figure(figsize=(10,7))

plt.xticks(rotation=30)

sns.scatterplot(x=mean_years.index, y='pop_density', data=mean_years)
plt.figure(figsize=(10,7))

sns.regplot(x='pop_density', y='LEx', data=mean_years)

mean_years.loc[:,['pop_density', 'LEx']].corr()
wb_mortality = wb_wide[['country', 'year', 'region', 'income_level', 'mort_adult_f', 'mort_adult_m', 'mort_under_5', 'LEx', 'pop_total']]



year = 2015

wb_mort_year = wb_mortality[wb_mortality.year == year]
fig, ax = plt.subplots(figsize=(10,7))



ax = sns.scatterplot(x='mort_adult_f', y='mort_adult_m', data = wb_mort_year, marker='o', size = 'pop_total',

                sizes=(5,2000), alpha=0.5, hue='region', legend='brief', ax=ax)



# Separate legends for region and pop_total and show only legend for region 

h,l = ax.get_legend_handles_labels()

# color

col_lgd = plt.legend(h[:7], l[:7], loc='upper left', 

                     bbox_to_anchor=(1.02, 1.01), fancybox=True, shadow=False, ncol=1)

# # size

# size_lgd = plt.legend(h[-5:], l[-5:], loc='lower center', borderpad=1.6, prop={'size': 20},

#                       bbox_to_anchor=(0.88,-.35), fancybox=False, shadow=False, ncol=5)



# # add color

# plt.gca().add_artist(col_lgd)

# add line (1.median)

plt.plot([30, 480], [30, 480], color='k', linestyle='-.', linewidth=1)

plt.show()
wb_mort_year[['region', 'mort_adult_f', 'mort_adult_m', 'mort_under_5']].groupby('region').median()
def lin_reg(df, col):

    reg_data = df[[col, 'LEx']].dropna()



    try:

        X_train, X_test, Y_train, Y_test = train_test_split(reg_data[col].values.reshape(-1,1), reg_data['LEx'].values.reshape(-1,1), test_size=0.3, random_state=3)



        lm = LinearRegression()

        lm.fit(X_train,Y_train)

        predictions = lm.predict(X_test)



        ## Output of parameter values

        print('\n######## LEx and {} ########\n'.format(col.upper()))

        print('Number of nan-values in {}: {} out of {}'.format(col, df[col].isnull().sum(), int(df[col].shape[0])))

        print('\nCoefficients: slope = %6.3f, intercept = %6.3f' % (lm.coef_[0,0], lm.intercept_[0]))

        print('Coefficient of Determination R^2: %6.3f' % lm.score(X_train,Y_train))

        print('MAE:', metrics.mean_absolute_error(Y_test, predictions))

        print('MSE:', metrics.mean_squared_error(Y_test, predictions))

        print('R^2:', lm.score(X_train,Y_train))

        print(df.loc[:,[col, 'LEx']].corr())



        return lm.coef_[0,0], lm.intercept_[0]

    except:

        print('\n######## LEx and {} ########\n'.format(col.upper()))

        print('Not enough values in {}: {} out of {}'.format(col, df[col].notnull().sum(), int(df[col].shape[0])))

        

        return np.nan, 0



def plot_reg(df, col, slope, intercept):

    col_min = df[col].min()

    col_max = df[col].max()

    fig, ax = plt.subplots(figsize=(8,6))

    if not math.isnan(slope):

        plt.plot([col_min - 10, col_max + 10], [(col_min - 10)*slope + intercept, (col_max + 10)*slope + intercept], color='k', linestyle='-.', linewidth=1)

    ax = sns.scatterplot(x=col, y='LEx', data = df, marker='o', size = 'pop_total',

                    sizes=(5,2000), alpha=0.5, hue='region', legend='brief', ax=ax)



    # Separate legend components and just draw regions.

    # Location: outside of the plot

    h,l = ax.get_legend_handles_labels()

    col_lgd = plt.legend(h[:7], l[:7], loc='upper left', 

                         bbox_to_anchor=(1.02, 1.01), fancybox=True, shadow=False, ncol=1)

    ax.figure.savefig('LEx_{}.png'.format(col))
slope, intercept = lin_reg(wb_mort_year, 'mort_adult_f')

plot_reg(wb_mort_year, 'mort_adult_f', slope, intercept)
slope, intercept = lin_reg(wb_mort_year, 'mort_adult_m')

plot_reg(wb_mort_year, 'mort_adult_m', slope, intercept)
slope, intercept = lin_reg(wb_mort_year, 'mort_under_5')

plot_reg(wb_mort_year, 'mort_under_5', slope, intercept)
cols = ['mort_adult_f', 'mort_adult_m', 'mort_under_5', 'LEx']



wb_mortality.groupby('year').median()[cols].plot(figsize=(8,6), title=' Development of mortality over time')



years = wb_mortality.year.unique()



r_2 = dict()

for c in cols[:-1]:

    val = list()

    for y in years:

        reg_data = wb_mortality[wb_mortality.year == y].dropna()

        try:

            X_train, X_test, Y_train, Y_test = train_test_split(reg_data[c].values.reshape(-1,1), reg_data['LEx'].values.reshape(-1,1), test_size=0.3, random_state=3)

            lm = LinearRegression()

            lm.fit(X_train,Y_train)

            val.append(lm.score(X_train,Y_train))

        except:

            val.append(np.nan)

    r_2[c] = val

pd.DataFrame(r_2, index=years).plot(figsize=(8,6), title='R^2 of mortality over time')
wb_immun = wb_wide[['country', 'year', 'region', 'income_level', 'immun_BCG', 'immun_DPT', 'immun_HepB3', 'immun_Hib3', 'immun_Pol3', 'immun_Measles', 'LEx', 'pop_total']]

year = 2015

wb_immun_year = wb_immun[wb_immun.year == year]
cols = ['immun_BCG', 'immun_DPT', 'immun_HepB3', 'immun_Hib3', 'immun_Pol3', 'immun_Measles']



for c in cols:

    slope, intercept = lin_reg(wb_immun_year, c)

    plot_reg(wb_immun_year, c, slope, intercept)
cols = ['immun_BCG', 'immun_DPT', 'immun_HepB3', 'immun_Hib3', 'immun_Pol3', 'immun_Measles', 'LEx']

wb_immun[wb_immun.year > 1980].groupby('year').median()[cols].plot(figsize=(10,6), title='Level of different immunizations over time')



#years = wb_immun.year.unique()

years = list(range(1980, 2017))



r_2 = dict()



for c in cols[:-1]:

    val = list()

    for y in years:

        reg_data = wb_immun[wb_immun.year == y].dropna()

        try:

            X_train, X_test, Y_train, Y_test = train_test_split(reg_data[c].values.reshape(-1,1), reg_data['LEx'].values.reshape(-1,1), test_size=0.3, random_state=3)

            lm = LinearRegression()

            lm.fit(X_train,Y_train)

            val.append(lm.score(X_train,Y_train))

        except:

            val.append(np.nan)

    r_2[c] = val

pd.DataFrame(r_2, index=years).plot(figsize=(10,6), title='R^2 of different immunisation over time')
world_bank_df = pd.read_csv(join(csvFolder, "WorldBank_Data.csv"), sep=";", decimal=",", encoding='latin-1')

world_bank_df.rename(columns={'Country Name': 'Country', 'Country Code': 'ISO3'}, inplace=True)

world_bank_df.reset_index(drop=True, inplace=True)

world_bank_latest = world_bank_df.loc[:, ["ISO3", "Sex", "2017"]]

world_bank_latest.dropna(inplace=True)

world_bank_latest.rename(columns={"2017": "Life expectancy"}, inplace=True)

Country_Codes = pd.read_csv(join(csvFolder, "Country_Codes.csv"), sep=";", decimal=".")

Country_Codes.drop(["ISONumeric"], axis=1, inplace=True)

lex = pd.merge(world_bank_latest, Country_Codes, on="ISO3")
svg = parse("../input/data/Universal_Healthcare_by_Country_20191229.svg").getroot()

ns = "{http://www.w3.org/2000/svg}"

gs = svg.findall(f".//{ns}g[@id]")

rgb2healthcare = {"rgb(51,160,44)": "Free and universal health care",

                  "rgb(80,255,80)": "Universal health care",

                  "rgb(43,140,190)": "Free but not universal health care",

                  "rgb(228,26,28)": "No free or universal health care",

                  #"rgb(224,224,224)": "Unknown"

                 }

healthcare_df = {"ISO2": [], "Coverage": []}

for g in gs:

    country_id = g.get("id").upper()

    for child in g:

        fill = child.get("fill")

        if fill in rgb2healthcare:

            if country_id not in healthcare_df["ISO2"]:

                healthcare_df["ISO2"].append(country_id)

                healthcare_df["Coverage"].append(rgb2healthcare[fill])

healthcare_df = pd.DataFrame(healthcare_df)
healthcare_df.head()
lex_healthcare = pd.merge(lex, healthcare_df, on="ISO2")
len(lex_healthcare)
lex_healthcare.Coverage.value_counts()
lex_healthcare[lex_healthcare.Coverage == "Universal health care"]
lex_healthcare.groupby("Coverage")["Life expectancy"].hist(bins=30)

plt.title("Distribution of life expectancy by health care system")

plt.show()
lex_healthcare.groupby("Coverage")["Life expectancy"].median().plot(kind="barh")

plt.title("Median life expectancy per health care system")

plt.show()
kruskalwallis(lex_healthcare[lex_healthcare["Coverage"] == "No free or universal health care"]["Life expectancy"].values,

              lex_healthcare[lex_healthcare["Coverage"] == "Free and universal health care"]["Life expectancy"].values)
lex_healthcare[(lex_healthcare["Coverage"] == "No free or universal health care") & (lex_healthcare["Life expectancy"] > 80)]
lex_healthcare[(lex_healthcare["Coverage"] == "Free and universal health care") & (lex_healthcare["Life expectancy"] < 65)]
gini_df = pd.read_csv(join(csvFolder, "WorldBank_Gini.csv"))

gini_df.rename(columns={"Country Code": "ISO3"}, inplace=True)
gini_df.head()
gini_df["Gini"] = gini_df.loc[:, "2007":"2017"].mean(axis=1)
gini_df["Gini"].describe()
gini_df["Gini"].plot(kind="hist")

plt.title("Distribution of the Gini index")

plt.show()
gini_avg = gini_df.loc[:, ["Country Name", "ISO3", "Gini"]]

gini_avg.dropna(inplace=True)

lex_gini = pd.merge(world_bank_latest, gini_avg, on="ISO3")
lex_gini.head()
gini_fig = px.choropleth(lex_gini,

                         locations="ISO3",

                         locationmode="ISO-3",

                         color="Gini",

                         hover_name="Country Name"

                        )

gini_fig.show()
sns.regplot("Gini", "Life expectancy", data=lex_gini)

plt.title("Life expectancy by mean Gini index (2007-2017)")

plt.show()
pearsonr(lex_gini["Gini"], lex_gini["Life expectancy"])
lex_gini[(lex_gini["Gini"] < 35) & (lex_gini["Life expectancy"] < 65)]
lex_gini[(lex_gini["Gini"] > 50) & (lex_gini["Life expectancy"] > 80)]
freedom_countries = pd.read_excel("../input/data/government/Country_and_Territory_Ratings_and_Statuses_FIW1973-2019.xls", sheet_name=1, skiprows=2, usecols=[0, 135], na_values=["-"])

freedom_territories = pd.read_excel("../input/data/government/Country_and_Territory_Ratings_and_Statuses_FIW1973-2019.xls", sheet_name=2, skiprows=2, usecols=[0, 135], na_values=["-"])

freedom = freedom_countries.append(freedom_territories)

freedom.dropna(inplace=True)

freedom.rename(columns={"Status.44": "Freedom status"}, inplace=True)

country_name_map = {"Congo (Brazzaville)": "Republic of the Congo",

                    "Congo (Kinshasa)": "Democratic Republic of the Congo",

                    "Timor-Leste": "East Timor",

                    "The Gambia": "Gambia",

                    "Cote d'Ivoire": "Ivory Coast",

                    "St. Lucia": "Saint Lucia",

                    "St. Vincent and the Grenadines": "Saint Vincent and the Grenadines",

                    "Eswatini": "Swaziland",

                    "Macao": "Macau",

                    "Palestinian Authority-Administered Territories": "Palestine",

                    "Israeli-Occupied Territories": "West Bank"}

freedom.replace(country_name_map, inplace=True)
freedom.head()
freedom["Freedom status"].value_counts()
lex_freedom = pd.merge(lex, freedom, left_on="Country", right_on="Unnamed: 0")
lex_freedom.head()
lex_freedom.groupby("Freedom status")["Life expectancy"].hist(bins=30)

plt.title("Life expectancy distribution by freedom status")

plt.show()
lex_freedom.groupby("Freedom status")["Life expectancy"].median().plot(kind="bar")

plt.title("Median Life expectancy by freedom status")

plt.show()
kruskalwallis(lex_freedom[lex_freedom["Freedom status"] == "F"]["Life expectancy"].values,

              lex_freedom[lex_freedom["Freedom status"] == "NF"]["Life expectancy"].values)
kruskalwallis(lex_freedom[lex_freedom["Freedom status"] == "PF"]["Life expectancy"].values,

              lex_freedom[lex_freedom["Freedom status"] == "NF"]["Life expectancy"].values)
lex_freedom[(lex_freedom["Freedom status"] == "NF") & (lex_freedom["Life expectancy"] > 80)]
lex_freedom[(lex_freedom["Freedom status"] == "PF") & (lex_freedom["Life expectancy"] > 80)]
lex_freedom[(lex_freedom["Freedom status"] == "F") & (lex_freedom["Life expectancy"] < 65)]
spending = pd.read_csv("../input/data/medical-research-expenditure/unesco_gerd_by_field.csv")

spending = spending[spending["Indicator"] == "GERD - Medical and health sciences (in '000 current PPP$)"]

spending = spending.loc[:, ["LOCATION", "TIME", "Value"]]

spending.dropna(inplace=True)

spending.rename(columns={"Value": "R&D spending absolute"}, inplace=True)
spending.head()
gdp = pd.read_csv("../input/data/medical-research-expenditure/unesco_gdp.csv")

gdp = gdp[gdp["Indicator"] == "GDP, PPP (current international $)"]

gdp = gdp.loc[:, ["LOCATION", "TIME", "Value"]]

gdp.rename(columns={"Value": "GDP"}, inplace=True)

gdp["GDP"] = gdp["GDP"] / 1000
gdp.head()
med_spend = pd.merge(spending, gdp, on=["TIME", "LOCATION"])

med_spend["R&D spending in % of GDP"] = med_spend["R&D spending absolute"] / med_spend["GDP"] * 100

med_spend = med_spend.groupby("LOCATION").mean()
med_spend.head()
med_spend["R&D spending absolute"].plot(kind="hist", bins=30)

plt.title("Distribution of R&D spending in absolute numbers")

plt.show()
med_spend["R&D spending in % of GDP"].plot(kind="hist", bins=30)

plt.title("Distribution of R&D spending as a share of GDP")

plt.show()
med_spend[med_spend["R&D spending absolute"] > 5000000]
med_spend[med_spend["R&D spending in % of GDP"] > 0.3]
lex_med_spend = pd.merge(lex, med_spend, left_on="ISO3", right_on="LOCATION")

lex_med_spend.head()
sns.regplot("R&D spending in % of GDP", "Life expectancy", data=lex_med_spend)

plt.title("Life expectancy by mean medical R&D spend (2012-2017)")

plt.show()
pearsonr(lex_med_spend["R&D spending in % of GDP"], lex_med_spend["Life expectancy"])
lex_med_spend[(lex_med_spend["R&D spending in % of GDP"] < 0.03) & (lex_med_spend["Life expectancy"] > 85)]
# read indicators dataset

un_wpp = pd.read_csv("../input/data/un_health_population_prospects/WPP2019_Period_Indicators_Medium.csv")



# read population dataset, select subset of interest

un_wpp_population = pd.read_csv("../input/data/un_health_population_prospects/WPP2019_TotalPopulationBySex.csv")

un_wpp_population = un_wpp_population.loc[un_wpp_population.Variant == "Medium"]

un_wpp_population = un_wpp_population.loc[un_wpp_population.Time.isin(un_wpp.MidPeriod.unique())].drop("MidPeriod", axis=1).rename(columns={"Time": "MidPeriod"})



# merge indicators dataset and population dataset

un_wpp = pd.merge(left=un_wpp,right=un_wpp_population, how='left', on=["LocID", "Location", "VarID", "MidPeriod"])



# read location metadata excel file, preprocess as necessary

un_wpp_metadata = pd.read_excel("../input/data/un_health_population_prospects/WPP2019_F01_LOCATIONS.XLSX", skiprows=15, header=[0, 1]) 

un_wpp_metadata.columns = [' '.join(col).strip() if not col[0].startswith("Unnamed:") else col[1] for col in un_wpp_metadata.columns.values]

un_wpp_metadata.set_index("Index", inplace=True)



# merge Geographic region (e.g. Africa, Asia, Europe...) and Location Type Name from metadata to un_wpp

un_wpp = pd.merge(left=un_wpp, 

                  right=un_wpp_metadata.loc[:, ["Location code", "Geographic region Name", "Location Type Name", "ISO3 Alpha-code"]],

                  how='left', left_on=["LocID"], right_on=["Location code"])

un_wpp.drop("Location code", axis=1, inplace=True)

un_wpp.rename(columns={"Variant_x": "Variant", "Geographic region Name": "Continent"}, inplace=True)



un_wpp_wrapper = DatasetWrapper(un_wpp, "Location")
# dataset statistics

print("Years: ", un_wpp.MidPeriod.unique())

print("Population by 2100:", int(un_wpp_wrapper.get_col_instances("MidPeriod", 2098, "World").PopTotal.values))

print("Average human life expectancy by 2100:", int(un_wpp_wrapper.get_col_instances("MidPeriod", 2098, "World").LEx.values))
# plot life expectancy developement for specified countries

countries = ["Austria", "Germany", "United States of America", "South Africa", "Japan", "China", "India"]

un_wpp_countries = un_wpp_wrapper.get_countries(countries).pivot("Location", "MidPeriod", "LEx").T

un_wpp_countries.plot(grid=True, title="Life Expectancy 1960-2100")

plt.axvline(x=2020, linestyle='--', c="gray", linewidth=2.5, alpha=0.5)
# animate life expectancy developement for specified countries

countries = ["Austria", "Germany", "United States of America", "South Africa", "Japan", "China", "India"]

un_wpp_countries = un_wpp_wrapper.get_countries(countries)

px.scatter(un_wpp_countries,

           x="PopTotal", 

           y="LEx",

           animation_frame="MidPeriod",

           animation_group="Location",

           color="Location",

           size="PopTotal",

           hover_name="Location",  

           size_max=70,

           range_x=[un_wpp_countries.PopTotal.min() - 200000,un_wpp_countries.PopTotal.max() + 200000],

           range_y=[25,100],

           height=500)
countries_income = [

    'High-income countries', 

    'Low-income countries', 

    'Lower-middle-income countries',

    'Middle-income countries',

    'Upper-middle-income countries'

]

un_wpp_income = un_wpp_wrapper.get_countries(countries_income)

px.scatter(un_wpp_income, 

           x="PopTotal",

           y="LEx", animation_frame="MidPeriod", 

           size="PopTotal", 

           size_max=70,

           animation_group="Location",

           color="Location", 

           hover_name="Location",  

           range_x=[un_wpp_income.PopTotal.min() - 200000, un_wpp_income.PopTotal.max() + 200000], 

           range_y=[25,100],

           height=500)
# plot life expectancy developement for specified regions

continents = ["Europe", "Asia", "Africa", "Oceania", "Latin America and the Caribbean", "Northern America"]

un_wpp_continents = un_wpp_wrapper.get_countries(continents)

un_wpp_continents = un_wpp_continents[un_wpp_continents["Location Type Name"] == "Geographic region"]

un_wpp_continents = un_wpp_continents.pivot("Location", "MidPeriod", "LEx").T

un_wpp_continents.plot(title="Life Expectancy 1960-2100")

plt.axvline(x=2020, linestyle='--', c="gray", linewidth=2.5, alpha=0.5)
# animate life expectancy developement for specified countries

continents = ["Europe", "Asia", "Africa", "Oceania", "Latin America and the Caribbean", "Northern America"]

un_wpp_continents = un_wpp_wrapper.get_countries(continents)

un_wpp_continents = un_wpp_continents[un_wpp_continents["Location Type Name"] == "Geographic region"]

px.scatter(un_wpp_continents,

           x="PopTotal", 

           y="LEx",

           animation_frame="MidPeriod",

           animation_group="Location",

           color="Location",

           size="PopTotal",

           hover_name="Location",  

           size_max=70,

           range_x=[un_wpp_continents.PopTotal.min() - 200000,un_wpp_continents.PopTotal.max() + 200000],

           range_y=[25,100],

           height=500)
# life expectancy change from 2020-2100

absolute_change = un_wpp_continents[un_wpp_continents.MidPeriod == 2098].LEx.values - un_wpp_continents[un_wpp_continents.MidPeriod == 2023].LEx.values

percentual_change = absolute_change / un_wpp_continents[un_wpp_continents.MidPeriod == 2023].LEx.values * 100

un_wpp_continents_change = pd.DataFrame({"Percentual change 2020-2100": percentual_change, "Absolute change 2020-2100 in years": absolute_change},

                                        index=un_wpp_continents[un_wpp_continents.MidPeriod == 2098].Location.values)

display(un_wpp_continents_change)

ax = un_wpp_continents_change.plot(kind="bar",rot=0, title="Change in life expectancy 2020-2100")

ax.set_ylabel("Change")
# choropleth map of life expectancy in all countries

un_wpp_all_countries = un_wpp[un_wpp["Location Type Name"] == "Country/Area"]



fig = px.choropleth(un_wpp_all_countries, 

                    locations="ISO3 Alpha-code",

                    color="LEx", 

                    hover_name="Location",

                    animation_frame="MidPeriod",

                    width=1000, 

                    height=500, 

                    color_continuous_scale=px.colors.sequential.thermal,

                    range_color=[un_wpp_all_countries.LEx.min(), un_wpp_all_countries.LEx.max()])

fig.show()
# animate life expectancy for all countries

un_wpp_all_countries = un_wpp[un_wpp["Location Type Name"] == "Country/Area"]



px.scatter(un_wpp_all_countries,

           x="MidPeriod",

           y="LEx",

           facet_col="Continent",

           color="Continent",

           hover_name="Location",

           size_max=45, 

           range_x=[1940,2110], 

           range_y=[25,100],

)
# animate life expectancy developement for specified countries

un_wpp_all_countries = un_wpp[un_wpp["Location Type Name"] == "Country/Area"]



px.scatter(un_wpp_all_countries,

           x="PopTotal", 

           y="LEx",

           animation_frame="MidPeriod",

           color="Continent",

           size="PopTotal",

           hover_name="Location",  

           size_max=70,

           height=500, 

           range_x=[un_wpp_all_countries.PopTotal.min() - 100000,un_wpp_all_countries.PopTotal.max() + 100000],

           range_y=[25,100], )
countries = ["Europe", "Asia", "Africa", "Oceania", "Latin America and the Caribbean", "Northern America"]

un_wpp_pop = un_wpp_wrapper.get_col_instances("Location Type Name", "Geographic region", countries).pivot("Location", "MidPeriod", "PopTotal")

un_wpp_pop = un_wpp_pop / un_wpp_pop.sum() * 100

ax = un_wpp_pop.T.plot(title="Perentage of World population by continent")

ax.set_ylabel("Percentage")

plt.axvline(x=2020, linestyle='--', c="gray", linewidth=2.5, alpha=0.5)
display(un_wpp_pop[2098].to_frame(), un_wpp_pop[2023].to_frame())
un_wpp_other = pd.read_csv("../input/data/un_health_population_prospects/WPP2019_Period_Indicators_OtherVariants.csv")

un_wpp_other_wrapper = DatasetWrapper(un_wpp_other, "Location")
un_wpp_other_world = un_wpp_other_wrapper.get_col_instances("Variant", ['Upper 80 PI', 'Lower 80 PI', 'Upper 95 PI', 'Lower 95 PI'], "World")

un_wpp_world= un_wpp_wrapper.get_countries("World")

un_wpp_combined = un_wpp_world.append(un_wpp_other_world, ignore_index=True, sort=False).pivot("Variant", "MidPeriod", "LEx").T
un_wpp_combined.plot(c="black", figsize=(14, 10), legend=False, title="Probabilisitc Projctions Life Expectancy 1950-2100")

plt.axvline(x=2023, linestyle='--', c="gray", linewidth=2.5, alpha=0.5)

plt.fill_between(un_wpp_combined.index, un_wpp_combined.Medium, un_wpp_combined["Upper 80 PI"], facecolor='blue', alpha=0.2)

plt.fill_between(un_wpp_combined.index, un_wpp_combined.Medium, un_wpp_combined["Lower 80 PI"], facecolor='blue', alpha=0.2)

plt.fill_between(un_wpp_combined.index, un_wpp_combined["Lower 80 PI"], un_wpp_combined["Lower 95 PI"], facecolor='green', alpha=0.2)

plt.fill_between(un_wpp_combined.index, un_wpp_combined["Upper 80 PI"], un_wpp_combined["Upper 95 PI"], facecolor='green', alpha=0.2)
un_wpp_other_austria = un_wpp_other_wrapper.get_col_instances("Variant", ['Upper 80 PI', 'Lower 80 PI', 'Upper 95 PI', 'Lower 95 PI'], "Austria")

un_wpp_austria = un_wpp_wrapper.get_countries("Austria")

un_wpp_combined = un_wpp_austria.append(un_wpp_other_austria, ignore_index=True, sort=False).pivot("Variant", "MidPeriod", "LEx").T



un_wpp_combined.plot(c="black", figsize=(14, 10), legend=False, title="Probabilisitc Projctions Life Expectancy 1950-2100 in Austria")

plt.axvline(x=2023, linestyle='--', c="gray", linewidth=2.5, alpha=0.5)

plt.fill_between(un_wpp_combined.index, un_wpp_combined.Medium, un_wpp_combined["Upper 80 PI"], facecolor='blue', alpha=0.2)

plt.fill_between(un_wpp_combined.index, un_wpp_combined.Medium, un_wpp_combined["Lower 80 PI"], facecolor='blue', alpha=0.2)

plt.fill_between(un_wpp_combined.index, un_wpp_combined["Lower 80 PI"], un_wpp_combined["Lower 95 PI"], facecolor='green', alpha=0.2)

plt.fill_between(un_wpp_combined.index, un_wpp_combined["Upper 80 PI"], un_wpp_combined["Upper 95 PI"], facecolor='green', alpha=0.2)
un_wpp_age = pd.read_excel("../input/data/un_health_population_prospects/WPP2019_SA1_POP_F09_1_PERCENTAGE_OF_TOTAL_POPULATION_BY_BROAD_AGE_GROUP_BOTH_SEXES.XLSX", 

                           sheet_name=0, skiprows=15, header=[0, 1])
# the excel sheet is multiheaded, which requires some preprocessing

un_wpp_age.columns = [' '.join(col).strip()

                      if not (col[0].startswith("Unnamed:") or col[0].startswith("Percentage of total") or col[0].startswith("Economic"))

                      else col[1] 

                      for col in un_wpp_age.columns.values]

un_wpp_age.set_index("Index", inplace=True)



# get only countries

un_wpp_age = un_wpp_age[un_wpp_age.Type == "Country/Area"]



# rename overlong columns resulting from excel sheet

un_wpp_age.rename(columns={"Type of aggregate, group, and constituents *": "Country", "Reference date (as of 1 July)": "Year"}, inplace=True)



# drop redundant columns

un_wpp_age.drop(["Variant", "Parent code", "Notes", "Total", "Type"], axis=1, inplace=True)



# merge metadata to obtain continents

un_wpp_age = pd.merge(left=un_wpp_age, 

                      right=un_wpp_metadata.loc[:, ["Location code", "Geographic region Name", "ISO3 Alpha-code"]],

                      how='left', left_on=["Country code"], right_on=["Location code"])



# postprocessing after merge - drop redundant columns, clearn col names

un_wpp_age.drop("Location code", axis=1, inplace=True)

un_wpp_age.rename(columns={"Geographic region Name": "Continent"}, inplace=True)



# wrap dataset for fast querying

un_wpp_age_wrapper = DatasetWrapper(un_wpp_age, "Country")
print(un_wpp_age.shape)

un_wpp_age.head()
# average by continent and year

un_wpp_age_continent = un_wpp_age.drop("Country code", axis=1)

un_wpp_age_continent.loc[:, "0-1": "90+"] = un_wpp_age_continent.loc[:, "0-1": "90+"].apply(pd.to_numeric)

un_wpp_age_continent = un_wpp_age_continent.groupby(["Continent", "Year"]).mean().unstack()



# average to get the values for whole World

world_age = un_wpp_age_continent.mean()

# name Series to enable appending and append

world_age.name = "World"

un_wpp_age_continent = un_wpp_age_continent.append(world_age)
# "Proportion of population of age 65+"

ax = un_wpp_age_continent["65+"].T.plot(title="Proportion of population of age 65+")

ax.set_ylabel("Percent")

plt.axvline(x=2020, linestyle='--', c="gray", linewidth=2.5, alpha=0.5)
# "Proportion of population of age 80+"

ax = un_wpp_age_continent["80+"].T.plot(title="Proportion of population of age 80+")

ax.set_ylabel("Percent")

plt.axvline(x=2020, linestyle='--', c="gray", linewidth=2.5, alpha=0.5)
# "Proportion of population of age 90+"

ax = un_wpp_age_continent["90+"].T.plot(title="Proportion of population of age 90+")

ax.set_ylabel("Percent")

plt.axvline(x=2020, linestyle='--', c="gray", linewidth=2.5, alpha=0.5)