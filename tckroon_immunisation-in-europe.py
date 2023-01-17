import numpy as np 

import pandas as pd

import plotly.express as px

pd.plotting.register_matplotlib_converters()

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns
# Reading global health care csv

# Quickview of columns names and the rows

dfx = pd.read_csv('/kaggle/input/uncover/UNCOVER/oecd/health-care-utilization.csv')

dfx.head()



# Explaining the most import columns



# 'variable': the variable we are looking at, in this case immunisation op hepatitis-B

# 'measure': explanation of what we are measuring, in this case the percentage of children immunised

# 'value': the actual percentage value of children immunised

# 'country': the country name

# 'cou': the country code

# 'year': year of measurement

# 'flag': important additional information, like a difference in methodology or estimated value

# 'flag_codes': Abbreviation code which represents a specific flag, 'Estimated value' has the flag_code 'E'
# Mapping the full size of the dataframe (rows and columns)

# The dataframe exists out of 158.095 rows and 11 columns

dfx.shape
# Zooming in into all the different types of variables in the dataframe

dfx.variable.unique()
# Locating only the rows with 'Immunisation'as a variable

dfx = dfx.loc[dfx.variable.isin(['Immunisation: Hepatitis B', 

                                   'Immunisation: Influenza',

                                   'Immunisation: Diphtheria, Tetanus, Pertussis',

                                   'Immunisation: Measles'])]
# Mapping the full size of the new dataframe with only 'immunisation' variables (rows and columns)

# The new dataframe has 1329 rows and 11 columns

dfx.shape
# Checking the dataframe for missing entries

# As it seems there are not as many non-null entries as rows for the columns 'flag_codes' and 'flags', but that is to be expected as they provide additonal information

# Further no missing entries

dfx.info()
# Quick view of all the countries with Immunisation data

dfx.country.unique()
# Quick view of what years the Immunisation data represents

dfx.year.unique()
# Mapping how the amount of rows is distributed over years

# 2018 seems to be incomplete in compare to other years

dfx.groupby('year').year.count()
dfx.head()
# removing year 2018 as the data of this year is incomplete

dfx = dfx[dfx.year != 2018]
# Renaming columns 'cou' to 'country_code' and 'value' to 'immunisation'

dfx = dfx.rename(columns={'cou':'country_code', 'value': 'immunisation'})
# Importing 'continent' column to dataframe by merging with a second dataframe

import plotly.express as px

dfy = px.data.gapminder()

df = pd.merge(dfx, dfy, how = 'left', left_on='country_code', right_on='iso_alpha', suffixes=('_x', '_y'))

df = df[['var', 'variable', 'unit', 'measure', 'country_x', 'country_code', 'continent', 'year_x', 'immunisation', 'flag_codes', 'flags']]

df = df.drop_duplicates()

df = df.rename(columns={'country_x': 'country', 'year_x': 'year'})
# Checking for missing values after as result of mergin

# There seems to be some missing values in the 'continent' column.

df.info()
# Checking for which countries values are missing in the continent column

dfc = df[pd.isnull(df.continent)]

dfc.country.unique()
# Manually adding continents to the missing countries, as well as adjusting Turkeys continent to Asia

df.loc[df['country'] == 'Luxembourg', 'continent'] = 'Europe'

df.loc[df['country'] == 'Estonia', 'continent'] = 'Europe'

df.loc[df['country'] == 'Russia', 'continent'] = 'Asia'

df.loc[df['country'] == 'Latvia', 'continent'] = 'Europe'

df.loc[df['country'] == 'Lithuania', 'continent'] = 'Europe'

df.loc[df['country'] == 'Turkey', 'continent'] = 'Asia'
# Filtering the countries where continent is Europe

df= df.loc[df.continent == 'Europe']
df.head()
# Number of European countries avaiable for analysis

print("No. of Countries available for analysis :", df['country'].nunique())#
# # Number of European countries with specific immunisation data

print("No. of Countries available for analysis (Measles):", df.country[(df.variable == 'Immunisation: Measles')].nunique())

print("No. of Countries available for analysis (Diphtheria, Tetanus, Pertussis):", df.country[(df.variable == 'Immunisation: Diphtheria, Tetanus, Pertussis')].nunique())

print("No. of Countries available for analysis (Hepatitis-B):", df.country[(df.variable == 'Immunisation: Hepatitis B')].nunique())

print("No. of Countries available for analysis (Influenza):", df.country[(df.variable == 'Immunisation: Influenza')].nunique())
# European countries with immunisation data

plt.rcParams['figure.figsize'] = (18, 8)

plt.style.use('fivethirtyeight')



dfq = px.data.gapminder()

fig = px.choropleth(df,

                    locations="country_code", 

                    hover_name="country", 

                    )

fig.show()
# European countries with unknown immunisation data

plt.rcParams['figure.figsize'] = (18, 8)

plt.style.use('fivethirtyeight')



dfz = pd.read_csv('/kaggle/input/country-mapping-iso-continent-region/continents2.csv')

dfz = dfz.loc[dfz.region == 'Europe']

dfz = dfz.rename(columns={'name': 'country', 'alpha-3': 'country_code'}) 

dfz = dfz.loc[dfz.country != 'Russia']

common = dfz.merge(df, on=['country_code'])

dfg = dfz[(~dfz.country_code.isin(common.country_code))]



dfq = px.data.gapminder()

fig = px.choropleth(dfg,

                    locations="country_code",

                    hover_name="country")



fig.show()
# European population

dfd = pd.read_csv('/kaggle/input/population-by-country-2020/population_by_country_2020.csv')

dfd = dfd[['Country (or dependency)', 'Population (2020)']]

dfd = dfd.rename(columns={'Country (or dependency)': 'country', 'Population (2020)': 'population'})

dfd['country'].replace({'Czech Republic (Czechia)': 'Czech Republic', 'Bosnia and Herzegovina': 'Bosnia And Herzegovina', 'North Macedonia': 'Macedonia'}, inplace=True)

dfp = pd.merge(dfz, dfd, how = 'left', on='country', suffixes=('', '_r'))

dfk = dfp.population.sum()



# Population of European countries with immunisation data

dfl = pd.merge(dfp, df , how='inner', on='country_code', suffixes=('', '_r'))

dfe = dfl[['country', 'population']].groupby('country').population.agg('mean')

dfe = dfe.sum()



# Immunisation data represents part of european population

part_of_population = ((dfe / dfk) *100).round(1)

dfe_millions = (dfe / 1000000).round(1)

print("The countries with immunisation data cover", dfe_millions, "million European citizens, which represents", part_of_population, "percent of the European population")

# Europeans biggest countries (with immunisation data) according to population 

plt.style.use('seaborn-dark')

plt.figure(figsize=(25, 9))



dfl = dfl[['country', 'year', 'population']]

dfl = dfl.loc[dfl.groupby('country').year.idxmax()]

xyc = dfl[['country', 'population']].set_index('country')

xyc = xyc.sort_values(by = 'population', ascending = False)

xyc = (xyc / 1000000).round(1)



color = plt.cm.winter(np.linspace(0, 10, 100))

sns.barplot(x=xyc.index, y=xyc['population'], palette = 'winter')

plt.title("European Countries with Immunisation data, sorted by Population", fontsize = 30)

plt.xlabel("Name of Country")

plt.xticks(rotation = 90)

plt.ylabel("Population (in millions)")

plt.show()
# Creating a new dataframe with only Measles Immunisation data

europe_mea = df.loc[df.variable == 'Immunisation: Measles']



# Most recent Measles immunisation data per country in Europe

europe_mea_recent = europe_mea.loc[europe_mea.groupby('country').year.idxmax()]
# Top 5 countries with the highest immunisation rate for Measles

europe_mea_recent[['variable', 'measure', 'country', 'year', 'immunisation']].set_index('country').sort_values(by='immunisation', ascending=False).head(5).style.background_gradient(cmap = 'Wistia', subset= 'immunisation')
# Top 5 countries with the lowest immunisation rate for Measles

europe_mea_recent[['variable', 'measure', 'country', 'year', 'immunisation']].set_index('country').sort_values(by='immunisation', ascending=True).head(5).style.background_gradient(cmap = 'Wistia', subset= 'immunisation')
# Measles immunisation trend per country in Europe

plt.rcParams['figure.figsize'] = (18, 8)

plt.style.use('fivethirtyeight')



dfq = px.data.gapminder()

fig = px.choropleth(europe_mea,

                    locations="country_code", 

                    color="immunisation", 

                    hover_name="country",

                    animation_frame="year", 

                    range_color=[86,100],

                    )

fig.show()

# European Measles Immunisation Trend per year

europe_mea_mean = europe_mea[['year','immunisation']].groupby('year').immunisation.agg(['mean', 'min', 'max']).round(1).rename(columns={'mean': 'european_measles_mean'})

europe_mea_mean
# Measles Immunisation growth per country (difference between start_year and end_year)

europe_mea_year_max = europe_mea.loc[europe_mea.groupby('country').year.idxmax()]

europe_mea_year_min = europe_mea.loc[europe_mea.groupby('country').year.idxmin()]

europe_mea_value_difference = pd.merge(europe_mea_year_min, europe_mea_year_max, how = 'left', on='country_code', suffixes=('', '_r'))

europe_mea_value_difference = europe_mea_value_difference[['var', 'variable', 'unit', 'measure', 'country', 'country_code', 'continent', 'year', 'immunisation', 'flag_codes', 'flags', 'year_r', 'immunisation_r', 'flag_codes_r', 'flags_r']]

europe_mea_value_difference = europe_mea_value_difference.rename(columns={'year': 'start_year', 'immunisation': 'start_immunisation', 'year_r': 'end_year', 'immunisation_r': 'end_immunisation'})

europe_mea_value_difference['immunisation_growth'] = europe_mea_value_difference['end_immunisation'] - europe_mea_value_difference['start_immunisation']

europe_mea_value_difference = europe_mea_value_difference[['variable', 'measure', 'country', 'immunisation_growth']]

# Top 5 European countries with highest Measles Immunisation gain (between 2010 and 2017)

europe_mea_value_difference.set_index('country').sort_values(by='immunisation_growth', ascending=False).head(5).style.background_gradient(cmap = 'Wistia')
# Top 5 European countries with highest Measles Immunisation loss (between 2010 and 2017)

europe_mea_value_difference.set_index('country').sort_values(by='immunisation_growth', ascending=True).head(5).style.background_gradient(cmap = 'Wistia')
# Europes most populated countries (with immunisation data)

europe6_mea = europe_mea.loc[europe_mea.country.isin(['Germany', 'France', 'United Kingdom', 'Italy', 'Spain', 'Poland'])]



# Big-6 Measles Immunisation Trend per year

europe6_mea_mean = europe6_mea[['year','immunisation']].groupby('year').immunisation.agg(['mean', 'min', 'max']).round(1).rename(columns={'mean': 'big-6_measles_mean'})

europe6_mea_mean
# The Big-6 mean in comparison to the European mean

plt.style.use('seaborn-dark')

plt.figure(figsize=(20,8))



europe_compare_mea_mean = pd.merge(europe6_mea_mean, europe_mea_mean, how = 'left', on = 'year')

europe_compare_mea_mean1 = europe_compare_mea_mean[['big-6_measles_mean', 'european_measles_mean']]



sns.lineplot(data=europe_compare_mea_mean1)

plt.title('Big-6 mean VS European mean', fontsize = 20)

plt.xlabel('year')

plt.ylabel('Immunisation (in %)')

plt.show()
# Big-6 Most recent Measles Immunisation data

europe6_mea_recent = europe6_mea.loc[europe6_mea.groupby('country').year.idxmax()]

europe6_mea_recent[['variable', 'measure', 'country', 'immunisation']].set_index('country').sort_values(by='immunisation', ascending=False).head(6).style.background_gradient(cmap = 'Wistia')
# Big-6 Measless Immunisation growth (between 2010 and 2017)

europe6_mea_value_difference = europe_mea_value_difference.loc[europe_mea_value_difference.country.isin(['Germany', 'France', 'United Kingdom', 'Italy', 'Spain', 'Poland'])]

europe6_mea_value_difference = europe6_mea_value_difference[['variable', 'measure', 'country', 'immunisation_growth']]

europe6_mea_value_difference.set_index('country').sort_values(by='immunisation_growth', ascending=False).head(6).style.background_gradient(cmap = 'Wistia')
# Big-6 Measles Immunisation trend per country

plt.style.use('seaborn-dark')

plt.figure(figsize=(20,8))



europe6_mea1= europe6_mea[['country', 'year', 'immunisation']].set_index('year')

europe6_mea2 = europe6_mea1.pivot_table('immunisation', ['year'], 'country')



sns.lineplot(data=europe6_mea2)

plt.title('Big-6 Measles Immunisation Trend', fontsize = 20)

plt.xlabel('year')

plt.ylabel('Immunisation (in %)')

plt.show()
# Stability of the Big-6 Measles Immunisation rates (per country)

europe6_mea_std = europe6_mea.loc[europe6_mea.country.isin(['Germany', 'France', 'United Kingdom', 'Italy', 'Spain', 'Poland'])]

europe6_mea_std.groupby('country').immunisation.agg(['std']).round(1).rename(columns={'std': 'standard_deviation'}).sort_values(by='standard_deviation', ascending=True).head(6).style.background_gradient(cmap = 'Wistia')
# Creating a new dataframe with only DTP Immunisation data

europe_dtp = df.loc[df.variable == 'Immunisation: Diphtheria, Tetanus, Pertussis']



# Most recent DTP immunisation data per country in Europe

europe_dtp_recent = europe_dtp.loc[europe_dtp.groupby('country').year.idxmax()]
# Top 5 countries with the highest immunisation rate for DTP

europe_dtp_recent[['variable', 'measure', 'country', 'year', 'immunisation']].set_index('country').sort_values(by='immunisation', ascending=False).head(5).style.background_gradient(cmap = 'Wistia', subset= 'immunisation')
# Top 5 countries with the lowest immunisation rate for DTP

europe_dtp_recent[['variable', 'measure', 'country', 'year', 'immunisation']].set_index('country').sort_values(by='immunisation', ascending=True).head(5).style.background_gradient(cmap = 'Wistia', subset= 'immunisation')
# DTP Immunisation trend per country in Europe

plt.rcParams['figure.figsize'] = (18, 8)

plt.style.use('fivethirtyeight')



dfq = px.data.gapminder()

fig = px.choropleth(europe_dtp,

                    locations="country_code", 

                    color="immunisation", 

                    hover_name="country",

                    animation_frame="year", 

                    range_color=[86,100],

                    )

fig.show()

# European Measles Immunisation Trend per year

europe_dtp_mean = europe_dtp[['year','immunisation']].groupby('year').immunisation.agg(['mean', 'min', 'max']).round(1).rename(columns={'mean': 'european_dtp_mean'})

europe_dtp_mean
# DTP Immunisation growth per country (difference between start_year and end_year)

europe_dtp_year_max = europe_dtp.loc[europe_dtp.groupby('country').year.idxmax()]

europe_dtp_year_min = europe_dtp.loc[europe_dtp.groupby('country').year.idxmin()]

europe_dtp_value_difference = pd.merge(europe_dtp_year_min, europe_dtp_year_max, how = 'left', on='country_code', suffixes=('', '_r'))

europe_dtp_value_difference = europe_dtp_value_difference[['var', 'variable', 'unit', 'measure', 'country', 'country_code', 'continent', 'year', 'immunisation', 'flag_codes', 'flags', 'year_r', 'immunisation_r', 'flag_codes_r', 'flags_r']]

europe_dtp_value_difference = europe_dtp_value_difference.rename(columns={'year': 'start_year', 'immunisation': 'start_immunisation', 'year_r': 'end_year', 'immunisation_r': 'end_immunisation'})

europe_dtp_value_difference['immunisation_growth'] = europe_dtp_value_difference['end_immunisation'] - europe_dtp_value_difference['start_immunisation']

europe_dtp_value_difference = europe_dtp_value_difference[['variable', 'measure', 'country', 'immunisation_growth']]
# Top 5 European countries with highest DTP Immunisation gain (between 2010 and 2017)

europe_dtp_value_difference.set_index('country').sort_values(by='immunisation_growth', ascending=False).head(5).style.background_gradient(cmap = 'Wistia')
# Top 5 European countries with highest DTP Immunisation loss (between 2010 and 2017)

europe_dtp_value_difference.set_index('country').sort_values(by='immunisation_growth', ascending=True).head(10).style.background_gradient(cmap = 'Wistia')
# Europes most populated countries (with immunisation data)

europe6_dtp = europe_dtp.loc[europe_dtp.country.isin(['Germany', 'France', 'United Kingdom', 'Italy', 'Spain', 'Poland'])]



# Big-6 DTP Immunisation Trend per year

europe6_dtp_mean = europe6_dtp[['year','immunisation']].groupby('year').immunisation.agg(['mean', 'min', 'max']).round(1).rename(columns={'mean': 'big-6_dtp_mean'})

europe6_dtp_mean
# The Big-6 mean in comparison to the European mean

plt.style.use('seaborn-dark')

plt.figure(figsize=(20,8))



europe_compare_dtp_mean = pd.merge(europe6_dtp_mean, europe_dtp_mean, on = 'year', how = 'left')

europe_compare_dtp_mean1 = europe_compare_dtp_mean[['big-6_dtp_mean', 'european_dtp_mean']]



sns.lineplot(data=europe_compare_dtp_mean1)

plt.title('Big-6 mean VS European mean', fontsize = 20)

plt.xlabel('year')

plt.ylabel('Immunisation (in %)')

plt.show()
# Big-6 Most recent DTP Immunisation data

europe6_dtp_recent = europe6_dtp.loc[europe6_dtp.groupby('country').year.idxmax()]

europe6_dtp_recent[['variable', 'measure', 'country', 'immunisation']].set_index('country').sort_values(by='immunisation', ascending=False).head(6).style.background_gradient(cmap = 'Wistia')
# Big-6 DTP Immunisation growth (between 2010 and 2017)

europe6_dtp_value_difference = europe_dtp_value_difference.loc[europe_dtp_value_difference.country.isin(['Germany', 'France', 'United Kingdom', 'Italy', 'Spain', 'Poland'])]

europe6_dtp_value_difference = europe6_dtp_value_difference[['variable', 'measure', 'country', 'immunisation_growth']]

europe6_dtp_value_difference.set_index('country').sort_values(by='immunisation_growth', ascending=False).head(6).style.background_gradient(cmap = 'Wistia')
# Big-6 DTP Immunisation trend per country

plt.style.use('seaborn-dark')

plt.figure(figsize=(20,8))



europe6_dtp1= europe6_dtp[['country', 'year', 'immunisation']].set_index('year')

europe6_dtp2 = europe6_dtp1.pivot_table('immunisation', ['year'], 'country')



sns.lineplot(data=europe6_dtp2)

plt.title('Big-6 DTP Immunisation Trend', fontsize = 20)

plt.xlabel('year')

plt.ylabel('Immunisation (in %)')

plt.show()
# Stabilitiy of the Big-6 DTP Immunisation rates (per country)

europe6_dtp_std = europe6_dtp.loc[europe6_dtp.country.isin(['Germany', 'France', 'United Kingdom', 'Italy', 'Spain', 'Poland'])]

europe6_dtp_std.groupby('country').immunisation.agg(['std']).round(1).rename(columns={'std': 'standard_deviation'}).sort_values(by='standard_deviation', ascending=True).head(6).style.background_gradient(cmap = 'Wistia')

# Creating a new dataframe with only Hepatitis-B Immunisation data

europe_hep = df.loc[df.variable == 'Immunisation: Hepatitis B']



# Most recent Hepatitis-B immunisation data per country in Europe

europe_hep_recent = europe_hep.loc[europe_hep.groupby('country').year.idxmax()]
# Top 5 countries with the highest immunisation rate for Hepatitis-B

europe_hep_recent[['variable', 'measure', 'country', 'year', 'immunisation']].set_index('country').sort_values(by='immunisation', ascending=False).head(5).style.background_gradient(cmap = 'Wistia', subset= 'immunisation')
# Top 5 countries with the lowest immunisation rate for Hepatitis-B

europe_hep_recent[['variable', 'measure', 'country', 'year', 'immunisation']].set_index('country').sort_values(by='immunisation', ascending=True).head(5).style.background_gradient(cmap = 'Wistia', subset= 'immunisation')
# Hepatitis-B Immunisation trend per country in Europe

plt.rcParams['figure.figsize'] = (18, 8)

plt.style.use('fivethirtyeight')



dfq = px.data.gapminder()

fig = px.choropleth(europe_hep,

                    locations="country_code", 

                    color="immunisation", 

                    hover_name="country",

                    animation_frame="year", 

                    range_color=[86,100],

                    )

fig.show()
# European Hepatitis-B Immunisation Trend per year

europe_hep_mean = europe_hep[['year','immunisation']].groupby('year').immunisation.agg(['mean', 'min', 'max']).round(1).rename(columns={'mean': 'european_hepatitis_mean'})

europe_hep_mean
# Hepatitis-B Immunisation growth per country (difference between start_year and end_year)

europe_hep_year_max = europe_hep.loc[europe_hep.groupby('country').year.idxmax()]

europe_hep_year_min = europe_hep.loc[europe_hep.groupby('country').year.idxmin()]

europe_hep_value_difference = pd.merge(europe_hep_year_min, europe_hep_year_max, how = 'left', on='country_code', suffixes=('', '_r'))

europe_hep_value_difference = europe_hep_value_difference[['var', 'variable', 'unit', 'measure', 'country', 'country_code', 'continent', 'year', 'immunisation', 'flag_codes', 'flags', 'year_r', 'immunisation_r', 'flag_codes_r', 'flags_r']]

europe_hep_value_difference = europe_hep_value_difference.rename(columns={'year': 'start_year', 'immunisation': 'start_immunisation', 'year_r': 'end_year', 'immunisation_r': 'end_immunisation'})

europe_hep_value_difference['immunisation_growth'] = europe_hep_value_difference['end_immunisation'] - europe_hep_value_difference['start_immunisation']

europe_hep_value_difference = europe_hep_value_difference[['variable', 'measure', 'country', 'immunisation_growth']]
# Top 5 European countries with highest Hepatitis-B Immunisation gain (between 2010 and 2017)

europe_hep_value_difference.set_index('country').sort_values(by='immunisation_growth', ascending=False).head(5).style.background_gradient(cmap = 'Wistia')
# Top 5 European countries with highest Hepatitis-B Immunisation loss (between 2010 and 2017)

europe_hep_value_difference.set_index('country').sort_values(by='immunisation_growth', ascending=True).head(5).style.background_gradient(cmap = 'Wistia')
# Europes most populated countries (with immunisation data)

europe6_hep = europe_hep.loc[europe_hep.country.isin(['Germany', 'France', 'United Kingdom', 'Italy', 'Spain', 'Poland'])]



# Big-6 Hepatitis-B Immunisation Trend per year

europe6_hep_mean = europe6_hep[['year','immunisation']].groupby('year').immunisation.agg(['mean', 'min', 'max']).round(1).rename(columns={'mean': 'big-6_hepatitis_mean'})

europe6_hep_mean
# The Big-6 mean in comparison to the European mean

plt.style.use('seaborn-dark')

plt.figure(figsize=(20,8))



europe_compare_hep_mean = pd.merge(europe6_hep_mean, europe_hep_mean, on = 'year', how = 'left')

europe_compare_hep_mean1 = europe_compare_hep_mean[['big-6_hepatitis_mean', 'european_hepatitis_mean']]



sns.lineplot(data=europe_compare_hep_mean1)

plt.title('Big-6 mean VS European mean', fontsize = 20)

plt.xlabel('year')

plt.ylabel('Immunisation (in %)')

plt.show()
# Big-6 Most recent Hepatitis-B Immunisation data

europe6_hep_recent = europe6_hep.loc[europe6_hep.groupby('country').year.idxmax()]

europe6_hep_recent[['variable', 'measure', 'country', 'immunisation']].set_index('country').sort_values(by='immunisation', ascending=False).head(6).style.background_gradient(cmap = 'Wistia')
# Big-6 Hepatitis-B Immunisation growth (between 2010 and 2017)

europe6_hep_value_difference = europe_hep_value_difference.loc[europe_hep_value_difference.country.isin(['Germany', 'France', 'United Kingdom', 'Italy', 'Spain', 'Poland'])]

europe6_hep_value_difference = europe6_hep_value_difference[['variable', 'measure', 'country', 'immunisation_growth']]

europe6_hep_value_difference.set_index('country').sort_values(by='immunisation_growth', ascending=False).head(6).style.background_gradient(cmap = 'Wistia')
# Big-6 Hepatitis-b Immunisation trend per country

plt.style.use('seaborn-dark')

plt.figure(figsize=(20,8))



europe6_hep1= europe6_hep[['country', 'year', 'immunisation']].set_index('year')

europe6_hep2 = europe6_hep1.pivot_table('immunisation', ['year'], 'country')



sns.lineplot(data=europe6_hep2)

plt.title('Big-6 Hepatitis-B Immunisation Trend', fontsize = 20)

plt.xlabel('year')

plt.ylabel('Immunisation (in %)')

plt.show()
# Stability of the Big-6 Hepatitis-B Immunisation rates (per country)

europe6_hep_std = europe6_hep.loc[europe6_hep.country.isin(['Germany', 'France', 'United Kingdom', 'Italy', 'Spain', 'Poland'])]

europe6_hep_std.groupby('country').immunisation.agg(['std']).round(1).rename(columns={'std': 'standard_deviation'}).sort_values(by='standard_deviation', ascending=True).head(6).style.background_gradient(cmap = 'Wistia')

# Merging immunisation_mea, immunisation_dtp and immunisation_hep into one single dataframe

result_a = pd.merge(europe_mea, europe_dtp, how='outer', left_on=['country', 'country_code', 'year', 'continent', 'unit', 'measure'], right_on=['country', 'country_code', 'year', 'continent', 'unit', 'measure'], suffixes=('_l', '_r'))

result_b = pd.merge(result_a, europe_hep, how='outer', left_on=['country', 'country_code', 'year', 'continent', 'unit', 'measure'], right_on=['country', 'country_code', 'year', 'continent', 'unit', 'measure'], suffixes=('_l', '_r'))

result_b = result_b[['country', 'country_code', 'continent', 'unit', 'measure', 'year', 'variable_l', 'immunisation_l', 'flag_codes_l', 'variable_r', 'immunisation_r', 'flag_codes_r', 'variable', 'immunisation', 'flag_codes']]

result_b = result_b.rename(columns={'variable_l': 'variable_mea', 'variable_r': 'variable_dtp', 'variable': 'variable_hep', 'immunisation_l': 'immunisation_mea', 'immunisation_r': 'immunisation_dtp', 'immunisation': 'immunisation_hep', 'flag_codes_l': 'flag_codes_mea', 'flag_codes_r': 'flag_codes_dtp', 'flag_codes': 'flag_codes_hep'})

result_b = result_b.sort_values(by=['country', 'year'], ascending=True)

europe_overall = result_b.loc[result_b.continent == 'Europe']



# quickview of the europe_overall dataframe

europe_overall.head()

# Countries available for analysis

print("No. of European Countries available for analysis :", europe_overall.country.nunique())
# Countries available for each individual variable

print("No. of Countries available for analysis (Measles):", europe_overall.country[pd.notnull(europe_overall.immunisation_mea)].nunique())

print("No. of Countries available for analysis (Diphtheria, Tetanus, Pertussis):", europe_overall.country[pd.notnull(europe_overall.immunisation_dtp)].nunique())

print("No. of Countries available for analysis (Hepatitis-B):", europe_overall.country[pd.notnull(europe_overall.immunisation_hep)].nunique())

# Countries with missing Hepatitis-B immunisation data

europe_overall.country[pd.isnull(europe_overall.immunisation_hep)].unique()
# Amount of years with missing Hepatitis-B data per country

europe_overall[pd.isnull(europe_overall.immunisation_hep)].country.value_counts()
# Filling in all missing Hepatitis-B values with 0

values = {'variable_hep': 'Immunisation: Hepatitis B', 'immunisation_hep': 0}

europe_overall = europe_overall.fillna(value=values)
# Replacing the value of Slovenia 2010 with 92

# Slovenia only misses one year of data, using the same value as in 2011 because it represents a more accurate resemblence

europe_overall.loc[184, 'immunisation_hep'] = 92
# Adding a new column 'immunisation_overall' (the mean of all the three child immunisations)

europe_overall['immunisation_overall'] = europe_overall[['immunisation_mea', 'immunisation_dtp', 'immunisation_hep']].mean(axis=1).round(1)
# Checking for missing values

# All seems good

europe_overall.info()
# Quickview of the new dataframe including the new 'immunisation_overall' column

europe_overall.head()
# Most recent immunisation data per country in Europe

europe_overall_recent = europe_overall.loc[europe_overall.groupby('country').year.idxmax()]
# Overall top 10 european countries when it comes to all three in child immunisations

europe_overall_recent[['country', 'measure', 'immunisation_overall']].set_index('country').sort_values(by='immunisation_overall', ascending=False).head(10).style.background_gradient(cmap = 'Wistia')
# Overall worst 10 european countries when it comes to all three child immunisations

europe_overall_recent[['country', 'measure', 'immunisation_overall']].set_index('country').sort_values(by='immunisation_overall', ascending=True).head(10).style.background_gradient(cmap = 'Wistia')
# Overall Immunisation trend per country in Europe

plt.rcParams['figure.figsize'] = (18, 8)

plt.style.use('fivethirtyeight')



dfq = px.data.gapminder()

fig = px.choropleth(europe_overall,

                    locations="country_code", 

                    color="immunisation_overall", 

                    hover_name="country",

                    animation_frame="year", 

                    range_color=[86,100],

                    )



fig.show()
# European Overall Immunisation Trend per year

europe_overall_mean = europe_overall[['year', 'immunisation_overall']].groupby('year').immunisation_overall.agg(['mean', 'min', 'max']).round(1).rename(columns={'mean': 'european_overall_mean'})

europe_overall_mean
# European Immunisation per year (the contribution of the three childimmunisations)

europe_overall_mean1 = europe_overall[['year', 'immunisation_mea', 'immunisation_dtp', 'immunisation_hep', 'immunisation_overall']].groupby('year').agg('mean').round(1).rename(columns={'immunisation_mea': 'european_mea_mean', 'immunisation_dtp': 'european_dtp_mean', 'immunisation_hep': 'european_hep_mean', 'immunisation_overall': 'european_overall_mean'})

europe_overall_mean1.style.background_gradient(cmap="Wistia")
# Overall Immunisation growth per country (difference between start_year and end_year)

europe_overall_year_max = europe_overall.loc[europe_overall.groupby('country').year.idxmax()]

europe_overall_year_min = europe_overall.loc[europe_overall.groupby('country').year.idxmin()]

europe_overall_value_difference = pd.merge(europe_overall_year_min, europe_overall_year_max, on = 'country_code', how = 'outer', suffixes=('', '_r'))

europe_overall_value_difference = europe_overall_value_difference.rename(columns={'year': 'start_year', 'immunisation_overall': 'start_immunisation', 'year_r': 'end_year', 'immunisation_overall_r': 'end_immunisation'})

europe_overall_value_difference['immunisation_growth'] = europe_overall_value_difference['end_immunisation'] - europe_overall_value_difference['start_immunisation']

europe_overall_value_difference = europe_overall_value_difference[['country', 'measure', 'immunisation_growth']]
# Top 10 European countries with highest Overall Immunisation gain (between 2010 and 2017)

europe_overall_value_difference.set_index('country').sort_values(by='immunisation_growth', ascending=False).head(10).style.background_gradient(cmap = 'Wistia')
# Top 10 European countries with the lowest Overall Immunisation gain (between 2010 and 2017)

europe_overall_value_difference.set_index('country').sort_values(by='immunisation_growth', ascending=True).head(10).style.background_gradient(cmap = 'Wistia')
# Europes 6 most populated countries (with immunisation data)

europe6_overall = europe_overall.loc[europe_overall.country.isin(['Germany', 'France', 'United Kingdom', 'Italy', 'Spain', 'Poland'])]

europe6_overall
# Big-6 Overall Immunisation Trend per year

europe6_overall_mean = europe6_overall[['year','immunisation_overall']].groupby('year').immunisation_overall.agg(['mean', 'min', 'max']).round(1).rename(columns={'mean': 'big-6_overall_mean'})

europe6_overall_mean
# Big-6 Overall Immunisation per year (the contribution of the three childimmunisations)

europe6_overall_mean = europe6_overall[['year', 'immunisation_mea', 'immunisation_dtp', 'immunisation_hep', 'immunisation_overall']].groupby('year').agg('mean').round(1).rename(columns={'immunisation_mea': 'big-6_mea_mean', 'immunisation_dtp': 'big-6_dtp_mean', 'immunisation_hep': 'big-6_hep_mean', 'immunisation_overall': 'big-6_overall_mean'})

europe6_overall_mean.style.background_gradient(cmap="Wistia")
# The Big-6 mean in comparison to the European mean

plt.style.use('seaborn-dark')

plt.figure(figsize=(20,8))



europe_compare_overall_mean = pd.merge(europe6_overall_mean, europe_overall_mean, on = 'year', how = 'left')

europe_compare_overall_mean1 = europe_compare_overall_mean[['big-6_overall_mean', 'european_overall_mean']]



sns.lineplot(data=europe_compare_overall_mean1)

plt.title('Big-6 mean VS European mean', fontsize = 20)

plt.xlabel('year')

plt.ylabel('Immunisation (in %)')

plt.show()
# Big-6 Most recent Overall Immunisation data

europe6_overall_recent = europe6_overall.loc[europe6_overall.groupby('country').year.idxmax()]

europe6_overall_recent[['country', 'measure', 'immunisation_overall']].set_index('country').sort_values(by='immunisation_overall', ascending=False).head(6).style.background_gradient(cmap = 'Wistia')
# Big-6 Overall Immunisation growth (between 2010 and 2017)

europe6_overall_value_difference = europe_overall_value_difference.loc[europe_overall_value_difference.country.isin(['Germany', 'France', 'United Kingdom', 'Italy', 'Spain', 'Poland'])]

europe6_overall_value_difference = europe6_overall_value_difference[['country', 'measure', 'immunisation_growth']]

europe6_overall_value_difference.set_index('country').sort_values(by='immunisation_growth', ascending=False).head(6).style.background_gradient(cmap = 'Wistia')
# Big-6 Overall Immunisation trend per country

plt.style.use('seaborn-dark')

plt.figure(figsize=(20,8))



europe6_overall1= europe6_overall[['country', 'year', 'immunisation_overall']].set_index('year')

europe6_overall2 = europe6_overall1.pivot_table('immunisation_overall', ['year'], 'country')



sns.lineplot(data=europe6_overall2)

plt.title('Big-6 Overall Immunisation Trend', fontsize = 20)

plt.xlabel('year')

plt.ylabel('Immunisation (in %)')

plt.show()
# Stability of the Big-6 Overall Immunisation rates (per country)

europe6_overall_std = europe6_overall.loc[europe6_overall.country.isin(['Germany', 'France', 'United Kingdom', 'Italy', 'Spain', 'Poland'])]

europe6_overall_std.groupby('country').immunisation_overall.agg(['std']).round(1).rename(columns={'std': 'standard_deviation'}).sort_values(by='standard_deviation', ascending=True).head(6).style.background_gradient(cmap = 'Wistia')