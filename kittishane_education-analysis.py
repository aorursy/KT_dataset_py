

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline





# import os

# for dirname, _, filenames in os.walk('/kaggle/input'):

#     for filename in filenames:

#         print(os.path.join(dirname, filename))

# Read pisa test score

pisa_data = pd.read_csv('/kaggle/input/pisa-scores-2015/Pisa mean perfromance scores 2013 - 2015 Data.csv')

pisa_source = pd.read_csv('/kaggle/input/pisa-scores-2015/Pisa mean performance scores 2013 - 2015 Definition and Source.csv')
pisa_data.head()
pisa_source.head()
pisa_data.dropna(subset=['2013 [YR2013]', '2014 [YR2014]','2015 [YR2015]'], thresh = 1, inplace=True)

pisa_data.drop(pisa_data[(pisa_data['2013 [YR2013]'] == '..')&(pisa_data['2014 [YR2014]'] == '..')&(pisa_data['2015 [YR2015]'] == '..')].index, axis=0, inplace=True)

pisa_data.drop(pisa_data[(pisa_data['2013 [YR2013]'] == '...')&(pisa_data['2014 [YR2014]'] == '...')&(pisa_data['2015 [YR2015]'] == '...')].index, axis=0, inplace=True)

pisa_data.head()
# It looks like there's a lot of missing values here

_2013_not_null  = pisa_data.loc[(pd.notnull(pisa_data['2013 [YR2013]'] )) & (pisa_data['2013 [YR2013]'] != '..' ) &(pisa_data['2013 [YR2013]'] != '...' ) ].count()

_2014_not_null  = pisa_data.loc[(pd.notnull(pisa_data['2014 [YR2014]'] )) & (pisa_data['2014 [YR2014]'] != '..' ) &(pisa_data['2014 [YR2014]'] != '...' ) ].count()

_2015_not_null  = pisa_data.loc[(pd.notnull(pisa_data['2015 [YR2015]'] )) & (pisa_data['2015 [YR2015]'] != '..' ) &(pisa_data['2015 [YR2015]'] != '...' ) ].count()

# print(_2013_not_null)

# print(_2014_not_null)

# print(_2015_not_null)
pisa_data['2015 [YR2015]'] = pisa_data['2015 [YR2015]'].map(lambda x: float(x) if x not in  ['..','...']  else np.nan )

# pisa_data.loc[(pd.notnull(pisa_data['2015 [YR2015]'] )) & (pisa_data['Series Name'] == 'PISA: Mean performance on the mathematics scale') ].info()
def pisa_sum(country, year):

    math = pisa_data.loc[(pisa_data['Country Name'] == country ) & (pisa_data['Series Code'] == 'LO.PISA.MAT') , [year]][year]

    reading = pisa_data.loc[(pisa_data['Country Name'] == country ) & (pisa_data['Series Code'] == 'LO.PISA.REA') , [year]][year]

    science = pisa_data.loc[(pisa_data['Country Name'] == country ) & (pisa_data['Series Code'] == 'LO.PISA.SCI') , [year]][year]

    sum_score = (float(math)+float(reading)+float(science))/3 if (math.dtype == np.float64) & (reading.dtype == np.float64) & (science.dtype == np.float64) else np.nan                   

    return sum_score

    

countries = pisa_data['Country Name'].unique()

for country in countries:    

    new_df = pd.DataFrame({

            'Country Name': country,

            'Country Code': pisa_data.drop_duplicates(['Country Name']).loc[pisa_data['Country Name'] == country , ['Country Code']]['Country Code'],

            'Series Name': "PISA: Mean performance in total.",

            'Series Code': 'PISA_TOTAL', 

             "2013 [YR2013]": pisa_sum(country, "2013 [YR2013]"), 

             "2014 [YR2014]": pisa_sum(country, "2014 [YR2014]"),

             "2015 [YR2015]": pisa_sum(country, "2015 [YR2015]")

            

        })

    pisa_data = pd.concat([pisa_data, new_df], ignore_index=True, axis = 'index')



    
total_df = pisa_data.loc[(pisa_data['Series Name'] == 'PISA: Mean performance in total.') & (pd.notnull(pisa_data['2015 [YR2015]']))].copy()

total_df.sort_values(by='2015 [YR2015]',ascending = False, inplace=True)

total_df.head()

_2015_score = total_df[['Country Name','2015 [YR2015]','Country Code']]

# pisa_data.loc[pisa_data['Country Name']]



countries = total_df['Country Name']



fig = plt.figure()

fig.set_size_inches(15,10)

plt.xlabel('Countries')

plt.ylabel("PISA Mean Score")

plt.xticks(rotation='vertical')

bar_graph = plt.bar(countries, _2015_score['2015 [YR2015]'])





# Let's color Thailand to emphasize how shitty we are doing

# Firstly we have to find the index of Thailand

countries = countries.to_list()

thailand_index = countries.index('Thailand')



bar_graph[thailand_index].set_color('red')




import geopandas as gpd

from mpl_toolkits.axes_grid1 import make_axes_locatable



world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))



#merge both data sets using country code/iso_a3 as unique identifiers

geomap_df = world.merge(_2015_score, left_on = 'iso_a3', right_on = 'Country Code')[['geometry','Country Name','2015 [YR2015]']]





fig, ax = plt.subplots()

fig.set_size_inches(20,15)

# fig = plt.figure()

# ax = fig.add_subplot(111)



divider = make_axes_locatable(ax)

cax = divider.append_axes("right", size="2%", pad=0.1)

geomap_df.plot(column=geomap_df['2015 [YR2015]'], legend = True, ax=ax, cax=cax, cmap='RdYlGn',linestyle=":",edgecolor='grey' )

# ax = PHL.plot(figsize=(20,20), color='whitesmoke', linestyle=":", edgecolor='black')







gdp_df = pd.read_csv("../input/gdp-world-bank-data/GDP by Country.csv",skiprows=3)

gdp_df.head()
_10_years_span_gdp = gdp_df[['Country Name','Country Code','2005','2006','2007','2008','2009','2010','2011','2012','2013','2014','2015']].copy()

len(_10_years_span_gdp['Country Name'])
_10_years_span_gdp['mean'] = _10_years_span_gdp.mean(axis =1)

_10_years_span_gdp.sort_values(by=['mean'], inplace=True, ascending=False)

_10_years_span_gdp.head()
pisa_and_gdp = pd.merge(_2015_score[['Country Code','2015 [YR2015]']], _10_years_span_gdp, on='Country Code')[['Country Name','mean','2015 [YR2015]']]

pisa_and_gdp.head()
sns.scatterplot(x=pisa_and_gdp['2015 [YR2015]'], y=pisa_and_gdp['mean'])



cor = pisa_and_gdp['2015 [YR2015]'].corr(pisa_and_gdp['mean']) 

print('correlation coeefficient')

print(cor)
sns.regplot(x=pisa_and_gdp['2015 [YR2015]'], y=pisa_and_gdp['mean'])
# Set your own project id here

PROJECT_ID = 'kaggle-278402'

from google.cloud import bigquery

# Create a "Client" object

client = bigquery.Client(project=PROJECT_ID)
dataset_ref = client.dataset("world_bank_intl_education", project="bigquery-public-data")

dataset = client.get_dataset(dataset_ref)



tables = list(client.list_tables(dataset))

for table in tables:

    print(table.table_id)
query = """

SELECT DISTINCT indicator_name, indicator_code

FROM `bigquery-public-data.world_bank_intl_education.international_education`

WHERE country_name LIKE '%Thailand%' AND

      indicator_name LIKE '%education%' AND

      indicator_name LIKE '%expenditure%' 

"""



# Set up the query (cancel the query if it would use too much of 

# your quota, with the limit set to 1 Gb)

safe_config = bigquery.QueryJobConfig(maximum_bytes_billed=10**9)

query_job = client.query(query, job_config=safe_config)



# API request - run the query, and convert the results to a pandas DataFrame

query_result = query_job.to_dataframe()



# Print the first five rows

pd.options.display.width = 50

pd.options.display.max_colwidth = 200

pd.set_option('display.max_rows', None)

query_result





query = """

SELECT country_name,country_code, AVG(value) as mean_spending

FROM `bigquery-public-data.world_bank_intl_education.international_education`

WHERE 

    indicator_code = "SE.XPD.TOTL.GD.ZS" AND

    year > 2004 AND 

    year < 2016

GROUP BY country_name,country_code

ORDER BY mean_spending DESC

"""



# Set up the query (cancel the query if it would use too much of 

# your quota, with the limit set to 1 Gb)

safe_config = bigquery.QueryJobConfig(maximum_bytes_billed=10**9)

query_job = client.query(query, job_config=safe_config)



# API request - run the query, and convert the results to a pandas DataFrame

exp_on_ed = query_job.to_dataframe()



# Print the first five rows

pd.options.display.width = 50

pd.options.display.max_colwidth = 200

pd.set_option('display.max_rows', None)

exp_on_ed.head()



countries = exp_on_ed['country_name']



fig = plt.figure()

fig.set_size_inches(15,10)

plt.xlabel('Countries')

plt.ylabel("Expenditure on Education in percentage of GDP")

plt.xticks(rotation='vertical')

bar_graph = plt.bar(countries, exp_on_ed['mean_spending'])





countries = countries.to_list()



thailand_index = countries.index('Thailand')

singapore_index = countries.index('Singapore')

finland_index = countries.index('Finland')



bar_graph[thailand_index].set_color('red')

bar_graph[singapore_index].set_color('green')

bar_graph[finland_index].set_color('yellow')



pisa_and_gdp = pd.merge(_2015_score[['Country Code','2015 [YR2015]']], _10_years_span_gdp, on='Country Code')[['Country Name','mean','2015 [YR2015]']]

pisa_and_gdp.head()

# sns.scatterplot(x=pisa_and_gdp['2015 [YR2015]'], y=pisa_and_gdp['mean'])

sns.regplot(x=pisa_and_gdp['2015 [YR2015]'], y=pisa_and_gdp['mean'])





cor = pisa_and_gdp['2015 [YR2015]'].corr(pisa_and_gdp['mean']) 

print('correlation coeefficient')

print(cor)
# score_and_expenditure = pd.merge(_2015_score, exp_on_ed, on='Country Name')

pisa_and_expenditure = exp_on_ed.merge(_2015_score, left_on = 'country_code', right_on = 'Country Code')[['Country Name','Country Code','2015 [YR2015]','mean_spending']]

pisa_and_expenditure.head()
# let's define each for easier code

gov_spending = pisa_and_expenditure['mean_spending']

pisa_score = pisa_and_expenditure['2015 [YR2015]']

sns.regplot(x=pisa_score, y=gov_spending)
# using pandas

cor = pisa_score.corr(gov_spending) 

print(f"Pearson's Correlation Coeefficient from Pandas: {cor}")

# using numpy

# cor_coef = np.corrcoef(pisa_score, gov_spending)

# print(f"Pearson's Correlation Coeefficient from Numpy: {cor_coef[0,1]}")



# using scipi

# import scipy.stats

# correlation_coef, p_value = scipy.stats.pearsonr(merge_2015_score, merge_exp_on_ed)

# print(f"Pearson's Correlation Coeefficient from Scipy: {correlation_coef}")

# print(f"p_value: {p_value}")



#p value is 3% meaning the there is a correlation. But of does there is not gaurantee that the government's expenditure cause that.
