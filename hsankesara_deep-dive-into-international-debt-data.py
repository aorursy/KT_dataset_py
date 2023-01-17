# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input/"))

# Any results you write to the current directory are saved as output.
import bq_helper
from bq_helper import BigQueryHelper
wbid = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="world_bank_intl_debt")
bq_assistant = BigQueryHelper("bigquery-public-data", "world_bank_intl_debt")
bq_assistant.list_tables()
bq_assistant.head("international_debt", num_rows=5)
query1 = """
SELECT
  id.country_name,
  id.value AS debt --format in DataStudio
FROM (
  SELECT
    country_code,
    region
  FROM
    `bigquery-public-data.world_bank_intl_debt.country_summary`
  WHERE
    region != "" ) cs --aggregated countries do not have a region
INNER JOIN (
  SELECT
    country_code,
    country_name,
    value,
    year
  FROM
    `bigquery-public-data.world_bank_intl_debt.international_debt`
  WHERE
    indicator_code = "DT.DOD.PVLX.CD"
    AND year = 2016 ) id
ON
  cs.country_code = id.country_code
ORDER BY
  id.value DESC
;
        """
response1 = wbid.query_to_pandas_safe(query1)
response1.head(5)
import matplotlib.pyplot as plt
import seaborn as sns
bq_assistant.head("country_series_definitions", num_rows=5)
bq_assistant.table_schema('country_series_definitions')
# Number of countries
query2 = """
SELECT count(distinct country_code) from `bigquery-public-data.world_bank_intl_debt.country_series_definitions`
"""

# Number of series code 
query3 = """
SELECT count(distinct series_code) from `bigquery-public-data.world_bank_intl_debt.country_series_definitions`
"""

# total rows in the table
query4 = """
SELECT count(country_code) from `bigquery-public-data.world_bank_intl_debt.country_series_definitions`
"""
# Number of countries
response2 = wbid.query_to_pandas_safe(query2)
response3 = wbid.query_to_pandas_safe(query3)
response4 = wbid.query_to_pandas_safe(query4)
print('Total number of countries:', response2.at[0, 'f0_'])
print('Total number of series codes:', response3.at[0, 'f0_'])
print('Total number of rows:', response4.at[0, 'f0_'])
country_series_map_q = """
SELECT country_code, count(series_code) from `bigquery-public-data.world_bank_intl_debt.country_series_definitions` group by country_code
""" 
country_series_map = wbid.query_to_pandas_safe(country_series_map_q)
country_series_map.columns = ['country_code', 'total_series_per_country']
country_series_map[country_series_map['total_series_per_country'] == country_series_map['total_series_per_country'].max()]
series_country_map_q = """
SELECT series_code, count(country_code) from `bigquery-public-data.world_bank_intl_debt.country_series_definitions` group by series_code
""" 
series_country_map = wbid.query_to_pandas_safe(series_country_map_q)
series_country_map.columns = ['series_code', 'total_countries_per_series']
a4_dims = (25.7, 6.27)
fig, ax = plt.subplots(figsize=a4_dims)
sns.barplot('series_code', 'total_countries_per_series', data = series_country_map)
plt.show()
query5 =  """
SELECT distinct(series_code), description from `bigquery-public-data.world_bank_intl_debt.country_series_definitions` where series_code = 'DT.DOD.DECT.CD'"""
print('DT.DOD.DECT.CD is a ' + wbid.query_to_pandas_safe(query5).iloc[1,1])
query6 =  """
SELECT distinct(series_code), description from `bigquery-public-data.world_bank_intl_debt.country_series_definitions` where series_code = 'BX.KLT.DINV.CD.DT'"""
print('BX.KLT.DINV.CD.DT is ' + wbid.query_to_pandas_safe(query6).iloc[0,1])
bq_assistant.head("country_summary", num_rows=5)
bq_assistant.table_schema('country_summary')
response1 = wbid.query_to_pandas_safe("SELECT distinct(income_group) from `bigquery-public-data.world_bank_intl_debt.country_summary` ")
response1.head(10)
regions = wbid.query_to_pandas_safe("""SELECT * from `bigquery-public-data.world_bank_intl_debt.country_summary` where income_group = ''""")
regions.head()
response1 = wbid.query_to_pandas_safe("""SELECT short_name, income_group from `bigquery-public-data.world_bank_intl_debt.country_summary`""")
response1.head(5)
response1 = wbid.query_to_pandas_safe("SELECT distinct(lending_category) from `bigquery-public-data.world_bank_intl_debt.country_summary` ")
response1
country_summary = wbid.query_to_pandas_safe("SELECT * from `bigquery-public-data.world_bank_intl_debt.country_summary` ")
country_summary.head()
countries_details = country_summary.drop(country_summary[country_summary['region'] == ''].index).reset_index().drop(['index'], axis = 1)
## countries_datails contains details of countries only
countries_details.head()
countries_details['lending_category'].unique()
lending_details = countries_details.groupby('lending_category').count().reset_index()
lending_details = lending_details[['lending_category', 'country_code']]
lending_details.columns= ['lending_category', 'total_number_of_countries']
sns.barplot(x = 'lending_category', y = 'total_number_of_countries', data = lending_details)
plt.show()
## Assign black column value undefine
countries_details.loc[countries_details['system_of_trade'] == '', 'system_of_trade'] = 'undefined'
countries_details['system_of_trade'].unique()
trading_details = countries_details.groupby('system_of_trade').count().reset_index()
trading_details = trading_details[['system_of_trade', 'country_code']]
trading_details.columns= ['system_of_trade', 'total_number_of_countries']
sns.barplot(x = 'system_of_trade', y = 'total_number_of_countries', data = trading_details)
plt.show()
special_trade_countries = countries_details[countries_details['system_of_trade'] == 'Special trade system'][['country_code', 'short_name', 'region']]
special_trade_countries.head()
special_trade_countries_region_wise = special_trade_countries.groupby(['region']).count().reset_index()
special_trade_countries_region_wise.drop(['short_name'], axis = 1, inplace=True)
special_trade_countries_region_wise.columns = ['region', 'number_of_countries']
a4_dims = (25.7, 6.27)
fig, ax = plt.subplots(figsize=a4_dims)
sns.barplot('region', 'number_of_countries', data=special_trade_countries_region_wise)
plt.show()
countries_details['government_accounting_concept'].unique()
## Assign black column value to undefine
countries_details.loc[countries_details['government_accounting_concept'] == '', 'government_accounting_concept'] = 'undefined'
a4_dims = (10.7, 4.27)
fig, ax = plt.subplots(figsize=a4_dims)
gove_accounting_details = countries_details.groupby('government_accounting_concept').count().reset_index()
gove_accounting_details = gove_accounting_details[['government_accounting_concept', 'country_code']]
gove_accounting_details.columns= ['government_accounting_concept', 'total_number_of_countries']
sns.barplot(x = 'government_accounting_concept', y = 'total_number_of_countries', data = gove_accounting_details)
plt.show()
countries_details.loc[countries_details['imf_data_dissemination_standard'] == '', 'imf_data_dissemination_standard'] = 'undefined'

a4_dims = (25.7, 6.27)
fig, ax = plt.subplots(figsize=a4_dims)
imf_data_dissemination_standard_details = countries_details.groupby('imf_data_dissemination_standard').count().reset_index()
imf_data_dissemination_standard_details  = imf_data_dissemination_standard_details [['imf_data_dissemination_standard', 'country_code']]
imf_data_dissemination_standard_details .columns= ['imf_data_dissemination_standard', 'total_number_of_countries']
sns.barplot(x = 'imf_data_dissemination_standard', y = 'total_number_of_countries', data = imf_data_dissemination_standard_details)
plt.show()
bq_assistant.head("international_debt", num_rows=5)
total_countries = wbid.query_to_pandas_safe("""SELECT count(distinct(country_name)) from `bigquery-public-data.world_bank_intl_debt.international_debt`""")
total_rows = wbid.query_to_pandas_safe("""SELECT count(country_name) from `bigquery-public-data.world_bank_intl_debt.international_debt`""")
total_indicators = wbid.query_to_pandas_safe("""SELECT count(distinct(indicator_code)) from `bigquery-public-data.world_bank_intl_debt.international_debt`""")
print('Total Number of countries are', total_countries.iloc[0,0])
print('Total Number of indicators are', total_indicators.iloc[0,0])
print('Total Number of rows are', total_rows.iloc[0,0])
bq_assistant.table_schema('international_debt')
total_debt = """
SELECT country_name, country_code, year, sum(value) from `bigquery-public-data.world_bank_intl_debt.international_debt` group by country_code, year, country_name
"""
yearwise_debt = wbid.query_to_pandas_safe(total_debt)
yearwise_debt.columns = ['country_name', 'country_code', 'year', 'debt']
yearwise_debt.head()
yearwise_debt.info()
yearwise_debt.describe()
## Converting year attribute to datetime datatype
yearwise_debt['year'] = pd.to_datetime(yearwise_debt['year'], format='%Y')
## Plotting debt vs year graph
def plot_debt(country_code, country_name = None):
    country_yearwise_debt = yearwise_debt[yearwise_debt['country_code'] == country_code]
    plt.plot_date(country_yearwise_debt['year'], country_yearwise_debt['debt'])
    plt.xlabel('Years')
    plt.ylabel('Debt')
    if country_name is None:
        country_name = country_code
    plt.title(country_name + ' yearly debt analysis')
plot_debt('AFG')
plt.show()
plot_debt('IND', 'India')
plt.show()
plt.figure(1)
plot_debt('LIC', 'Low Income Countries')
plt.figure(2)
plot_debt('MIC', 'Middle Income Countries')
plt.figure(3)
plot_debt('LMC', '       Low-Middle Income Countries')
plt.figure(4)
plot_debt('UMC', '       Upper-Middle Income Countries')
plt.show()
plot_debt('LDC')
plt.show()
countries_details.head()
country_region_dynamics = countries_details.groupby(['region']).count().reset_index()
country_region_dynamics = country_region_dynamics[['region', 'country_code']]
country_region_dynamics.columns = ['region', 'number_of_countries']
a4_dims = (15.7, 6.27)
fig, ax = plt.subplots(figsize=a4_dims)
sns.barplot('region', 'number_of_countries', data=country_region_dynamics)
plt.show()
inc_countries = """
SELECT distinct country_name,cs.income_group  from `bigquery-public-data.world_bank_intl_debt.international_debt` as id join `bigquery-public-data.world_bank_intl_debt.country_summary` as cs on id.country_code = cs.country_code where cs.income_group != ''
"""
income_county_dynamic = wbid.query_to_pandas_safe(inc_countries)
income_county_dynamic.head()
country_per_income_grp = income_county_dynamic.groupby(['income_group']).count().reset_index()
country_per_income_grp.columns = ['income_group', 'total_number_of_countries']
a4_dims = (8, 6)
fig, ax = plt.subplots(figsize=a4_dims)
plt.pie(labels=country_per_income_grp['income_group'], x=country_per_income_grp['total_number_of_countries'], autopct='%1.0f%%')
plt.show()
yearwise_debt[yearwise_debt['debt'] == yearwise_debt['debt'].max()]
lmy_countries = """
SELECT distinct(country_name) from `bigquery-public-data.world_bank_intl_debt.international_debt` as id join `bigquery-public-data.world_bank_intl_debt.country_summary` as cs on id.country_code = cs.country_code where  cs.income_group = 'Lower middle income'
"""
## List of lower middle class countries
response1 = wbid.query_to_pandas_safe(lmy_countries)
response1
no_income_group = wbid.query_to_pandas_safe("""SELECT country_code, short_name from `bigquery-public-data.world_bank_intl_debt.country_summary` where income_group = ''""")
no_income_group.head()
yearwise_debt.head()
## region wise debt
debt_region_wise = yearwise_debt.merge(no_income_group, how = 'inner', on='country_code')
debt_region_wise.head()
debt_region_wise['country_code'].unique()
plot_debt('EAP')
plt.show()
## Region with the highest and the lowest debt 
print(debt_region_wise[debt_region_wise['country_code'] == debt_region_wise['country_code'].max()]['country_code'].unique())
print(debt_region_wise[debt_region_wise['country_code'] == debt_region_wise['country_code'].min()]['country_code'].unique())
plt.figure(1)
plot_debt('EAP', 'East Asia & Pacific')
plt.figure(2)
plot_debt('UMC', 'Upper Middle countries')
plt.show()
plot_debt('IDX')
bq_assistant.head("series_summary", num_rows=5)
bq_assistant.table_schema('series_summary')
response1 = wbid.query_to_pandas_safe("""SELECT distinct(periodicity) from `bigquery-public-data.world_bank_intl_debt.series_summary` """)
response1
bq_assistant.head("series_times", num_rows=5)
code_region_mapping = countries_details[['country_code', 'region']]
total_debt = yearwise_debt.groupby(by=['country_code']).sum().reset_index()
total_debt.head()
total_debt = total_debt.merge(code_region_mapping)
total_debt['region'].unique()
a4_dims = (15.7, 6.27)
fig, ax = plt.subplots(figsize=a4_dims)
sns.violinplot('region', 'debt', data=total_debt)
plt.show()
#a4_dims = (15.7, 6.27)
#fig, ax = plt.subplots(figsize=a4_dims)
sns.factorplot('region', 'debt', data=total_debt, size=12)
plt.show()
a4_dims = (15.7, 6.27)
fig, ax = plt.subplots(figsize=a4_dims)
sns.swarmplot('region', 'debt', data=total_debt, size=15)
plt.show()
a4_dims = (15.7, 6.27)
fig, ax = plt.subplots(figsize=a4_dims)
sns.boxplot('region', 'debt', data=total_debt)
plt.show()
a4_dims = (15.7, 6.27)
fig, ax = plt.subplots(figsize=a4_dims)
sns.lvplot('region', 'debt', data=total_debt)
plt.show()
total_debt[total_debt['debt'] == total_debt['debt'].max()]
plot_debt('CHN', 'china')
plt.show()
code_lending_mapping = countries_details[['country_code', 'lending_category']]
total_debt = total_debt.merge(code_lending_mapping)
sns.factorplot('lending_category', 'debt', data=total_debt)
plt.show()
a4_dims = (15.7, 6.27)
fig, ax = plt.subplots(figsize=a4_dims)
sns.lvplot('lending_category', 'debt', data=total_debt)
plt.show()
year_debt = yearwise_debt.groupby('year').sum().reset_index()
year_debt.head()
a4_dims = (15.7, 6.27)
fig, ax = plt.subplots(figsize=a4_dims)
plt.plot(year_debt['year'], year_debt['debt'])
plt.xlabel('year')
plt.ylabel('debt')
plt.title('Total debt taken from world bank')
plt.show()
sns.distplot(year_debt['debt'])
plt.show()
pd.plotting.lag_plot(year_debt['debt'])
plt.show()
year_debt['debt'].rolling(4).mean().plot(figsize=(10,6), linewidth=5, fontsize=20)
plt.show()
year_debt['debt'].diff(4).plot(figsize=(10,6), linewidth=5, fontsize=20)
plt.show()
pd.plotting.autocorrelation_plot(year_debt['debt'])
plt.show()

debt_each_country = yearwise_debt.groupby(by=['country_name']).sum().reset_index()
debt_each_country = debt_each_country.merge(income_county_dynamic)
a4_dims = (10, 6)
fig, ax = plt.subplots(figsize=a4_dims)
sns.boxplot('income_group', 'debt', data=debt_each_country)
plt.show()
a4_dims = (10, 6)
fig, ax = plt.subplots(figsize=a4_dims)
sns.violinplot('income_group', 'debt', data=debt_each_country)
plt.show()
sns.factorplot('income_group', 'debt', data=debt_each_country, size=6)
plt.show()
debt_each_country_d = pd.get_dummies(columns=['income_group'], data=debt_each_country)
sns.heatmap(debt_each_country_d[['debt', 'income_group_Low income','income_group_Lower middle income', 
                               'income_group_Upper middle income']].corr(), annot=True, cmap='BrBG')
plt.show()
yearwise_debt = yearwise_debt.merge(income_county_dynamic)
yearwise_debt.head()
yearwise_debt = yearwise_debt.groupby(['year', 'income_group']).mean().reset_index()
yearwise_debt.head()
ax = sns.factorplot(x = 'year', y = 'debt', hue='income_group', data=yearwise_debt,size=8)
ax.set_xticklabels(rotation=90, ha="right")
plt.tight_layout()
plt.show()