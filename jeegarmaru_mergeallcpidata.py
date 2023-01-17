# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input/'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



pd.set_option('display.max_columns', 1000)

base = "/kaggle/input/corruption-perceptions-index-for-10-years"



# Any results you write to the current directory are saved as output.
files_by_year = {str(x):f"CPI_{x}_final_dataset.csv" for x in range(2010, 2020)}

dfs = {k:pd.read_csv(f"{base}/{v}") for k,v in files_by_year.items()}
final_cols = ['Year', 'Country', 'CPI Score', 'Rank', 'Number of Sources', 'Minimum score', 'Maximum score',

              'Standard Error', '90% Confidence Interval (Higher bound)', '90% Confidence Interval (Lower bound)',

              'ADB', 'AfDB', 'BF_SGI', 'BF_TI', 'EIU', 'FH', 'GI', 'IMD', 'PERC', 'PRS', 'TI', 'VDP', 'WB', 'WEF', 'WJP'

]
col_map = {'Country Rank': 'Rank', 'Country / Territory': 'Country', 'CPI 2010 Score': 'CPI Score', 

           'Surveys Used': 'Number of Sources', 'Standard Deviation': 'Standard Error', 'ADB 2009': 'ADB',

           'AfDB 2009': 'AfDB', 'BF 2009': 'BF_TI', 'EIU 2010': 'EIU', 'FH 2010': 'FH', 'GI 2010': 'GI',

           'IMD 2010': 'IMD', 'PERC2010': 'PERC', 'WB 2009': 'WB', 'WEF 2010': 'WEF'}

df_2010 = dfs['2010'].rename(columns=col_map)

df_2010
col_map.update({'CPI 2011 Score': 'CPI Score', 'AFDB' : 'AfDB', 'EIU_CRR': 'EIU', 'FH_NIT': 'FH', 'GI_CRR': 'GI', 'IMD2011': 'IMD', 'PERC2011': 'PERC', 'WEF2011': 'WEF', 'PERC2010': 'PERC2010', 'PRS_ICRG': 'PRS', 'TI_BPI': 'TI', 'WB_CPIA': 'WB', 'WJP_ROL': 'WJP'})

df_2011 = dfs['2011'].rename(columns=col_map)

df_2011
def append(df1, df2, year1, year2):

    if year1:

        df1['Year'] = year1

    if year2:

        df2['Year'] = year2

    return df1.loc[:, final_cols].append(df2.loc[:, final_cols], ignore_index=True)

result = append(df_2010, df_2011, 2010, 2011)

result
def rename_and_append(df, result, year, col_map):

    df = df.rename(columns=col_map)

    return append(result, df, None, year)
col_map.update({'CPI 2012 Score': 'CPI Score', '90% Confidence interval (Lower bound)': '90% Confidence Interval (Lower bound)',

                '90% Confidence interval (Higher bound)': '90% Confidence Interval (Higher bound)',

                'BF (SGI)' : 'BF_SGI', 'BF (BTI)': 'BF_TI', 'ICRG': 'PRS', 'PERC2011': 'PERC',

                'WEF2011': 'WEF', 'PERC2010': 'PERC2010'})

result = rename_and_append(dfs['2012'], result, 2012, col_map)

result
col_map.update({'CPI 2013 Score': 'CPI Score'})

result = rename_and_append(dfs['2013'], result, 2013, col_map)

result
col_map.update({'Country/Territory': 'Country', 'CPI 2014': 'CPI Score', 'Number of Surveys Used': 'Number of Sources', 'Std Error': 'Standard Error', 'Min': 'Minimum score', 'Max': 'Maximum score', '90% Lower CI': '90% Confidence Interval (Lower bound)', '90% Upper CI': '90% Confidence Interval (Higher bound)'})

result = rename_and_append(dfs['2014'], result, 2014, col_map)

result
col_map.update({'CPI 2015 Score': 'CPI Score', 'Min': 'Minimum score', 'Max': 'Maximum score', '90%Upper CI': '90% Confidence Interval (Higher bound)', '90% Lower CI': '90% Confidence Interval (Lower bound)'})

col_map.update({'World Bank CPIA': 'WB', 'World  Economic Forum EOS': 'WEF', 'Bertelsmann  Foundation TI': 'BF_TI', 'Arican Development Bank': 'AfDB', 'IMD World Competitiveness Year Book': 'IMD', 'Bertelsmann Foundation SGI': 'BF_SGI', 'World Justice Project ROL': 'WJP', 'PRS Internationl Country Risk Guide': 'PRS', 'Economist Intelligence Unit': 'EIU', 'IHS Global Insight': 'GI', 'PERC Asia Risk Guide': 'PERC', 'Freedom House NIT': 'FH'})

result = rename_and_append(dfs['2015'], result, 2015, col_map)

result
col_map.update({'CPI2016': 'CPI Score', 'World Economic Forum EOS': 'WEF', 

                'Global Insight Country Risk Ratings': 'GI', 

                'Bertelsmann Foundation Transformation Index': 'BF_TI', 

                'African Development Bank CPIA': 'AfDB', 'IMD World Competitiveness Yearbook': 'IMD', 

                'Bertelsmann Foundation Sustainable Governance Index': 'BF_SGI', 

                'World Justice Project Rule of Law Index': 'WJP', 

                'PRS International Country Risk Guide': 'PRS', 'Varities of Democracy Project': 'VDP', 

                'Economist Intelligence Unit Country Ratings': 'EIU', 

                'Freedom House Nations in Transit Ratings': 'FH', 

                'Std Error 2016': 'Standard Error', 'Lower CI': '90% Confidence Interval (Lower bound)',

                'Upper CI': '90% Confidence Interval (Higher bound)'})

result = rename_and_append(dfs['2016'], result, 2016, col_map)

result
col_map.update({'CPI Score 2017': 'CPI Score', 'Rank 2017': 'Rank', 'Standard error 2017': 'Standard Error',

                'Lower CI 2017': '90% Confidence Interval (Lower bound)', 

                'Upper CI 2017': '90% Confidence Interval (Higher bound)', 'Sources': 'Number of Sources',

                'Varieties of Democracy Project': 'VDP'})

result = rename_and_append(dfs['2017'], result, 2017, col_map)

result
col_map.update({'CPI Score 2018': 'CPI Score', 'Rank ': 'Rank', 'Standard error': 'Standard Error',

                'Number of sources': 'Number of Sources', 

                'Lower CI ': '90% Confidence Interval (Lower bound)'})

result = rename_and_append(dfs['2018'], result, 2018, col_map)

result
col_map.update({'CPI score 2019': 'CPI Score', 'standard error ': 'Standard Error'})

result = rename_and_append(dfs['2019'], result, 2019, col_map)

result
# No column has all nulls

for col in result.columns:

    assert not all(result[col].isnull())
no_nulls_cols = ['Year', 'Country', 'CPI Score', 'Rank', '90% Confidence Interval (Lower bound)', 

                 '90% Confidence Interval (Higher bound)', 'Standard Error']

for col in no_nulls_cols:

    assert not any(result[col].isnull())
min_count = min([len(df) for df in dfs.values()])

assert all([x >= min_count for x in result.groupby('Year')['Country'].count().values])
total_count = sum([len(df) for df in dfs.values()])

assert total_count == len(result)
min_max_cols = ['Minimum score', 'Maximum score']

for col in min_max_cols:

    assert set(result[result[col].isnull()]['Year'].drop_duplicates().values) == {2017, 2018, 2019}
assert not any(result['Minimum score'] > result['Maximum score'])

assert not any(result['90% Confidence Interval (Lower bound)'] > result['90% Confidence Interval (Higher bound)'])
result['Number of Sources'] = result['Number of Sources'].astype('int')

result
# Scores for 2010 & 2011 are between 0 & 10 (rather than 0 & 100 for other years; hence multiplying them by 10)

result.loc[result['Year'] == 2010, 'CPI Score'] *= 10

result.loc[result['Year'] == 2011, 'CPI Score'] *= 10



result.loc[result['Year'] == 2010, 'Minimum score'] *= 10

result.loc[result['Year'] == 2011, 'Minimum score'] *= 10

result.loc[result['Year'] == 2010, 'Maximum score'] *= 10

result.loc[result['Year'] == 2011, 'Maximum score'] *= 10



result.loc[result['Year'] == 2010, 'Standard Error'] *= 10

result.loc[result['Year'] == 2011, 'Standard Error'] *= 10



result.loc[result['Year'] == 2010, '90% Confidence Interval (Higher bound)'] *= 10

result.loc[result['Year'] == 2011, '90% Confidence Interval (Higher bound)'] *= 10

result.loc[result['Year'] == 2010, '90% Confidence Interval (Lower bound)'] *= 10

result.loc[result['Year'] == 2011, '90% Confidence Interval (Lower bound)'] *= 10

result
# Standardize the country names so that joins are better

country_renames = {}

country_renames['Brunei'] = ['Brunei Darussalam']

country_renames['Republic of the Congo'] = ['Congo  Republic', 'Congo Republic', 'Congo-Brazzaville', 'Republic of Congo']

country_renames["Côte d'Ivoire"] = ["Cote d'Ivoire", "Côte D'Ivoire", "Côte d´Ivoire", "Côte d’Ivoire"]

country_renames['Democratic Republic of the Congo'] = ['Democratic Republic of Congo', 'The Democratic Republic of Congo', 'Democratic Republic of the Congo ']

country_renames['Guinea-Bissau'] = ['Guinea Bissau']

country_renames['North Korea'] = ['Korea (North)', 'Korea, North']

country_renames['South Korea'] = ['Korea (South)', 'Korea, South']

country_renames['Saint Vincent and the Grenadines'] = ['Saint Vincent and The Grenadines']

country_renames['Sao Tome & Principe'] = ['Sao Tome and Principe']

country_renames['United States of America'] = ['The United States of America', 'United States']
def standardize_country_names(df, col='Country'):

    df = df[df[col] != 'p']

    for key, values in country_renames.items():

        for value in values:

            df.loc[df[col] == value, col] = key

    return df

def merge_ref_data(df1, df2, cols_df1, cols_df2, idx_col1='Country', idx_col2='Country'):

    df1 = df1[[idx_col1] + cols_df1] if cols_df1 else df1.reset_index()

    df2 = df2[[idx_col2] + cols_df2] if cols_df2 else df2.reset_index()

    ref_data1 = standardize_country_names(df1, idx_col1).set_index(idx_col1)

    ref_data2 = standardize_country_names(df2, idx_col2).set_index(idx_col2)



    merged = ref_data1.merge(ref_data2, left_index=True, right_index=True, how='outer', suffixes=('', '_y'))

    second_cols = [x for x in merged.columns if x.endswith('_y')]

    for col in second_cols:

        first_col = col[:-2]

        merged[first_col].fillna(merged[col], inplace=True)



    cols = [x for x in merged.columns if not x.endswith('_y') and not (idx_col1 != idx_col2 and x == idx_col2)]

    merged = merged[cols]

    merged.index.name = idx_col1

    return merged



ref_data = merge_ref_data(dfs['2019'], dfs['2018'], ['ISO3', 'Region'], ['ISO3', 'Region'])

ref_data
ref_data = merge_ref_data(ref_data, dfs['2017'], None, ['ISO3'])

ref_data
ref_data = merge_ref_data(ref_data, dfs['2016'], None, ['Region', 'WB Code', 'OECD', 'G20', 'BRICS', 'EU', 'Arab states'])

ref_data
ref_data = merge_ref_data(ref_data, dfs['2015'], None, ['Region'], idx_col2='Country/Territory')

ref_data
ref_data = merge_ref_data(ref_data, dfs['2014'], None, ['Region'], idx_col2='Country/Territory')

ref_data
ref_data = merge_ref_data(ref_data, dfs['2013'], None, ['WB Code', 'Region', 'IFS Code'], idx_col2='Country / Territory')

ref_data
ref_data = merge_ref_data(ref_data, dfs['2012'], None, ['Region'], idx_col2='Country / Territory')

ref_data
ref_data['Region'].value_counts()
ref_data.loc[ref_data['Region'] == 'AM', 'Region'] = 'AME'

ref_data['Region'].value_counts()
ref_data[ref_data['ISO3'].isnull()]
ref_data[ref_data['Region'].isnull()]
result = standardize_country_names(result)

result
result = result.merge(ref_data, left_on='Country', right_index=True, how='left')

result
result.to_csv('merged_cpi_data.csv', index=False)