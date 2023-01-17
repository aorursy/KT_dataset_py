import math



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
countries_df = pd.read_csv('/kaggle/input/world-bank-education/countries.csv')

education_data_df = pd.read_csv('/kaggle/input/world-bank-education/education_data.csv')

indicators_df = pd.read_csv('/kaggle/input/world-bank-education/indicators.csv')



display(countries_df.head())

display(education_data_df.head())

display(indicators_df.head())
ExactCountofRows = len(education_data_df.index)

print('ExactCountofRows: ', ExactCountofRows)
df = education_data_df.merge(countries_df, left_on='country_id', right_on='id', how='inner') # Inner join "countries" table

df = df[df['is_country'] == True] # Only actual* countries



print('actual_countries: ', len(df))
df = education_data_df.merge(countries_df, left_on='country_id', right_on='id', how='inner') # Inner join "countries" table

df = df[df['is_country'] == True] # Only actual* countries

df = df.merge(indicators_df, left_on='indicator_id', right_on='id', how='inner', suffixes=('','_indicator')) # Inner join "indicators" table



indicator_name = df.groupby('name_indicator')["country_id"].nunique().sort_values(ascending=True).reset_index(name='count')['name_indicator'][:2].rename('indicator_name')

display(indicator_name)
df = education_data_df.merge(countries_df, left_on='country_id', right_on='id', how='inner') # Inner join "countries" table

df = df[df['is_country'] == True] # Only actual* countries

df = df.merge(indicators_df, left_on='indicator_id', right_on='id', how='inner', suffixes=('','_indicator')) # Inner join "indicators" table



country_names = []

medians = []

for each in df.groupby(['country_id'])['country_id'].nunique().index:

    sub_df = df[df['country_id'] == each]

    country_name = sub_df['name'].values[0]

    

    # Find median

    try:

        median = sub_df.groupby(['country_id', 'indicator_id'])["year"].nunique().sort_values(ascending=False).iloc[[(math.floor((len(indicators_df)+1)/2)), 

                                                                                                                     (math.floor((len(indicators_df)+2)/2))]].median()

    except:

        # If not available means total number of indicators are below middle basically median is 0

        median = 0

    

    

    # Append to array

    country_names.append(country_name)

    medians.append(median)



most_complete = pd.DataFrame({'country_name': country_names, 'median': medians})

display(most_complete.sort_values(by=['median'], ascending=False)[:1])
df = education_data_df.merge(countries_df, left_on='country_id', right_on='id', how='inner') # Inner join "countries" table

df = df[df['is_country'] == True] # Only actual* countries

df = df.merge(indicators_df, left_on='indicator_id', right_on='id', how='inner', suffixes=('','_indicator')) # Inner join "indicators" table

df = df.sort_values(by=['country_id', 'indicator_id', 'year']) # Sort the table

df = df[df['value'] != 0] # Drop zeros to prevent inf changes



df['lagged_value'] = df.groupby(['country_id', 'indicator_id'])['value'].shift(1) # Shift previous value to next row

df['year_dif'] = df.groupby(['country_id', 'indicator_id'])['year'].shift(1)  # Shift previous year to next row 



df['year'] = pd.to_datetime(df['year']) # Convert column to datetime

df['year_dif'] = df['year'] - pd.to_datetime(df['year_dif']) # Convert column to datetime and remove it from previous date

df['year_dif'] = df['year_dif']/np.timedelta64(1,'Y') # Convert to years



df = df[df['year_dif'] <= 3] # Year difference rule



df['percentage_change'] = df.groupby(['country_id', 'indicator_id'])['lagged_value'].pct_change() # Find percentage change

df = df.sort_values(by=['percentage_change'], ascending=False)  # Sort by percentage change



result = df[['name', 'name_indicator', 'year', 'percentage_change']].rename(columns={'name': 'country_name',

                                                                                     'name_indicator': 'indicator_name',

                                                                                     'year': 'base_year'}) # Select and rename header names

display(result[:1])