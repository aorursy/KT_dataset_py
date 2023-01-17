import pandas as pd
from urllib.request import urlretrieve

urlretrieve('https://hub.jovian.ml/wp-content/uploads/2020/09/countries.csv', 
            'countries.csv')
countries_df = pd.read_csv('countries.csv')
countries_df
num_countries = countries_df.shape[0]
print('There are {} countries in the dataset'.format(num_countries))
continents = countries_df['continent'].unique().tolist()
continents
total_population = countries_df['population'].sum()
print('The total population is {}.'.format(int(total_population)))

most_populous_df = countries_df.sort_values(by = 'population', ascending = False).head(10)
most_populous_df
countries_df['gdp'] = countries_df['population'] * countries_df['gdp_per_capita']
countries_df

country_counts_df = countries_df.groupby('continent')['location'].count()
country_counts_df
continent_populations_df = countries_df.groupby('continent')['population'].sum()
continent_populations_df
urlretrieve('https://hub.jovian.ml/wp-content/uploads/2020/09/covid-countries-data.csv', 
            'covid-countries-data.csv')
covid_data_df = pd.read_csv('covid-countries-data.csv')
covid_data_df
total_tests_missing = covid_data_df['total_tests'].isna().sum()
print("The data for total tests is missing for {} countries.".format(int(total_tests_missing)))
combined_df = countries_df.merge(covid_data_df, on="location")
combined_df
combined_df['tests_per_million'] = combined_df['total_tests'] * 1e6 / combined_df['population']
combined_df['cases_per_million'] = combined_df['total_cases'] * 1e6 / combined_df['population']
combined_df['deaths_per_million'] = combined_df['total_deaths'] * 1e6 / combined_df['population']
combined_df
highest_tests_df = combined_df.sort_values(by = 'tests_per_million', ascending = False).head(10)
highest_tests_df
highest_cases_df = combined_df.sort_values(by = 'cases_per_million', ascending = False).head(10)
highest_cases_df
highest_deaths_df = combined_df.sort_values(by = 'deaths_per_million', ascending = False).head(10)
highest_deaths_df

