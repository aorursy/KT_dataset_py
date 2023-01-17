import pandas as pd

from pandas import DataFrame

from IPython.display import HTML

import matplotlib.pyplot as plt

import seaborn as sns

import numpy as np

import warnings
HTML('''<script>

code_show=true; 

function code_toggle() {

 if (code_show){

 $('div.input').hide();

 } else {

 $('div.input').show();

 }

 code_show = !code_show

} 

$( document ).ready(code_toggle);

</script>

The raw code for this notebook is by default hidden for easier reading.

To toggle on/off the raw code, click <a href="javascript:code_toggle()">here</a>.''')
df = pd.read_csv("/kaggle/input/suicide-rates-overview-1985-to-2016/master.csv")
df.head(10)
df.sample(10)
df.info()
df.index # observing the total amount of rows
df.isnull().any()
df.drop('HDI for year', axis=1, inplace=True) # dropping the column
arr_year = df['year'].unique()

arr_year.sort()

arr_year # checking any missing year in the range 1985-2016
arr_countries = df['country'].unique()

arr_countries # visualising the list of countries
len(arr_countries) # counting the amount of countries
alpha = 0.7

plt.figure(figsize=(10, 25))

sns.countplot(y='country', data=df, alpha=alpha, color='blue')

plt.title('Observation\'s number by country')

plt.show() # visualisation of the observation's count by country
by_country = df.groupby('country')
# creating a Series with only the country's name and the number of observations

observation = pd.Series()

for country, country_df in by_country:

    observation[country] = len(country_df.loc[df['country']==country])

df_obs = observation.to_frame()

df_obs.rename(columns={0: 'Observations'}, inplace=True)

df_obs

# This approach is faster than create a dictionary and turn it into a dataframe.
# country with the lowest value of observation



index_min = df_obs.idxmin() # index of minimum value

df_obs.loc[index_min, 'Observations']
df_obs.loc[df_obs['Observations']<=100]
# country with the higher value of observation



index_max = df_obs.idxmax()

df_obs.loc[index_max, 'Observations']
df_obs.loc[df_obs['Observations']>=350]
y = df.groupby('year') # grouping by year
by_year = pd.Series()

for year, year_df in y:

    by_year[str(year)] = year_df['suicides_no'].sum() # The object supports both integer and label-based indexing

by_year = by_year.to_frame()

by_year.rename(columns={0: 'Tot_Suicide'}, inplace=True)

by_year.sample(10)
# highlighting tot suicides by year

graph_by_year = by_year.plot(legend=False, grid=True) 

graph_by_year.set_xlabel('Year')

graph_by_year.set_ylabel('Tot Suicide')

plt.title('Tot suicides by year')
# creating a frame with the 10 largest values

largest = by_year.nlargest(10, 'Tot_Suicide')

largest.sort_index(inplace=True) # sorting

largest
# plotting the frame to highlight the trend 

graph_largest = largest.plot(legend=False, grid=True) 

graph_largest.set_xlabel('Year')

graph_largest.set_ylabel('Tot Suicide') 

plt.title('Tot suicides by top 10 years')
# year with the minimum amount of suicides

year_min_index = by_year.idxmin()

by_year.loc[year_min_index, 'Tot_Suicide']
# year with the maximum amount of suicides

year_max_index = by_year.idxmax()

by_year.loc[year_max_index, 'Tot_Suicide']
# group by year and sex

gb_year_sex = df.groupby(['year', 'sex'])

df_year_sex = gb_year_sex[['suicides_no']].sum()

df_year_sex.head(10)
df_year_sex.loc[[1999, 2016]].plot(kind='bar')

plt.title('Suicides by gender on highest and lower year')
# overview of the age's categories

arr_age = df['age'].unique()

arr_age 
a = df.groupby('age') # grouping by age
by_age = pd.Series()

for age, age_df in a:

    by_age[age] = age_df['suicides_no'].sum()

by_age.sort_values(ascending=False, inplace=True)

by_age = by_age.to_frame()

by_age.rename(columns={0: 'Tot_Suicides'}, inplace=True)

by_age
# highlighting the number of suicides per age

graph_by_age = by_age.plot(kind='bar')

graph_by_age.set_xlabel('Age')

plt.title('Tot suicides by age category')
# highlighting the correlation between tot suicides and gender by age category

gb_age_sex = df.groupby(['age', 'sex'])

gb_age_sex = gb_age_sex[['suicides_no']].sum()

gb_age_sex
gb_age_sex.plot(kind='barh', figsize=(10, 10))

plt.title('suicides number by gender per each age category')
gb_age_country = df.groupby(['age', 'country'])

gb_age_country = gb_age_country[['suicides_no']].sum()

gb_age_country.sample(10)
# plotting per each age category the top 5 countries

ages = df['age'].unique()

def plotting_data_frame(data_f, iteration):

    for item in iteration:

        new = data_f.loc[item]

        largest = new.nlargest(5, 'suicides_no')

        largest.plot(kind='barh')

        plt.title(f'{item}')

        plt.show()





plotting_data_frame(gb_age_country, ages)
sx = df.groupby('sex') # grouping by gender
# tot amount of suicides by gender

by_sex = pd.Series()

for sex, sex_df in sx:

    by_sex[sex] = sex_df['suicides_no'].sum()

by_sex = by_sex.to_frame()

by_sex.rename(columns={0: 'Tot_Suicides'}, inplace=True)

by_sex
by_sex.plot(kind='pie', subplots=True, legend=False, figsize=(5, 5))

plt.title('Tot suicides by gender')
# grouping by country and sex

gb_country_gender = df.groupby(['country', 'sex'])

gb_country_gender = gb_country_gender[['suicides_no']].sum()
# pivoting the data frame to have male and female as columns

new_df = gb_country_gender.pivot_table(values='suicides_no', index=['country'], columns=['sex'])

new_df
# creating a new column named 'ratio' with the male/female suicides ratio

new_df['ratio'] = new_df['male'] / new_df['female']

new_df
# checking if there is any 'nan' or 'inf' values:

new_df['ratio'].values
# replacing inf values with nan values

new_df['ratio'].replace(np.inf, np.nan, inplace=True)

# dropping nan values

new_df['ratio'].dropna(inplace=True)

new_df['ratio'].values
new_df.reset_index(inplace=True)
condition = new_df['ratio'] > 5

new_df[condition]
sg = df.groupby('generation')
by_gen = pd.Series()

for gen, gen_df in sg:

    by_gen[gen] = gen_df['suicides_no'].sum()

by_gen = by_gen[['G.I. Generation', 'Silent', 'Boomers', 'Generation X', 'Millenials', 'Generation Z']]

# generations are now ordered chronologically

by_gen = by_gen.to_frame()

by_gen.rename(columns={0: 'Tot_Suicides'}, inplace=True)

by_gen
graph_by_gen = by_gen.plot(kind='barh', legend=False)

graph_by_gen.set_ylabel('Generation')

graph_by_gen.set_xlabel('Tot Suicides')

plt.title('Tot suicides by generation')
gb_country = df.groupby('country') # grouping the dataframe by country
by_country = pd.Series()

for country, country_df in gb_country:   

    by_country[country] = country_df['suicides_no'].sum()

by_country = by_country.to_frame()

by_country.rename(columns={0: 'Tot_Suicide'}, inplace=True)



# visualising the top 15 countries by total number of suicides

by_country_largest = by_country.nlargest(15, 'Tot_Suicide') 

by_country_largest.plot(kind='barh', figsize=(10, 8))

plt.title('Top 15 countries per suicide no')
# visualising the 15 countries with the lowest tot number of suicides

by_country_smallest = by_country.nsmallest(15, 'Tot_Suicide')

by_country_smallest.plot(kind='barh', figsize=(10, 8))

plt.title(' top 15 countries per lowest suicide no')
# re-calling the dataframe previously made grouping by 'country' and 'sex' in the By Gender section.

gb_country_gender.head(10)
# visualising the tot of suicides per gender on the 15 countries with the highest number of suicides

index_largest = by_country_largest.index

df_country_gender = gb_country_gender.loc[index_largest, 'suicides_no'].to_frame()

df_country_gender.plot(kind='barh', figsize=(10, 10))

plt.title('Tot suicides by gender on top 15 countries')
gb_year_country = df.groupby(['year', 'country']) # grouping by year/country

gb_year_country = gb_year_country[['suicides_no']].sum()

gb_1990_1999 = gb_year_country.loc[1990:1999]

gb_1990_1999
years = [x for x in range(1990, 2000)]

plotting_data_frame(gb_1990_1999, years) # using function plotting_data_frame()
# grouping by country/year

gb_country_year_population = df.groupby(['country', 'year']) 

# getting the tot amount of population per year

gb_country_year_population = gb_country_year_population[['suicides_no','population']].sum() 

gb_country_year_population
# getting the top 5 countries indexes:

top_5 = by_country.nlargest(5, 'Tot_Suicide')

top_5_indexes = top_5.index
warnings.filterwarnings('ignore') # to ignore warning related to older pandas version

def plotting(data_f, iteration):

    for item in iteration:

        data_f.loc[[item]].unstack(level=0).plot(subplots=True, figsize=(8, 8))



plotting(gb_country_year_population, top_5_indexes)
df_trimmed = df[['country', 'suicides_no', 'population']]

df_trimmed = df_trimmed.set_index('country')

df_trimmed
df_trim_gb = df_trimmed.groupby('country')
df_russia = df_trim_gb.get_group('Russian Federation')

df_usa = df_trim_gb.get_group('United States')

df_japan = df_trim_gb.get_group('Japan')

df_france = df_trim_gb.get_group('France')

df_ukr = df_trim_gb.get_group('Ukraine')

df_germany = df_trim_gb.get_group('Germany')

df_korea = df_trim_gb.get_group('Republic of Korea')

df_brazil = df_trim_gb.get_group('Brazil')

df_pol = df_trim_gb.get_group('Poland')

df_uk = df_trim_gb.get_group('United Kingdom')

top10_list = [df_russia, df_usa, df_japan, df_france, df_ukr, df_germany, df_korea, df_brazil, df_pol, df_uk]

dicts_list = []

def making_dict(df):

    new_df = {'Country': df.index[0],

              'Mean population': df['population'].mean(),

              'Tot suicides': df['suicides_no'].sum()}

    dicts_list.append(new_df)



for df_countries in top10_list:

    making_dict(df_countries)



df_pop = pd.DataFrame(dicts_list).set_index('Country')

# calculating suicides/100k population

df_pop['suicides/100k pop'] = df_pop['Mean population'] / df_pop['Tot suicides']

df_pop[['suicides/100k pop']].plot(kind='barh', legend=True, figsize=(10, 8))
df_pop[['suicides/100k pop']].sort_values(by='suicides/100k pop', ascending=False)