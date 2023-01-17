import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline
suicide_data = pd.read_csv('../input/suicide-rates-overview-1985-to-2016/master.csv')

suicide_data.head()
suicide_data.info()
suicide_data.drop(['country-year','HDI for year'], axis = 1, inplace = True)
suicide_data.tail()
year_group = suicide_data.groupby('year')
years = suicide_data.year.unique()

years.sort()
world_suicides_per_100k_pop = ((year_group.suicides_no.sum()) / (year_group.population.sum())) *100000
worldwide_total_suicide_by_year = year_group['suicides_no'].agg('sum')
fig = plt.figure(figsize = (12,6))

plt.plot(years, world_suicides_per_100k_pop, marker = 'o', label = 'Suicides per 100k pop.')

plt.xticks(years, rotation = 45)

plt.xlabel('Year')

plt.ylabel('Suicides/100k pop.')

plt.title('Wordwide Suicides per 100k pop. from 1985-2016')

plt.legend()

plt.grid()

plt.show()

fig.savefig('Wordwide Suicides per 100k pop from 1985-2016')
fig = plt.figure(figsize = (12,6))

plt.bar(years, worldwide_total_suicide_by_year, color = 'g')

plt.xticks(years, rotation = 45)

plt.xlabel('Year')

plt.ylabel('Suicides')

plt.title('Wordwide Total Suicides from 1985-2016')

plt.show()

fig.savefig('Wordwide Total Suicides from 1985-2016')
male_group = suicide_data[suicide_data['sex']=='male'].groupby('year')

female_group = suicide_data[suicide_data['sex']=='female'].groupby('year')
world_male_suicides_per_100k_pop = (male_group['suicides_no'].sum()) / (male_group['population'].sum())*100000

world_female_suicides_per_100k_pop = (female_group['suicides_no'].sum()) / (female_group['population'].sum())*100000
fig = plt.figure(figsize = (12,6))

plt.plot(years, world_male_suicides_per_100k_pop, marker = 'o', label = 'male', color = 'orange')

plt.plot(years, world_female_suicides_per_100k_pop, marker = 'o', label = 'female', color = 'red')

plt.xticks(years, rotation = 45)

plt.xlabel('Year')

plt.ylabel('Suicides/100k pop.')

plt.title('Wordwide Suicides per 100k pop. by Sex ')

plt.legend()

plt.grid()

plt.show()

fig.savefig('Wordwide Suicides per 100k pop by Sex')
fig = plt.figure(figsize = (12,6))

plt.bar(years, male_group['suicides_no'].sum(), label = 'male', color = 'orange')

plt.bar(years, female_group['suicides_no'].sum(), label = 'female', color = 'red')

plt.xticks(years, rotation = 45)

plt.xlabel('Year')

plt.ylabel('Suicides/100k pop.')

plt.title('Wordwide Total Suicides by Sex from 1985-2016')

plt.legend()

plt.show()

fig.savefig('Wordwide Total Suicides by Sex from 1985-2016' )
male_perc_by_year = (male_group['suicides_no'].sum() / year_group.suicides_no.sum())*100

female_perc_by_year = (female_group['suicides_no'].sum() / year_group.suicides_no.sum())*100
fig = plt.figure(figsize = (12,6))

plt.bar(years, male_perc_by_year, label = 'male', color = 'orange')

plt.bar(years, female_perc_by_year, label = 'female', color = 'red', bottom= np.array(male_perc_by_year))

plt.xticks(years, rotation = 45)

plt.yticks(np.arange(0,110,10))

plt.xlabel('Year')

plt.ylabel('Suicides %')

plt.title('Wordwide Suicides Percentage % by Sex')

plt.legend()

plt.show()

fig.savefig('Wordwide Suicides Percentage % by Sex')
suicide_data.age.value_counts()
world_5_14_age_group = suicide_data[suicide_data['age']=='5-14 years'].groupby('year')

world_15_24_age_group = suicide_data[suicide_data['age']=='15-24 years'].groupby('year')

world_25_34_age_group = suicide_data[suicide_data['age']=='25-34 years'].groupby('year')

world_35_54_age_group = suicide_data[suicide_data['age']=='35-54 years'].groupby('year')

world_55_74_age_group = suicide_data[suicide_data['age']=='55-74 years'].groupby('year')

world_75_above_age_group = suicide_data[suicide_data['age']=='75+ years'].groupby('year')
a = ((world_5_14_age_group.suicides_no.sum()) / (world_5_14_age_group.population.sum()))*100000

b = (world_15_24_age_group.suicides_no.sum() / (world_15_24_age_group.population.sum()))*100000

c = (world_25_34_age_group.suicides_no.sum() / (world_25_34_age_group.population.sum()))*100000

d = (world_35_54_age_group.suicides_no.sum() / (world_35_54_age_group.population.sum()))*100000

e = (world_55_74_age_group.suicides_no.sum() / (world_55_74_age_group.population.sum()))*100000

f = (world_75_above_age_group.suicides_no.sum() / (world_75_above_age_group.population.sum()))*100000
fig = plt.figure(figsize = (12,6))

plt.plot(a, label = 'age 5-14', color = 'red', marker = 'o')

plt.plot(b, label = 'age 15-24', color = 'indigo', marker = 'o')

plt.plot(c, label = 'age 25-34', color = 'blue', marker = 'o')

plt.plot(d, label = 'age 35-54', color = 'green', marker = 'o')

plt.plot(e, label = 'age 55-74', color = 'yellow', marker = 'o')

plt.plot(f, label = 'age 75 above', color = 'orange', marker = 'o')

plt.xticks(years, rotation = 45)

plt.xlabel('Year')

plt.ylabel('Suicides/100k pop.')

plt.title('Wordwide Suicides per 100k pop. by Age Group')

plt.legend()

plt.grid()

plt.show()

fig.savefig('Wordwide Suicides per 100k pop by Age Group')
suicide_data.generation.value_counts()
world_x_gen_group = suicide_data[suicide_data['generation']=='Generation X'].groupby('year')

world_silent_gen_group = suicide_data[suicide_data['generation']=='Silent'].groupby('year')

world_millenials_gen_group = suicide_data[suicide_data['generation']=='Millenials'].groupby('year')

world_boomers_gen_group = suicide_data[suicide_data['generation']=='Boomers'].groupby('year')

world_gi_gen_group = suicide_data[suicide_data['generation']=='G.I. Generation'].groupby('year')

world_z_gen_group = suicide_data[suicide_data['generation']=='Generation Z'].groupby('year')
g = (world_x_gen_group.suicides_no.sum() / world_x_gen_group.population.sum())*100000

h = (world_silent_gen_group.suicides_no.sum() / world_silent_gen_group.population.sum())*100000

i = (world_millenials_gen_group.suicides_no.sum() / world_millenials_gen_group.population.sum())*100000

j = (world_boomers_gen_group.suicides_no.sum() / world_boomers_gen_group.population.sum())*100000

k = (world_gi_gen_group.suicides_no.sum() / world_gi_gen_group.population.sum())*100000

l = (world_z_gen_group.suicides_no.sum() / world_z_gen_group.population.sum())*100000
fig = plt.figure(figsize = (12,6))

plt.plot(g, label = 'Generation X', color = 'yellow', marker = 'o')

plt.plot(h, label = 'Silent', color = 'indigo', marker = 'o')

plt.plot(i, label = 'Millenials', color = 'blue', marker = 'o')

plt.plot(j, label = 'Boomers', color = 'green', marker = 'o')

plt.plot(k, label = 'G.I. Generation', color = 'red', marker = 'o')

plt.plot(l, label = 'Generation Z', color = 'orange', marker = 'o')

plt.xticks(years, rotation = 45)

plt.xlabel('Year')

plt.ylabel('Suicides/100k pop.')

plt.title('Wordwide Suicides per 100k pop. by Generation')

plt.legend()

plt.grid()

plt.show()

fig.savefig('Wordwide Suicides per 100k pop by Generation')
suicide_data.head()
country_group = suicide_data.groupby('country')
country_suicides_per_100k_pop = (country_group['suicides_no'].sum() / country_group['population'].sum())*100000

country_suicides_per_100k_pop.sort_values(  inplace = True)
fig = plt.figure(figsize = (12,30))

plt.barh(country_suicides_per_100k_pop.index, country_suicides_per_100k_pop, color = 'purple')

plt.xlabel('Suicides per 100k pop.')

plt.xticks(range(0,42,2))

plt.ylabel('Country')

plt.title('Suicides per 100k pop by country From 1985-2016' )

plt.show()

fig.savefig('Suicides per 100k pop by country From 1985-2016')
male_group_country = suicide_data[suicide_data['sex'] == 'male'].groupby('country')

female_group_country = suicide_data[suicide_data['sex'] == 'female'].groupby('country')
male_perc_suicides_by_country = (male_group_country['suicides_no'].sum() / country_group['suicides_no'].sum())*100

female_perc_suicides_by_country = (female_group_country['suicides_no'].sum() / country_group['suicides_no'].sum())*100
male_perc_suicides_by_country
fig = plt.figure(figsize = (12,30))

plt.barh(male_perc_suicides_by_country.index, male_perc_suicides_by_country, color = '#9402f5', label = 'male')

plt.barh(male_perc_suicides_by_country.index, female_perc_suicides_by_country, color = '#f502c4', label = 'female', left = male_perc_suicides_by_country)

plt.xlabel('Percentage %')

plt.xticks(range(0,110,10))

plt.ylabel('Country')

plt.legend()

plt.title('Suicides percentage % ratio of Sex by Country from 1985-2016')

plt.show()

fig.savefig('Suicides percentage % ratio of Sex by Country from 1985-2016')
suicide_data.head()
suicide_data['gdp_per_capita ($)'].describe()
suicide_data.loc[suicide_data['gdp_per_capita ($)'] < 5000, 'Income Category'] = 'Low'

suicide_data.loc[(suicide_data['gdp_per_capita ($)'] > 5000) & (suicide_data['gdp_per_capita ($)'] <25000), 'Income Category'] = 'Medium'

suicide_data.loc[suicide_data['gdp_per_capita ($)'] > 25000, 'Income Category'] = 'High'
suicide_data['Income Category'].value_counts()
suicide_data.tail()
income_group = suicide_data.groupby('Income Category')
income_perc_suicides = round((income_group['suicides_no'].sum() / suicide_data['suicides_no'].sum())*100,1)
income_perc_suicides
fig = plt.figure(figsize = (12,6))

plt.pie(income_perc_suicides, explode = (.03,.03,.03), labels= list(zip(income_perc_suicides.index,income_perc_suicides)))

plt.title('Wordwide Suicides Distribution by per capita GDP from 1985-2016')

plt.xlabel("High: GDP per capita > 25000 ,  Low: GDP per capita < 5000")

plt.show()

fig.savefig('Wordwide Suicides Distribution by per capita GDP from 1985-2016')