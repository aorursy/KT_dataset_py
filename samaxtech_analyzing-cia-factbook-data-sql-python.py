import pandas as pd
import sqlite3

conn = sqlite3.connect('../input/factbook-data/factbook.db')

q1 = "SELECT * FROM sqlite_master WHERE type='table';" 
pd.read_sql_query(q1, conn)
q2 = "SELECT * FROM facts LIMIT 5;"
pd.read_sql_query(q2, conn)
q3 = '''
SELECT MIN(Population) min_population, MAX(Population) max_population,
MIN(population_growth) min_pop_growth, MAX(population_growth) max_pop_growth
FROM facts
'''
pd.read_sql_query(q3, conn)
q4 = '''
SELECT name, population FROM facts
WHERE population = 0 OR population > 7000000000
'''
pd.read_sql_query(q4, conn)
import matplotlib.pyplot as plt
import seaborn as sns

%matplotlib inline

q5 = '''
SELECT population, population_growth, birth_rate, death_rate FROM facts
WHERE population < (SELECT MAX(Population) FROM facts) AND population > 0;
'''
df1 = pd.read_sql_query(q5, conn)

fig = plt.figure(figsize=(10,8));
ax1 = fig.add_subplot(1,1,1);
df1.hist(ax=ax1);
ax1.grid(False)
q6 = '''
SELECT name heavily_populated, population FROM facts
WHERE population > 100000000 AND name != (SELECT name FROM facts WHERE name IN ('European Union'))
ORDER BY population DESC
LIMIT 11;
'''

heavily_populated = pd.read_sql_query(q6, conn)
heavily_populated = heavily_populated[heavily_populated['heavily_populated']!='World']

ax = heavily_populated.plot(x='heavily_populated', y='population', kind='bar', rot=40, legend=False)
ax.set_title('Heavily Populated Countries')
ax.set_xlabel("Country")
ax.set_ylabel("Population")
ax.grid(False);
q7 = '''
SELECT name, CAST(population as float)/CAST(area_land as float) population_density FROM facts
WHERE population < (SELECT MAX(Population) FROM facts) AND population > 0;
'''

population_density = pd.read_sql_query(q7, conn)
fig = plt.figure(figsize=(10,8));
ax3 = fig.add_subplot(1,1,1);
population_density.hist(ax=ax3);
ax3.grid(False)
population_density.sort_values('population_density', ascending=False, inplace=True)
print(population_density.head(20))
populated_countries = heavily_populated['heavily_populated'].tolist()
populated_countries_density = population_density[population_density['name'].isin(populated_countries)]
populated_countries_density = populated_countries_density.sort_values('population_density', ascending=False)

q8 = "SELECT * FROM facts;"
countries = pd.read_sql_query(q8, conn)
large_countries = countries[countries['name'].isin(populated_countries)].sort_values('area_land', ascending=False)

ax1 = populated_countries_density.plot(x='name', y='population_density', kind='bar', rot=40, legend=False)
ax1.set_title('Population Density of\n Highly Populated Countries')
ax1.set_xlabel("Country")
ax1.set_ylabel("Population Density/Square Meter")
ax1.grid(False);

ax2 = large_countries.plot(x='name', y='area_land', kind='bar', rot=40, legend=False)
ax2.set_title('Area of Highly Populated Countries')
ax2.set_xlabel("Country")
ax2.set_ylabel("Area (Square Meters)")
ax2.grid(False);
q9 = '''
SELECT name, CAST(area_water as float)/CAST(area as float) water_ratio FROM facts
WHERE population < (SELECT MAX(Population) FROM facts) AND population > 0;
'''

water_ratio = pd.read_sql_query(q9, conn).sort_values('water_ratio', ascending=False).head(20)
water_ratio
q11 = '''
SELECT name water_countries FROM facts
WHERE area_water > area_land AND (population < (SELECT MAX(Population) FROM facts) AND population > 0);
'''

water_countries = pd.read_sql_query(q11, conn).head(20)
water_countries
