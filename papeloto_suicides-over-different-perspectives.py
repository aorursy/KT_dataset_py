import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns
df = pd.read_csv('../input/master.csv')

df.sample(5)
df.info()
# Strip whitespace since some of the columns would become harder to name.

df.columns = df.columns.str.strip()



# Drop unnecessary columns for the analysis objectives.

df.drop(columns=['suicides_no', 'country-year', 'HDI for year', 'gdp_for_year ($)', 'generation'], inplace=True)



df.sample(5)
# Rename columns for easier readability.

df.rename(columns={'suicides/100k pop':'suicides', 'gdp_per_capita ($)':'gpd'}, inplace=True)



df.head()
suicides_sex_age = df[['sex', 'age', 'suicides']].groupby(['sex', 'age']).mean()

suicides_sex_age
# Reorder age index for visualization.

suicides_sex_age.reset_index(inplace=True)

suicides_sex_age['age'] = suicides_sex_age['age'].str.replace(' years', '')



age_sort = {'5-14': 0, '15-24': 1, '25-34': 2, '35-54': 3, '55-74': 4, '75+': 5}

suicides_sex_age['sort'] = suicides_sex_age['age'].map(age_sort)

suicides_sex_age.sort_values(by='sort', inplace=True)

suicides_sex_age.drop('sort', axis=1, inplace=True)



suicides_sex_age
age_groups = suicides_sex_age['age'].unique()

male_suicides = suicides_sex_age[suicides_sex_age['sex'] == 'male']['suicides']

female_suicides = suicides_sex_age[suicides_sex_age['sex'] == 'female']['suicides']



plt.bar(age_groups, male_suicides, label='Male')

plt.bar(age_groups, female_suicides, label='Female')



plt.title('Average suicides across the world by sex')

plt.xlabel('Age group')

plt.ylabel('Suicides per 100k population')

plt.legend()

plt.show()
suicides_vs_gpd = df[['suicides', 'year', 'gpd']].groupby('year').mean()

suicides_vs_gpd.reset_index(inplace=True)



suicides_vs_gpd.head()
fig, ax1 = plt.subplots()



# Plot the suicides over the years.

lns1 = ax1.plot(suicides_vs_gpd['year'], suicides_vs_gpd['suicides'], 'C0', label='Suicides')



# Create a shared axis for plotting on a different scale the GPD.

ax2 = ax1.twinx()

lns2 = ax2.plot(suicides_vs_gpd['year'], suicides_vs_gpd['gpd'], 'C1', label='GPD')



# Join both legends into the same box.

lns = lns1 + lns2

labs = [l.get_label() for l in lns]

ax1.legend(lns, labs, loc=2)



# Set the labels.

ax1.set_ylabel('Suicides per 100k population')

ax2.set_ylabel('GDP per Capita')

ax1.set_xlabel('Years')



plt.tight_layout()

plt.show()
suicides_poor_rich = df[['year', 'country', 'gpd', 'suicides']]



# Sort the the countries by their average gpd over the years.

# Then get the list of the countries ordered.

countries_by_gpd = suicides_poor_rich.groupby('country').mean().sort_values('gpd', ascending=False).index
# Get the top and bottom 5 countries of the list.

top_countries = countries_by_gpd[:5]

bot_countries = countries_by_gpd[-5:]



# Append them for the future filter.

countries_to_compare = top_countries.append(bot_countries)

countries_to_compare
# Filter the rows that only are one of those countries.

suicides_poor_rich = suicides_poor_rich.loc[suicides_poor_rich['country'].isin(countries_to_compare)]

suicides_poor_rich.sample(5)
# Create a filter for splitting those countries into two groups.

country_filter = {country:'TOP' for country in top_countries}

country_filter.update({country:'BOT' for country in bot_countries})



country_filter
# Apply the filter.

suicides_poor_rich['country'] = suicides_poor_rich['country'].map(country_filter)

suicides_poor_rich.sample(5)
# Simply, plot the results.

sns.lineplot(x='year', y='suicides', data=suicides_poor_rich, hue='country', ci=None)

plt.legend(labels=['BOT', 'TOP'])



plt.title('Comparison between top and bottom economies')

plt.xlabel('Year')

plt.ylabel('Suicides per 100k pop')

plt.show()