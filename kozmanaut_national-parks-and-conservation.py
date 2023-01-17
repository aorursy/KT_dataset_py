from __future__ import division

import pandas as pd

import matplotlib.pyplot  as plt



# for prettier plots

plt.style.use('ggplot')

plt.rcParams['figure.figsize'] = (10, 9)
species = pd.read_csv('../input/species.csv', usecols=range(1,13))

parks = pd.read_csv('../input/parks.csv')



#change NaN to 'Not Threatened' in the Conservation Status column

species['Conservation Status'].fillna('Not Threatened', inplace=True)



# On inspection, the 'Conservation Category' contains Seasonality information - get rid of these rows

species = species[species['Conservation Status'].str.contains("Breeder|Resident|Migratory") == False]
# Calculate the number of each conservation category for each park

props = species[['Park Name', 'Conservation Status']].groupby(['Park Name', 'Conservation Status']).size()

# Convert to dataframe

props_df = props.to_frame().reset_index()

props_df.columns = ['Park Name', 'Conservation Status', 'Count']
cat_to_plot = ['Endangered', 'In Recovery', 'Species of Concern', 'Threatened']

for i in cat_to_plot:

	subset = props_df[props_df['Conservation Status'] == i]

	subset.sort_values(by='Count', ascending=False).plot(x='Park Name', y='Count', kind='bar', legend=None)

	plt.ylabel('Absolute number of species')

	plt.title('Conservation category: %s' % (i))

	plt.tight_layout()

plt.show()
# Count of each conservation category per acre for each NP - i.e. per acre of land, which park harbors the most e.g. endangered species

#1. Create a dictionary with 'Park' : 'Acres'

park_dict = dict(zip(parks['Park Name'], parks['Acres']))

#2. Create a function that divides each row's conservation category count by the park's area

def divide_count(row):

	return row['Count']/(park_dict[row['Park Name']])

#3. Create a new column with the count per acre measure

props_df['CountPerAcre'] = props_df.apply(divide_count, axis=1)

#4. Plot the results

for i in cat_to_plot:

	subset = props_df[props_df['Conservation Status'] == i]

	subset.sort_values(by='CountPerAcre', ascending=False).plot(x='Park Name', y='CountPerAcre', kind='bar', legend=None)

	plt.ylabel('Number of species per acre of park land')

	plt.title('Conservation category: %s' % (i))

	plt.tight_layout()

plt.show()
# Count the proportion of each conservation category for each park - i.e. find the park with the highest proportion of endangered species

#1. Sum up all conservation classes per park

park_sums = props_df.groupby(['Park Name']).agg('sum').reset_index()

# 2. Create a dictionary with 'Park' : 'Total count'

park_sums_dict = dict(zip(park_sums['Park Name'], park_sums['Count']))

# Create function that divides each conservation category count by the total count

def divide_total(row):

	return row['Count']/(park_sums_dict[row['Park Name']])

# 3. Create new column with proportional count of conservation category

props_df['ProportionalCount'] = props_df.apply(divide_total, axis=1)

# 4. Plot the results

for i in cat_to_plot:

	subset = props_df[props_df['Conservation Status'] == i]

	subset.sort_values(by='ProportionalCount', ascending=False).plot(x='Park Name', y='ProportionalCount', kind='bar', legend=None)

	plt.ylabel('Proportion of all species in the park')

	plt.title('Conservation category: %s' % (i))

	plt.tight_layout()

plt.show()