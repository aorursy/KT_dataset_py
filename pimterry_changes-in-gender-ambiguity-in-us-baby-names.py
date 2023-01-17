import math

import pandas as pd



names_data = pd.read_csv("../input/NationalNames.csv")



frequent_names = names_data[names_data['Count'] > 20]

indexed_names = frequent_names.set_index(['Year', 'Name'])['Count']



# Number between 0 and 1 representing ambiguity, from certain to totally ambiguous

# 0 = all the same gender, 1 = exactly 50% of each gender. Assumes only two options.

def ambiguity_measure(grouped_frame):

        return (2 * (1 - (grouped_frame.max() / grouped_frame.sum())))



# Various useful formattings of gender ambiguity data:

ambiguity_data = ambiguity_measure(indexed_names.groupby(level=['Year', 'Name'])).rename("Ambiguity")

yearly_ambiguity = ambiguity_data.groupby(level='Year')



ambiguity_with_counts = ambiguity_data.to_frame().join(indexed_names.groupby(level=['Year', 'Name']).sum())

data_vs_years = ambiguity_with_counts.unstack(level='Year')

data_vs_years["Total"] = data_vs_years['Count'].sum(axis=1)
yearly_ambiguity.idxmax().apply(lambda x: x[1]).to_frame() 
ambiguous_names = data_vs_years[(data_vs_years['Ambiguity'] > 0.1).any(axis=1)]

popular_ambiguous_names = ambiguous_names.sort_values(by='Total', ascending=False).head(7).drop("Total", axis=1)

popular_ambiguous_names['Ambiguity'].transpose().plot(figsize=(10, 10))
# Gender ambiguity by name (not *person*! See the next chart)

yearly_ambiguity.mean().transpose().plot(figsize=(10, 10))
# = SUM(probability of name in given year * ambiguity of name)

total_people_per_year = ambiguity_with_counts['Count'].groupby(level='Year').sum()

ambiguity_by_year = ambiguity_with_counts.unstack('Name')

ambiguity_by_year["total_people"] = total_people_per_year

weighted_ambiguity = ambiguity_by_year.apply(lambda x : x['Ambiguity'] * (x['Count']/x['total_people'][0]), axis=1)

weighted_ambiguity.sum(axis=1).plot(figsize=(10, 10))