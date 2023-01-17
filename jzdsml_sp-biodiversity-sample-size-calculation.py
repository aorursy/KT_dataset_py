from matplotlib import pyplot as plt
import pandas as pd
species = pd.read_csv("../input/species_info.csv")
species.head(5)
species.scientific_name.nunique()
species.category.unique()
species.category.value_counts()
species.conservation_status.unique()
species.conservation_status.value_counts()
species.groupby('conservation_status').scientific_name.count().reset_index()
species.conservation_status.fillna('No Intervention', inplace=True)
species.head(5)
species.groupby('conservation_status').common_names.count().reset_index()
protection_counts = species.groupby('conservation_status')\
    .scientific_name.count().reset_index()\
    .sort_values(by='scientific_name', ascending = False)
protection_counts
plt.figure(figsize=(12,7))
ax = plt.subplot(1,1,1)
plt.bar(range(len(protection_counts)),protection_counts.scientific_name)
ax.set_xticks(range(len(protection_counts)))
ax.set_xticklabels(protection_counts.conservation_status)
plt.ylabel('Number of Species')
plt.show()

plt.figure(figsize=(20,20))
plt.pie(protection_counts.scientific_name, explode=(0, 0, 1, 0, 1), autopct='%1.1f%%',
        shadow=False, startangle=0)
plt.legend(protection_counts.conservation_status)
species['is_protected'] = species.conservation_status.apply(lambda x: True if x!= 'No Intervention' else False)
species.head(5)
#use nunique, not count
category_counts = species.groupby(['category', 'is_protected']).scientific_name.nunique().reset_index()
category_counts.head(10)
category_pivot = category_counts.pivot(index='category',columns='is_protected',values='scientific_name').reset_index()
category_pivot.head(10)
category_pivot.columns = ['category', 'not_protected', 'protected']
category_pivot['percent_protected'] = category_pivot['protected']/(category_pivot['protected'] + category_pivot['not_protected'])
category_pivot
contingency = [[30, 146],
               [75, 413]]
from scipy.stats import chi2_contingency
chi2_contingency(contingency) #pval = 0.688
contingency = [[5, 73],
               [30, 146]]

chi2_contingency(contingency) #pval = 0.038
observations = pd.read_csv('../input/observations.csv')
observations.head()
# Does "Sheep" occur in this string?
str1 = 'This string contains Sheep'
'Sheep' in str1
'sheep' in str1 #case sentitive
# Does "Sheep" occur in this string?
str2 = 'This string contains Cows'
'Sheep' in str2
species['is_sheep'] = species.common_names.apply(lambda x: True if 'Sheep' in x else False)
species[species.is_sheep==True]
sheep_species = species[(species.is_sheep==True) & (species.category=='Mammal')]
sheep_observations = sheep_species.merge(observations, on=['scientific_name'], how='left')
sheep_observations
obs_by_park = sheep_observations.groupby('park_name').observations.sum().reset_index()
obs_by_park
fig = plt.figure(figsize=(16,4))
ax = plt.subplot(1,1,1)
plt.bar(range(len(obs_by_park)), obs_by_park.observations)
ax.set_xticks(range(len(obs_by_park)))
ax.set_xticklabels(obs_by_park.park_name)
plt.ylabel('Number of Observations')
plt.title('Observations of Sheep per Week')
plt.show()
baseline = 0.15
mde = (0.05)/0.15 #33%
mde
sample_size = 870 # from calculator
# Note: This could be 890 if you used 33% for the "Minimum Detectable Effect" instead of 33.33%.  That's fine.
bryce_week = 250 # from obs_by_park
yellowstone_week = 507
print(sample_size/bryce_week)
print(sample_size/yellowstone_week)
