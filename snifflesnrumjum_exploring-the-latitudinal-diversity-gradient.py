import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
indata_spp = pd.read_csv('../input/species.csv', sep=',', header=0)
indata_parks = pd.read_csv('../input/parks.csv', sep=',', header=0)
indata_parks.count()
indata_spp.count()
print(indata_spp['Unnamed: 13'].dropna())
trsh = indata_spp.pop('Unnamed: 13')
indata_spp.head()
indata_parks.head()
species_grouped = indata_spp.groupby('Category')
species_grouped.groups
target_group = species_grouped.get_group('Insect')
target_group.shape
target_group.keys()
park_div = target_group.groupby('Park Name')
park_div.count().shape
plt.bar(range(park_div.count().shape[0]), 
        park_div['Scientific Name'].count().divide(target_group.drop_duplicates( \
                                            subset='Scientific Name').count()[0]))
plt.show()
sns.regplot(indata_parks[indata_parks['Park Name'].isin(park_div.groups.keys())].Latitude, \
            park_div['Scientific Name'].count().divide(target_group.drop_duplicates( \
                                                subset='Scientific Name').count()[0]))
plt.ylabel('Diversity')
plt.show()
for cat in ['Algae', 'Fungi', 'Mammal', 'Reptile', 'Vascular Plant']:
    plt.figure()
    target_group = species_grouped.get_group(cat)
    park_div = target_group.groupby('Park Name')
    sns.regplot(indata_parks[indata_parks['Park Name'].isin(park_div.groups.keys())].Latitude, \
            park_div['Scientific Name'].count().divide(target_group.drop_duplicates( \
                                                subset='Scientific Name').count()[0]))
    plt.xlim(15, 65)
    plt.title(cat, size=25)
    plt.ylabel('Diversity')
plt.show()
for cat in species_grouped.groups:
    plt.figure()
    group = species_grouped.get_group(cat)
    park_div = group.groupby('Park Name')
    sns.regplot(indata_parks[indata_parks['Park Name'].isin(park_div.groups.keys())].Latitude, \
                park_div['Scientific Name'].count().values / (indata_parks[indata_parks['Park Name'] \
                                                                    .isin(park_div.groups.keys())].Acres))
              
    plt.xlim(15, 65)
    plt.title(cat, size=25)
    plt.ylabel('Diversity (norm)')
plt.show()
