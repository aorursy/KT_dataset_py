import pandas as pd 

import scipy.stats



# load the data

df = pd.read_csv('../input/DigiDB_digimonlist.csv')

# first with stage column

stage_count = df['Stage'].value_counts()

print(stage_count)
# second with attribute column

attribute_count = df['Attribute'].value_counts()

print(attribute_count)
# Now we can make the chi-square test for each one

print(scipy.stats.chisquare(stage_count))

print(scipy.stats.chisquare(attribute_count))
stage_vs_attribute = pd.crosstab(df['Stage'],df['Attribute'])
stage_vs_attribute
scipy.stats.chi2_contingency(stage_vs_attribute)