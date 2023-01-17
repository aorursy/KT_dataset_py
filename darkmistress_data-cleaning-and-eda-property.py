import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns
prop = pd.read_csv('../input/property/property data.csv')

print(prop.head(10))
# select numeric columns

prop_numeric = prop.select_dtypes(include=[np.number])

numeric_cols = prop_numeric.columns.values

print(numeric_cols)
# select non numeric columns

prop_non_num = prop.select_dtypes(exclude=[np.number])

non_num_cols = prop_non_num.columns.values

print(non_num_cols)
#Let's find the missing values in the data

cols = prop.columns[:10] # first 10 columns

colours = ['#EE82EE', '#ffff00'] # specify the colours - yellow is missing. violin is not missing.

sns.heatmap(prop[cols].isnull(), cmap=sns.color_palette(colours))
# % of missing.

for col in prop.columns:

    pct_missing = np.mean(prop[col].isnull())

    print('{} - {}%'.format(col, round(pct_missing*100)))
#ST_NUM and NUM_BEDROOM has more null values and we can't drop the entire column

#and we can't drop rows either, so we replace them



# categorical

prop['OWN_OCCUPIED'] = prop['OWN_OCCUPIED'].fillna('Y')

prop['NUM_BEDROOMS'] = prop['NUM_BEDROOMS'].fillna('2')

prop['NUM_BATH'] = prop['NUM_BATH'].fillna('1')

prop['SQ_FT'] = prop['SQ_FT'].fillna('800')







# numeric

prop['PID'] = prop['PID'].fillna(100005000.0)

prop['ST_NUM'] = prop['ST_NUM'].fillna(200.0)
#Now replace the values which are not null values



prop['NUM_BATH'] = prop['NUM_BATH'].replace('HURLEY','1')

prop['OWN_OCCUPIED'] = prop['OWN_OCCUPIED'].replace('12', 'N')

prop['NUM_BEDROOMS'] = prop['NUM_BEDROOMS'].replace('na', '2')

prop['SQ_FT'] = prop['SQ_FT'].replace('--', '800')

prop['ST_NAME'] = prop['ST_NAME'].replace('PUTNAM','WASHINGTON')
prop['OWN_OCCUPIED'] = prop['OWN_OCCUPIED'].replace('Y', '1')

prop['OWN_OCCUPIED'] = prop['OWN_OCCUPIED'].replace('N', '0')

prop.head(20)
print(prop.dtypes)
#Change the datatypes of NUM_BEDROOMS, NUM_BATH, SQ_FT to float





prop['NUM_BATH'] = prop['NUM_BATH'].astype(float)

prop['NUM_BEDROOMS'] = prop['NUM_BEDROOMS'].astype(float)

prop['SQ_FT'] = prop['SQ_FT'].astype(float)
print(prop.dtypes)
#Finally we have our clean data

#Lets check for missing values

for col in prop.columns:

    pct_missing = np.mean(prop[col].isnull())

    print('{} - {}%'.format(col, round(pct_missing*100)))
#We don't have any missing value or unnecessary data

prop
sns.barplot(x='NUM_BEDROOMS', y='SQ_FT', data=prop, hue='NUM_BATH')

plt.title('Num of bedroom and bathroom per sqft')
from matplotlib import pyplot as plt 

  

data = prop['OWN_OCCUPIED'].value_counts()

fig = plt.figure(figsize =(10, 8)) 

label = ['Yes', 'No']

colors = ( "cyan","indigo")

plt.pie(data,autopct='%1.1f%%', labels=label, shadow=True, colors=colors)

plt.show()
plt.hist(prop['ST_NAME'])

plt.xlabel('Street Name')

plt.ylabel('Count')

plt.title('Street distribution.')
sns.pairplot(prop,diag_kind="hist", hue='ST_NAME')