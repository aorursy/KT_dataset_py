import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
apps = pd.read_csv('../input/google-play-store-apps/googleplaystore.csv')
sns.set_style('whitegrid')
# Values in row 10472 are displaced. Fixing that

for i in range(11, 0, -1):

    apps.iloc[10472,i+1] = apps.iloc[10472,i]



apps.loc[10472,'Category'] = 'PHOTOGRAPHY'

apps.loc[10472,'Genres'] = 'Photography'
# Removing redundant duplicated rows

apps.drop_duplicates(inplace = True)
# Changing types of columns.

apps['Reviews'] = apps['Reviews'].astype('int')

apps['Rating'] = apps['Rating'].astype('float')

apps['Price'] = apps['Price'].str.replace('$', '').astype('float')

apps['Installs'] = apps['Installs'].str.replace(',', '').str.replace('+', '').astype('int')

apps['Size'] = apps['Size'].str.replace('M', 'e+6').str.replace('k', 'e+3').str.replace('Varies with device', '0').astype('float')

apps['Last Updated'] = pd.to_datetime(apps['Last Updated'])

apps['Category'] = apps['Category'].astype('category')

# Filling missing values and changing column type

apps['Type'] = apps['Type'].fillna('Free').str.replace('Free', '0').str.replace('Paid', '1').astype('category')

# Filing missing values in Size column with mean sizes in every category 

for i, row in apps[apps['Size'] == 0].iterrows():

    apps.loc[i, 'Size'] = apps.groupby(['Category'])['Size'].mean()[row['Category']]
# Dividing genres in 2 seprate columns within specific positions

apps.insert(10, 'Genre 1', apps['Genres'].apply(lambda genre: genre.split(';')[0]))

apps.insert(11, 'Genre 2', apps['Genres'].apply(lambda genre: genre.split(';')[1] if len(genre.split(';')) > 1 else np.nan))

apps['Genre 2'].fillna('Not specified', inplace = True)

# Removing original Genres column and saving it in a separate variable

Genres = apps.pop('Genres')
# Filling missing values in versions columns

apps['Current Ver'].fillna('Varies with device', inplace = True)

apps['Android Ver'].fillna('Varies with device', inplace = True)



# Implementing a replacement for Android versions due to year of last update

for i, row in apps[apps['Android Ver'] == 'Varies with device'].iterrows():

    apps.loc[i, 'Android Ver'] = apps.groupby(apps['Last Updated'].dt.year)['Android Ver'].value_counts()[row['Last Updated'].year].idxmax()



# Replacing Wearable version with general one

apps['Android Ver'] = apps['Android Ver'].str.replace('4.4W and up', '4.4 and up')



# Creating new column with only lower versioning limitation

apps['Min Android Ver'] = apps['Android Ver'].apply(lambda ver: ver.split(' ')[0])



# Removing original Android ver column and saving it in a separate variable

Android_Ver = apps.pop('Android Ver')
apps.head()
apps.info()
apps.describe()
apps.corr()
sns.distplot(apps['Rating'], bins = 15)
sns.countplot(apps['Type'])
plt.figure(figsize = (12,6))

sns.countplot(apps['Content Rating'])
# Category with the biggest number of applications offered

apps.groupby(['Category'])['App'].count().sort_values().tail(1)
plt.figure(figsize = (12,6))

bar = sns.barplot(apps.groupby(['Category'])['App'].count().index, apps.groupby(['Category'])['App'].count().values)

bar.set_xticklabels(bar.get_xticklabels(), rotation=90)

bar.set_title('Amount of applications per category', size = 20)

bar.set_xlabel('Categories')

bar.set_ylabel('Number of games')
# Category with the biggest number of installs

apps.groupby(['Category'])['Installs'].sum().sort_values().tail(1)
apps.groupby(['Category'])['Installs'].sum()/1000000
plt.figure(figsize = (12,6))

bar = sns.barplot((apps.groupby(['Category'])['Installs'].sum()/1000000000).index, (apps.groupby(['Category'])['Installs'].sum()/1000000000).values)

bar.set_xticklabels(bar.get_xticklabels(), rotation=90)

bar.set_title('Number of installs per category', size = 20)

bar.set_xlabel('Categories')

bar.set_ylabel('Number of installs (in Billions)')
# Looking for biggest size value

apps['Size'].max()
sns.distplot(apps['Size'], bins = 30)
# Looking for corresponding apps

apps[apps['Size'] == apps['Size'].max()]
# Setting reference date

ref = apps['Last Updated'].max()

ref
# Calculating differences between reference date and update dates

apps['Last Updated'].apply(lambda date: ref - date).sort_values(ascending = False)
# Application that hasn't been updated for 3001 days

apps.loc[7479]
# Looking for applications with biggest number of installs

apps[apps['Installs'] == apps['Installs'].max()]
# Looking for applications with biggest number of reviews

apps[apps['Reviews'] == apps['Reviews'].max()]