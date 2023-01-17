import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
apps = pd.read_csv('../input/google-play-store-apps/googleplaystore.csv')
reviews = pd.read_csv('../input/google-play-store-apps/googleplaystore_user_reviews.csv')
apps.head()
apps.info()
# Looking for apps, where 'Reviews' are represented with non-numerical value
apps[~apps['Reviews'].str.isnumeric()]
# Moving all the columns
for i in range(11, 0, -1):
    apps.iloc[10472,i+1] = apps.iloc[10472,i]
# Now let's find appropriate category for this app and genre, as this field is also not defined. To do that let's check list of possible options
print(apps['Category'].unique())
print('\n\n')
print(apps['Genres'].unique())
# As application name is 'Life Made WI-Fi Touchscreen Photo Frame' I assume, that correct Category is 'PHOTOGRAPHY' and 'Genres' should also be 'Photography'
apps.loc[10472,'Category'] = 'PHOTOGRAPHY'
apps.loc[10472,'Genres'] = 'Photography'
# Checking new values for the mentioned row
apps.iloc[10472]
# Looking for apps, where 'Price' is represented with non-decimal value
apps[~apps['Price'].str.replace('$','').str.replace('.','').str.isnumeric()]
# Checking values in Installs column
apps['Installs'].unique()
# Checking values in Installs column
apps['Size'].unique()
# Changing types of columns as described above.
apps['Reviews'] = apps['Reviews'].astype('int')
apps['Rating'] = apps['Rating'].astype('float')
apps['Price'] = apps['Price'].str.replace('$', '').astype('float')
apps['Installs'] = apps['Installs'].str.replace(',', '').str.replace('+', '').astype('int')
apps['Size'] = apps['Size'].str.replace('M', 'e+6').str.replace('k', 'e+3').str.replace('Varies with device', '0').astype('float')
plt.figure(figsize = (12,6))
bar = sns.barplot(apps[apps['Size'] != 0]['Category'], apps[apps['Size'] != 0]['Size'])
bar.set_xticklabels(bar.get_xticklabels(), rotation=90)
for i, row in apps[apps['Size'] == 0].iterrows():
    apps.loc[i, 'Size'] = apps.groupby(['Category'])['Size'].mean()[row['Category']]
apps['Last Updated'] = pd.to_datetime(apps['Last Updated'])
apps['Last Updated'].describe()
apps['App'].nunique()
sum(apps.duplicated())
# Checking duplicates in details
apps[apps.duplicated(keep = False)].sort_values('App').head(20)
apps.drop_duplicates(inplace = True)
# Checking categories
apps['Category'].unique()
apps['Category'] = apps['Category'].astype('category')
#Let's check the missing values in Type column
apps[apps['Type'].isnull()]
# As this app has price 0.0, let's put Type = Free
apps['Type'].fillna('Free', inplace = True)
# This column is good for encoding and updating to 'category' type. Let's replace Free Type with '0' and Paid - with '1'.
apps['Type'] = apps['Type'].str.replace('Free', '0').str.replace('Paid', '1').astype('category')
# Checking Content Rating for suspicious values
apps['Content Rating'].unique()
# Checking values again
apps['Genres'].unique()
# There are maximum 2 genres for one app, so I'm going to create 2 extra variables
Genre_1 = apps['Genres'].apply(lambda genre: genre.split(';')[0])
Genre_2 = apps['Genres'].apply(lambda genre: genre.split(';')[1] if len(genre.split(';')) > 1 else np.nan)
# Adding genres in dataset in 2 seprate column within specific positions
apps.insert(10, 'Genre 1', Genre_1)
apps.insert(11, 'Genre 2', Genre_2)
apps['Genre 2'].fillna('Not specified', inplace = True)
# Removing original Genres column and saving it in a separate variable
Genres = apps.pop('Genres')
# Checking missing values in Current Version
apps[apps['Current Ver'].isnull()]
# Which values are the most common for Current Version?
apps['Current Ver'].value_counts().head(20)
# Using 'Varies with device' for gaps in Current Version
apps['Current Ver'].fillna('Varies with device', inplace = True)
apps['Android Ver'].value_counts()
# Checking missing values in Android Version
apps[apps['Android Ver'].isnull()]
# Using 'Varies with device' for gaps in Android Version
apps['Android Ver'].fillna('Varies with device', inplace = True)
# Checking common versions in different years
apps.groupby(apps['Last Updated'].dt.year)['Android Ver'].value_counts().head(50)
# Implementing a replacement
for i, row in apps[apps['Android Ver'] == 'Varies with device'].iterrows():
    apps.loc[i, 'Android Ver'] = apps.groupby(apps['Last Updated'].dt.year)['Android Ver'].value_counts()[row['Last Updated'].year].idxmax()
apps[apps['Android Ver'] == '4.4W and up']
# Implementing replacement for '4.4W and up'
apps['Android Ver'] = apps['Android Ver'].str.replace('4.4W and up', '4.4 and up')
# Creating new column
apps['Min Android Ver'] = apps['Android Ver'].apply(lambda ver: ver.split(' ')[0])
# Removing original column and saving it in a separate variable
Android_Ver = apps.pop('Android Ver')
# Checking distribution of new column
apps['Min Android Ver'].value_counts()
apps.info()
apps.describe()
sns.set_style('whitegrid')
sns.distplot(apps['Rating'], bins = 20)
apps['Reviews'].value_counts()
sns.distplot(apps['Size'], bins = 30)
apps['Installs'].value_counts()
apps['Price'].value_counts()