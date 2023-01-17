import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns



%matplotlib inline



df = pd.read_csv('../input/Netflix Shows.csv', encoding='cp437')

df.head()
df.info()
df.describe()
df['title'].value_counts().head()
df.drop_duplicates(inplace=True)

df['title'].value_counts().head()
multiple_titles = df['title'].value_counts().iloc[0:4].keys()

df[df['title'].isin(multiple_titles)]
df.info()
sns.jointplot(data=df, y='user rating score', x='release year')

plt.xlim(1939, 2018)
order = np.sort(df['rating'].unique())

plt.figure(figsize=(10, 6))

sns.boxplot(data=df, y='user rating score', x='rating',order=order)

plt.title('User scores by Rating')
def age_group(rating):

    little = ['G','TV-Y','TV-G']

    older = ['PG', 'TV-Y7', 'TV-Y7-FV', 'TV-PG']

    teens = ['PG-13', 'TV-14']

    adult = ['R', 'NC-17', 'NR', 'UR', 'TV-MA']

    

    if rating in little:

        return 'Little Kids'

    elif rating in older:

        return 'Older Kids'

    elif rating in teens:

        return 'Teens'

    elif rating in adult:

        return 'Adults'

    else:

        return 'Missing'

    

df['age_group'] = df['rating'].apply(age_group)

df.head()
#Check for missing ratings

df['age_group'].unique()
order = ['Little Kids','Older Kids','Teens','Adults']

sns.countplot(df['age_group'],order=order)
plt.figure(figsize=(10, 6))

sns.boxplot(data=df, y='user rating score', x='age_group', order=order)

plt.title('User scores by Age Group')
print('Highest and lowest user rated shows by age group:')



for group in order:

    print('\n' + group+':')

    print('Highest')

    print(df[df['age_group']==group].sort_values('user rating score',ascending=False)[['title','user rating score']].head(1))

    print('\nLowest')

    print(df[df['age_group']==group].sort_values('user rating score')[['title','user rating score']].head(1))