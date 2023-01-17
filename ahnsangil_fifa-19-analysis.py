import numpy as np

import pandas as pd

from pandas import Series, DataFrame

import matplotlib.pyplot as plt
df = pd.read_csv('../input/data.csv', index_col='Name')
df.describe()
df.isna().sum().sort_values(ascending=False)
df.drop('Loaned From', axis=1, inplace=True)
df.fillna(df.mean(), inplace=True)
df.isna().sum().sort_values(ascending=False)
df.head(3)
def convert_money(money):

    if money[-1] == 'M':

        return int(float(money[:-1]) * 10 ** 6)

    elif money[-1] == 'K':

        return int(float(money[:-1]) * 10 ** 3)

    else:

        return int(money)
df.drop('Unnamed: 0', axis=1, inplace=True)

df['Release Clause'].fillna('€0', inplace=True)

modified = ['Value', 'Wage', 'Release Clause']

for m in modified:

    df.loc[:, m] = df[m].str.replace('€', '').map(convert_money)

columns = {m: m + '(€)' for m in modified}

df.rename(columns=columns, inplace=True)
def convert_height(height):

    feet, inch = height.split("'") # 1 feet = 30.48 cm, 1 inch = 2.54 cm

    return int(feet) * 30.48 + int(inch) * 2.54
df['Height'].fillna("0'0", inplace=True)

df.loc[:, 'Height'] = df['Height'].map(convert_height)

df.rename(columns={'Height': 'Height(cm)'}, inplace=True)
df.loc[:, 'Weight'] = df['Weight'].str.replace('lbs', '')

df['Weight'].fillna(0, inplace=True)

df.loc[:, 'Weight'] = df['Weight'].map(int) * 0.45 # 1 lbs = 0.453592 kg

df.rename(columns={'Weight': 'Weight(kg)'}, inplace=True)
df.fillna('Unassigned', inplace=True)
df.isna().sum().sort_values(ascending=False)
df['Nationality'].value_counts()[:10].plot(kind='bar')

plt.show()
df.sort_values(by='Value(€)', ascending=False)[:10]
# Top 10 countries for Value(€)

df.groupby('Nationality')['Value(€)'].sum().sort_values(ascending=False)[:10].plot(kind='bar')

plt.show()
# Top 10 Club for Value(€)

df.groupby('Club')['Value(€)'].sum().sort_values(ascending=False)[:10].plot(kind='bar')

plt.show()
# Top 10 countries for overall

# Result does not match our common sense on soccer.

# In this case, I need to filter out some countries whose counts is smaller than the mean counts.

df.groupby('Nationality')['Overall'].mean().sort_values(ascending=False)[:10].plot(kind='bar')

plt.show()
# Get top 20 countries

nationality_top20 = df['Nationality'].value_counts()[:20].index
# Again, top 10 countries for overall from top 20 countries index

df[df['Nationality'].isin(nationality_top20)].groupby('Nationality')['Overall'].mean().sort_values(ascending=False)[:10].plot(kind='bar')

plt.show()
korea = df[df['Nationality'] == 'Korea Republic']
korea_clubs = DataFrame(korea['Club'].value_counts().index, columns=['Club'])
korea_clubs['playwhere'] = np.where(korea['Club'].value_counts() >= 10, 'domestic', 'overseas')
korea_clubs.set_index('Club', inplace=True)
kclub_dict = korea_clubs['playwhere'].to_dict()
korea['playwhere'] = korea['Club'].map(kclub_dict)
# define font properties fot plot

titledict = {'fontsize': 30,

             'color': 'black',

             'weight': 'bold'}

xlabeldict = {'fontsize': 20,

              'color': 'green'}

ylabeldict = {'fontsize': 20,

              'color': 'red'}
ax = korea['playwhere'].value_counts().plot(kind='bar', figsize=(8, 8))

ax.set_xticklabels(['domestic', 'overseas'], fontsize=20, rotation=0)

ax.set_yticklabels([i * 50 for i in range(7)], fontsize=15)

ax.set_xlabel('Play where', fontdict=xlabeldict)

ax.set_ylabel('Number of Korean players', fontdict=ylabeldict)

ax.set_title("Korean players' ratio who play overseas", fontdict=titledict)

plt.show()
interests = ['Value(€)', 'Age', 'International Reputation', 'Height(cm)', 'Weight(kg)']

countries = ['Japan', 'Iran', 'United States', 'Germany', 'England', 'Spain',

             'France', 'Mexico', 'Italy', 'Brazil']

result = {}

for country in countries:

    temp = []

    df_country = df[df['Nationality'] == country]

    for interest in interests:

        temp.append(round(korea[interest].mean() / df_country[interest].mean(), 3))

    result[country] = temp

ax = DataFrame(result, index=interests).plot(kind='bar', figsize=(20, 20))

ax.plot([-2, 10], [1, 1], linewidth=3, linestyle='--', color='black')

ax.set_title("Korea's Relative Capability against several countries",

             fontdict=titledict)

ax.legend(loc='upper left', bbox_to_anchor=(1, 1), borderaxespad=0, fontsize=20)

ax.set_xticklabels(interests, rotation=0, fontsize=20)

ax.set_yticklabels([round(i * 0.2, 1) for i in range(6)], fontsize=20)

ax.set_xlabel('Abilities', fontdict=xlabeldict)

ax.set_ylabel('Relativity', fontdict=ylabeldict)

plt.show()
hson_overall = korea.loc['H. Son', 'Overall']

interests = ['Age', 'Value(€)', 'International Reputation', 'Height(cm)', 'Weight(kg)']

result = {}

for interest in interests:

    comparision = korea.loc['H. Son', interest] / df[df['Overall'] == hson_overall][interest].mean()

    result[interest] = comparision

result = Series(result)

ax = result.plot(kind='bar', figsize=(10, 10))

ax.set_title('Compare H. Son and same overall players(H. Son / Other players)',

             fontdict=titledict)

ax.set_xlabel('Abilities', fontdict=xlabeldict)

ax.set_xticklabels(interests, rotation=0)

ax.set_ylabel('Relativity', fontdict=ylabeldict)

plt.show()