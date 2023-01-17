import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import warnings

warnings.filterwarnings('ignore')



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df = pd.read_csv('../input/120-years-of-olympic-history-athletes-and-results/athlete_events.csv')

noc = pd.read_csv('../input/120-years-of-olympic-history-athletes-and-results/noc_regions.csv')

original = df.copy()
df = pd.merge(left = df, right = noc, on = 'NOC')
print(df.shape)
df.info()
df.describe()
df.drop_duplicates(inplace = True) 
df.drop(columns = ['ID', 'notes'], inplace = True)
num_cols = df.select_dtypes(['int64', 'float64']).columns.tolist()

obj_cols = df.select_dtypes('O').columns.tolist()
print('No. of Unique Values')

for i in obj_cols:

    print(i, ':', df[i].nunique())
df.sample(10)
def missing(df):

    total = df.isnull().sum().sort_values(ascending = False)

    percent = round(df.isnull().sum().sort_values(ascending = False) / len(df) * 100, 2)

    return pd.concat([total, percent], axis = 1, keys = ['Total', '% Missing'])
missing(df)
df.dropna(subset = ['Weight', 'Height', 'Age'], inplace = True)
bins = [0, 29, 45, 60, 100]

df['Age Group'] = pd.cut(df['Age'], bins = bins)
df['bmi'] = df['Weight'] / ((df['Height'] / 100) ** 2)
df['bmi group'] = np.where(df['bmi'] <= 18.5, 'Underweight',

                          np.where(df['bmi'] < 25, 'Normal',

                                  np.where(df['bmi'] < 30, 'Overweight',

                                          np.where(df['bmi'] >= 30, 'Obese', 'NA'))))
print("Number of Olympic Seasons Held :", df['Year'].nunique())

print("Number of Countries Participated :", df['NOC'].nunique())

print("Number of Players Participated :", df['Name'].nunique())

print("Number of Sports Conducted :", df['Sport'].nunique())

print("Number of Medals Won :\n", df['Medal'].dropna().value_counts())
medals_won = df.dropna(subset = ['Medal'])
plt.rcParams['figure.figsize'] = (8,6)



medals_won['region'].value_counts().nlargest(5).plot(kind = 'bar', linewidth = 1, facecolor = 'seagreen', edgecolor = 'k')

plt.title('Top 5 Countries with most medals')

plt.xlabel('Country')

plt.ylabel('# Medals')

plt.show()
df.groupby('region')['Name'].count().nlargest(5).plot(kind = 'bar', linewidth = 1, facecolor = 'seagreen', edgecolor = 'k')

plt.title('Top 5 Countries which sent most players')

plt.xlabel('Country')

plt.ylabel('# Players')

plt.show()
top5countries_medals = medals_won.loc[medals_won['region'].isin(['USA', 'UK', 'Russia', 'France', 'Germany']), :]
plt.rcParams['figure.figsize'] = (12,6)



pd.pivot_table(index = 'Year', columns = 'region', values = 'Medal', aggfunc = 'count',data = top5countries_medals).iloc[-5:, :].plot()

plt.title('Performance of Top 5 Countries Over Last 5 Seasons')

plt.xlabel('Country')

plt.ylabel('#Medals')

plt.show()
mf_ratio = pd.pivot_table(index = 'region', columns = 'Sex', values = 'Name', data = df, aggfunc = 'nunique')

mf_ratio.head()
mf_ratio['MRatio'] = ((mf_ratio['M'] / mf_ratio['M'])  * 100).astype(int)

mf_ratio['FRatio'] = ((mf_ratio['F'] / mf_ratio['M'])  * 100).astype(int)

mf_ratio['Overall'] = round(mf_ratio['MRatio'] / mf_ratio['FRatio'], 2)
mf_ratio[(mf_ratio['Overall'] > 0.90) & (mf_ratio['Overall'] < 1.1)]
mf_ratio.loc[((mf_ratio['Overall'] > 0.90) & (mf_ratio['Overall'] < 1.1) & (mf_ratio['M'] + mf_ratio['F'] > 100))].sort_values(by = 'Overall', ascending = False)
recent_olympic = df.loc[df['Year'] == 2016, :]
recent_olympic['Sport'].nunique()
country_by_sport = pd.pivot_table(index = 'region', values = 'Sport', aggfunc = 'nunique', data = recent_olympic)
print(country_by_sport[country_by_sport['Sport'] == 34].index)
plt.rcParams['figure.figsize'] = (8,6)



df['City'].value_counts().nlargest(5).plot(kind = 'bar', linewidth = 1, facecolor = 'seagreen', edgecolor = 'k')

plt.title('Top 5 Cities Hosted Most Matches')

plt.xlabel('City')

plt.ylabel('# Matches')

plt.show()
last_olympic = df.loc[df['Year'] == 2012, :]
last_olympic['Sport'].nunique()
recent_olympic['Sport'].nunique()
last = set(last_olympic['Sport'])

recent = set(recent_olympic['Sport'])
print('Games Introduced in the recent Summer Olympics : ', recent.difference(last))
first_olympic = df.loc[df['Year'] == 1896, :]
first = set(first_olympic['Sport'])

recent = set(recent_olympic['Sport'])
print('Sports Played Since the First Summer Olympic Season', '\n')

print(recent.intersection(first))
plt.rcParams['figure.figsize'] = (8,6)



medals_won['Name'].value_counts().nlargest(5).plot(kind = 'bar', linewidth = 1, facecolor = 'seagreen', edgecolor = 'k')

plt.title('Top 5 Players with most medals')

plt.xlabel('Players')

plt.ylabel('# Medals')

plt.show()
medals_by_age = pd.DataFrame()
medals_by_age['Total Players'] = df['Age Group'].value_counts()

medals_by_age['Players Won Medals'] = medals_won['Age Group'].value_counts()

medals_by_age['Percent'] = round(medals_by_age['Players Won Medals'] / medals_by_age['Total Players'] * 100, 2)
medals_by_age.sort_index()
medals_by_bmi = pd.DataFrame()
medals_by_bmi['Total Players'] = df['bmi group'].value_counts()

medals_by_bmi['Players Won Medals'] = medals_won['bmi group'].value_counts()

medals_by_bmi['Percent'] = round(medals_by_bmi['Players Won Medals'] / medals_by_bmi['Total Players'] * 100, 2)
medals_by_bmi