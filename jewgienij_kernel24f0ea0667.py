import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
df15 = pd.read_csv('/kaggle/input/world-happiness/2015.csv')
df16 = pd.read_csv('/kaggle/input/world-happiness/2016.csv')
df17 = pd.read_csv('/kaggle/input/world-happiness/2017.csv')
df18 = pd.read_csv('/kaggle/input/world-happiness/2018.csv')
df19 = pd.read_csv('/kaggle/input/world-happiness/2019.csv')
#df20 = pd.read_csv('/kaggle/input/world-happiness/2020.csv')
df15.shape,  df16.shape, df17.shape, df18.shape, df19.shape #, df20.shape
df15.isna().sum(), df16.isna().sum(), df17.isna().sum(), df18.isna().sum(), df19.isna().sum()#, df20.isna().sum()
df15.head(10)
df16.head(10)
df17.head(10)
df18.head(10)
df19.head(10)
#df20.head(10)
top10_15 = set([x for x in df15['Country'][:10]])
top10_16 = set([x for x in df16['Country'][:10]])
top10_17 = set([x for x in df17['Country'][:10]])
top10_18 = set([x for x in df18['Country or region'][:10]])
top10_19 = set([x for x in df19['Country or region'][:10]])
#top10_20 = set([x for x in df20['Country name'][:10]])
constant_countries=list(top10_15 & top10_16 & top10_17 & top10_18 & top10_19)
variable_countries=list((top10_15 | top10_16 | top10_17 | top10_18 | top10_19)-(top10_15 & top10_16 & top10_17 & top10_18 & top10_19))

print('Permanent inhabitants :) : \n' + str(constant_countries) + '\n')
print('Climbed to Olympus :) : \n' + str(variable_countries))
df15_rank = df15.drop(df15.columns[[1, 3, 4, 5, 6, 7, 8, 9, 10, 11]], axis='columns').iloc[:10, :][['Happiness Rank', 'Country']]
df16_rank = df16.drop(df16.columns[[1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]], axis='columns').iloc[:10, :][['Happiness Rank', 'Country']]
df17_rank = df17.drop(df17.columns[[2, 3, 4, 5, 6, 7, 8]], axis='columns').iloc[:10, :][['Happiness.Rank', 'Country']]
df18_rank = df18.drop(df18.columns[[2, 3, 4, 5, 6, 7, 8]], axis='columns').iloc[:10, :]
df19_rank = df19.drop(df19.columns[[2, 3, 4, 5, 6, 7, 8]], axis='columns').iloc[:10, :]

'''
df20_rank = df20.drop(df20.columns[[range(1, 20)]], axis='columns').iloc[:10, :]

rank = []
for i in range(1, 11):
    rank.append(i)

df20_rank['Rank'] = rank
df20_rank = df20_rank[['Rank', 'Country name']]
'''
df15_rank = df15_rank.rename(columns={'Happiness Rank': 'Rank'})
df16_rank = df16_rank.rename(columns={'Happiness Rank': 'Rank'})
df17_rank = df17_rank.rename(columns={'Happiness.Rank': 'Rank'})
df18_rank = df18_rank.rename(columns={'Overall rank': 'Rank', 'Country or region':'Country'})
df19_rank = df19_rank.rename(columns={'Overall rank': 'Rank', 'Country or region':'Country'})
#df20_rank = df20_rank.rename(columns={'Country name': 'Country'})
whole_df = pd.concat([df15_rank, df16_rank, df17_rank, df18_rank, df19_rank], ignore_index=True)
whole_df.pivot_table(values='Rank',index=['Country']).sort_values('Rank').drop(['Australia', 'Austria'])
denmark = whole_df[whole_df['Country'] == 'Denmark']['Rank'].values
finland = whole_df[whole_df['Country'] == 'Finland']['Rank'].values
norway = whole_df[whole_df['Country'] == 'Norway']['Rank'].values
iceland = whole_df[whole_df['Country'] == 'Iceland']['Rank'].values
switzerland = whole_df[whole_df['Country'] == 'Switzerland']['Rank'].values
netherlands = whole_df[whole_df['Country'] == 'Netherlands']['Rank'].values
canada = whole_df[whole_df['Country'] == 'Canada']['Rank'].values
new_zealand = whole_df[whole_df['Country'] == 'New Zealand']['Rank'].values
sweden = whole_df[whole_df['Country'] == 'Sweden']['Rank'].values
x = [2015, 2016, 2017, 2018, 2019]
fig = plt.figure()

ax = fig.add_axes([0,0,1,1])

ax.plot(x, denmark, label="Denmark")
ax.plot(x, finland, label="Finland")
ax.plot(x, norway, label="Norway")
ax.plot(x, iceland, label="Iceland")
ax.plot(x, switzerland, label="Switzerland")
ax.plot(x, netherlands, label="Netherlands")
ax.plot(x, canada, label="Canada")
ax.plot(x, new_zealand, label="New Zealand")
ax.plot(x, sweden, label="Sweden")
plt.xlabel('Year')
plt.ylabel('Happiness Rank')
plt.title('Ranking of country (lower - better)')
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
df15_gdp = df15[['Country', 'Economy (GDP per Capita)']].rename(columns={'Economy (GDP per Capita)':'GDP per capita'})[:10]
df16_gdp = df16[['Country', 'Economy (GDP per Capita)']].rename(columns={'Economy (GDP per Capita)':'GDP per capita'})[:10]
df17_gdp = df17[['Country', 'Economy..GDP.per.Capita.']].rename(columns={'Economy..GDP.per.Capita.':'GDP per capita'})[:10]
df18_gdp = df18[['Country or region', 'GDP per capita']].rename(columns={'Country or region':'Country'})[:10]
df19_gdp = df19[['Country or region', 'GDP per capita']].rename(columns={'Country or region':'Country'})[:10]
#df20_gdp = df20[['Country name', 'Explained by: Social support']].rename(columns={'Country name':'Country','Explained by: Social support':'GDP per capita'})[:10]
#df20_gdp['GDP per capita'] = df20_gdp['GDP per capita'].apply(np.log10)
whole_df = pd.concat([df15_gdp, df16_gdp, df17_gdp, df18_gdp, df19_gdp], ignore_index=True)
denmark = whole_df[whole_df['Country'] == 'Denmark']['GDP per capita'].values
finland = whole_df[whole_df['Country'] == 'Finland']['GDP per capita'].values
norway = whole_df[whole_df['Country'] == 'Norway']['GDP per capita'].values
iceland = whole_df[whole_df['Country'] == 'Iceland']['GDP per capita'].values
switzerland = whole_df[whole_df['Country'] == 'Switzerland']['GDP per capita'].values
netherlands = whole_df[whole_df['Country'] == 'Netherlands']['GDP per capita'].values
canada = whole_df[whole_df['Country'] == 'Canada']['GDP per capita'].values
new_zealand = whole_df[whole_df['Country'] == 'New Zealand']['GDP per capita'].values
sweden = whole_df[whole_df['Country'] == 'Sweden']['GDP per capita'].values
fig = plt.figure()

ax = fig.add_axes([0,0,1,1])

ax.plot(x, denmark, label="Denmark")
ax.plot(x, finland, label="Finland")
ax.plot(x, norway, label="Norway")
ax.plot(x, iceland, label="Iceland")
ax.plot(x, switzerland, label="Switzerland")
ax.plot(x, netherlands, label="Netherlands")
ax.plot(x, canada, label="Canada")
ax.plot(x, new_zealand, label="New Zealand")
ax.plot(x, sweden, label="Sweden")
plt.xlabel('Year')
plt.ylabel('GDP per capita')
plt.title('Wealth of country (higher - better)')
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
df15.tail(10)
df16.tail(10)
df17.tail(10)
df18.tail(10)
df19.tail(10)
#df20.tail(10)
worst10_15 = set([x for x in df15['Country'][-10:]])
worst10_16 = set([x for x in df16['Country'][-10:]])
worst10_17 = set([x for x in df17['Country'][-10:]])
worst10_18 = set([x for x in df18['Country or region'][-10:]])
worst10_19 = set([x for x in df19['Country or region'][-10:]])
#worst10_20 = set([x for x in df20['Country name'][-10:]])
constant_countries=list(worst10_15 & worst10_16 & worst10_17 & worst10_18 & worst10_19)
variable_countries=list((worst10_15 | worst10_16 | worst10_17 | worst10_18 | worst10_19)-(worst10_15 & worst10_16 & worst10_17 & worst10_18 & worst10_19))

print('Permanent bottom inhabitant :( : \n' + str(constant_countries) + '\n')
print('Survivors :) : \n' + str(variable_countries))
df15['Region'].value_counts()
df16['Region'].value_counts()
#df17['Region'].value_counts() отсутствует поле 'Region'
#df18['Region'].value_counts() отсутствует поле 'Region'
#df19['Region'].value_counts() отсутствует поле 'Region'
#df20['Regional indicator'].value_counts()
df15 = df15[['Happiness Score', 'Economy (GDP per Capita)', 'Family', 'Health (Life Expectancy)', 'Freedom', 'Trust (Government Corruption)', 'Generosity', 'Dystopia Residual']]
sns.heatmap(df15.corr(), annot=True, cmap='YlGnBu')
df16 = df16[['Happiness Score',
       'Economy (GDP per Capita)', 'Family', 'Health (Life Expectancy)',
       'Freedom', 'Trust (Government Corruption)', 'Generosity',
       'Dystopia Residual']]
sns.heatmap(df16.corr(), annot=True, cmap='YlGnBu')
df17 = df17[['Happiness.Score', 'Economy..GDP.per.Capita.', 'Family',
       'Health..Life.Expectancy.', 'Freedom', 'Generosity',
       'Trust..Government.Corruption.', 'Dystopia.Residual']]
sns.heatmap(df17.corr(), annot=True, cmap='YlGnBu')
df18 = df18[['Score', 'GDP per capita',
       'Social support', 'Healthy life expectancy',
       'Freedom to make life choices', 'Generosity',
       'Perceptions of corruption']]
sns.heatmap(df18.corr(), annot=True, cmap='YlGnBu')
df19 = df19[[ 'Score', 'GDP per capita',
       'Social support', 'Healthy life expectancy',
       'Freedom to make life choices', 'Generosity',
       'Perceptions of corruption']]
sns.heatmap(df19.corr(), annot=True, cmap='YlGnBu')
'''
df20 = df20[['Ladder score', 'Explained by: Log GDP per capita', 'Explained by: Social support',
       'Explained by: Healthy life expectancy',
       'Explained by: Freedom to make life choices',
       'Explained by: Generosity', 'Explained by: Perceptions of corruption',
       'Dystopia + residual']]
sns.heatmap(df20.corr(), annot=True, cmap='YlGnBu')
'''
sns.jointplot(x='Economy (GDP per Capita)', y = 'Happiness Score', data=df15, kind='reg')
sns.jointplot(x='Family', y = 'Happiness Score', data=df15, kind='reg')
sns.jointplot(x='Health (Life Expectancy)', y = 'Happiness Score', data=df15, kind='reg')