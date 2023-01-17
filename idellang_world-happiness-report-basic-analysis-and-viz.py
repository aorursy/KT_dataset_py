import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns



%matplotlib inline
data2015 = pd.read_csv('../input/world-happiness/2015.csv')

data2016 = pd.read_csv('../input/world-happiness/2016.csv')

data2017 = pd.read_csv('../input/world-happiness/2017.csv')

data2018 = pd.read_csv('../input/world-happiness/2018.csv')

data2019 = pd.read_csv('../input/world-happiness/2019.csv')
print(data2015.shape)

data2015.head()
print(data2016.shape)

data2016.head()
print(data2017.shape)

data2017.head()
print(data2018.shape)

data2018.head()
print(data2019.shape)

data2018.head()
todrop15, todrop16 = ['Standard Error', 'Dystopia Residual'], ['Upper Confidence Interval', 'Lower Confidence Interval', 'Dystopia Residual']

data2015.drop(todrop15, axis = 1, inplace = True)

data2016.drop(todrop16, axis = 1, inplace = True)

data2015.columns, data2016.columns
columns1516mapping = {'Happiness Rank':'rank', 'Happiness Score': 'score',

                     'Economy (GDP per Capita)': 'GDP_per_capita', 'Health (Life Expectancy)': 'life_expectancy',

                     'Trust (Government Corruption)': 'gov_trust'}



data2015.rename(columns = columns1516mapping, inplace = True)

data2016.rename(columns = columns1516mapping, inplace = True)

data2015.columns == data2016.columns



#fixed 2015 and 2016 columns
#lower case the columns

data2015.columns = data2015.columns.str.lower()

data2016.columns = data2016.columns.str.lower()

data2015.columns == data2016.columns
data2017.drop(['Whisker.high','Whisker.low','Dystopia.Residual'], axis = 1, inplace = True)

data2017.shape, data2017.columns
columns17mapping = {'Happiness.Rank':'rank', 'Happiness.Score': 'score',

                     'Economy..GDP.per.Capita.': 'GDP_per_capita', 'Health..Life.Expectancy.': 'life_expectancy',

                     'Trust..Government.Corruption.': 'gov_trust'}

data2017.rename(columns = columns17mapping, inplace = True)

data2017.shape, data2017.columns
#make lower case

data2017.columns = data2017.columns.str.lower()

data2017.shape, data2017.columns
for i in data2017.columns:

    print(i in data2016.columns, i)

#all columns are now in 2016 columns. 1 missing is region
data2018.columns
columns1819mapping = {'Overall rank':'rank','Country or region':'country',

                     'GDP per capita': 'GDP_per_capita', 'Freedom to make life choices':'freedom',

                      'Healthy life expectancy': 'life_expectancy',

                     'Perceptions of corruption': 'gov_trust', 'Social support':'family'}



data2018.rename(columns = columns1819mapping, inplace=  True)

data2019.rename(columns = columns1819mapping, inplace = True)

data2018.columns = data2018.columns.str.lower()

data2019.columns = data2019.columns.str.lower()

data2018.columns == data2019.columns
data2019.columns
#check if all 2018,2019 columns in 2015,2016



for i in data2018.columns:

    print(i in data2016.columns, i)
empty_dict = {}

for country, region in zip(data2015['country'],data2015['region']):

    empty_dict[country] = region

for country, region in zip(data2016['country'],data2016['region']):

    empty_dict[country] = region
data2017['region'] = data2017['country'].map(empty_dict)

data2018['region'] = data2018['country'].map(empty_dict)

data2019['region'] = data2019['country'].map(empty_dict)
data2015['year'] = 2015

data2016['year'] = 2016

data2017['year'] = 2017

data2018['year'] = 2018

data2019['year'] = 2019
columnlis = []

columnlis.append([data2015.columns.values,data2016.columns.values,data2017.columns.values,data2018.columns.values,data2019.columns.values])

data2015.shape, data2016.shape, data2017.shape, data2018.shape, data2019.shape, np.unique(columnlis)
df = pd.concat([data2015,data2016,data2017,data2018,data2019], axis = 0, ignore_index = True)

df.info()

df.head()
df[df['gov_trust'].isnull()]
df.loc[df['country'] == 'United Arab Emirates', 'gov_trust'] = df.loc[df['country'] == 'United Arab Emirates', 'gov_trust'].fillna(df.loc[df['country'] == 'United Arab Emirates', 'gov_trust'].mean())
df.loc[df['country'] == 'United Arab Emirates', 'gov_trust']

#filled with the mean
df[df['region'].isnull()]
df[df['country'] == 'Taiwan']
df[df['country']=='Hong Kong']
#Taiwan

df.loc[347,'country'] = 'Taiwan'

df.loc[347,'region'] = 'Eastern Asia'



#Hongkong

df.loc[385,'country'] = 'Hong Kong'

df.loc[385,'region'] = 'Eastern Asia'
df.loc[df['region'].isnull(), 'country']
#check unique region

df['region'].unique()
df[df['country'].str.contains('Trinidad')]
#manullay change

df.loc[[507, 664],'country'] = 'Trinidad and Tobago'

df.loc[[507, 664],'region'] = 'Latin America and Caribbean'
df[df['country'] == 'Macedonia']
df[df['country'] == 'North Macedonia']
# we can set north macedonia to macedonia

df.loc[709, 'country'] = 'Macedonia'

df.loc[709, 'region'] = 'Central and Eastern Europe'
#from google search, gambia is west african country



df.loc[745,'region'] = 'Middle East and Northern Africa'
df[df['country'] == 'North Cyprus']
df[df['country'] == 'Northern Cyprus']
# Manually input western europe

df.loc[[527,689], 'country'] = 'North Cyprus'

df.loc[[527,689], 'region'] = 'Western Europe'
#there are no missing data

df.isnull().any(1).sum()
year_grouped = df.groupby('country')['year'].count()

year_grouped.value_counts()
plt.style.use('ggplot')

plt.figure(figsize = (10,8))

year_grouped[year_grouped < 4].plot.barh()

plt.title('Countries with less than 4 entries in the data')

plt.xlabel('Count')

plt.locator_params(axis='x', nbins=4)
df[df['country'].str.contains('Somaliland')]
df.loc[90,'country'] = 'Somaliland Region'
df['region'].unique()
plt.style.use('ggplot')

plt.figure(figsize = (10,6))

df['region'].value_counts().sort_values().plot.barh()

plt.title('Number of countries by region')

plt.xlabel('Counts')
df.loc[df['region'] == 'North America', 'country'].unique()
rank_group = df.groupby(['year','rank'])['country'].count()

rank_group[rank_group > 1]
c1 = (df['rank'] == 82)

c2 = (df['year'] == 2015)



df[c1&c2]

#Jordan and Montenegro both has rank 82 in 2015
c1 = (df['year'] == 2016)

c2 = (df['rank'].isin([34,57,145]))



df[c1&c2]
df.describe().loc[:,'score']
plt.figure(figsize = (10,6))

sns.kdeplot(df['score'])

plt.hist(df['score'], density = True, color = 'orange')

plt.ylabel('Prob')
df.describe().loc[:,'gdp_per_capita']
plt.figure(figsize = (10,6))

sns.kdeplot(df['gdp_per_capita'])

plt.hist(df['gdp_per_capita'], density = True, color = 'orange')

plt.ylabel('Prob')
df.describe().loc[:,'family':'generosity']
plt.figure(figsize = (10,8))

sns.heatmap(df.corr(), annot = True, cmap = 'RdBu')
def top(df, col = 'rank', n = 10):

    return df.sort_values(by = col)[:n]
top10 = df.groupby('year', as_index = False).apply(top, n = 10)

top10.head()
fig = plt.figure(figsize = (15,12))

ax = fig.add_subplot()

ax = sns.lineplot(data = top10, y = 'rank', x = 'year', hue = 'country')

ax.set_yticklabels(['','1st','2nd','3rd','4th','5th','6th','7th','8th','9th','10th',''])

plt.locator_params(axis='y', nbins=11)

plt.locator_params(axis='x', nbins=6)

plt.legend(ncol = 3, frameon = False, fontsize = 8)

ax.set_ylim(11, 0)

plt.title('Top 10 countries per year')
denmark = df[df['country'] == 'Denmark']

denmark
denmark_details = denmark.describe().loc['mean','score':'generosity']

denmark_details.name = 'Denmark'
idx = pd.IndexSlice

average_by_region = df.groupby('region').describe().loc[:,idx['score':'generosity','mean']]

average_by_region.columns = average_by_region.columns.swaplevel().droplevel()

denmark_compared = average_by_region.T.merge(denmark_details, right_index = True, left_index = True).T

denmark_compared
fig, ax = plt.subplots(figsize = (15,15))

denmark_compared.drop(['family','gov_trust','generosity'],

                     axis = 1).sort_values(by = ['score'], 

                                           ascending = False).T.plot.bar(ax = ax, width = .8)

plt.title('Denmark compared to other regions')

ax.set_xticklabels(['Score','GDP','Life Expectancy','Freedom'], rotation = 60);
rank = df.loc[:,['country','rank','year','score']]

rank.head()
rankgroup = rank.groupby(['country','year'])['rank'].mean().unstack()

rankgroup['rank diff'] = rankgroup.max(1) - rankgroup.min(1)
top10changes = rankgroup.sort_values(by = 'rank diff', ascending = False).head(10)
country_top10_change = top10changes.drop('rank diff', 1).stack().reset_index().rename(columns = {0:'rank'})
plt.figure(figsize = (15,10))

sns.lineplot(data = country_top10_change, x = 'year', y = 'rank', hue = 'country', palette = sns.color_palette("hls", 10))

plt.legend(ncol = 2)

plt.locator_params(axis='x', nbins=6)

plt.gca().invert_yaxis()

plt.title('Top 10 countries with the highest change in rank');

fig = plt.figure(figsize = (15,12))

ax = fig.add_subplot()

sns.boxplot(data = df, x = 'region', y = 'score', ax = ax)

ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right');
df.loc[df['region'].str.contains('Australia'),'country'].unique()
def top_score(df, col = 'score', n = 5, ascending = True):

    return df.sort_values(by = col)[-n:]
top_byregion = df.groupby(['country','region'], as_index = False)['score'].mean().groupby('region').apply(top_score, n = 1).reset_index(drop = True)

fig = plt.figure(figsize = (12,10))

ax = fig.add_subplot()

top_byregion.set_index(['region','country']).plot.bar(ax = ax)



#make 2 rows for xticklabels 

indexes = list(top_byregion.set_index(['region','country']).index.values)

multilist = []

for i in indexes:

    multilist.append(list(i))

region_country = []

for i in multilist:

    region_country.append('\n'.join(i))

ax.set_xticklabels(region_country);

plt.legend('')

plt.title('Happiest Country by Region')
EuMe = df[(df['region'] == 'Western Europe') | df['region'].str.contains('Middle East')]

Top5EuMe = EuMe.groupby(['country','region'], as_index = False)['score'].mean().groupby('region').apply(top_score).reset_index(drop = True)



#plotting

fig = plt.figure(figsize = (12,10))

ax = fig.add_subplot()

sns.barplot(x = 'country', y ='score', data = Top5EuMe, hue = 'region', ax = ax)

plt.title('Top 5 highest scores in Regions WE and ME-NA')

ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right');
def bottom(df, col = 'score', n = 5, ascending = True):

    return df.sort_values(by = col)[:n]
SubAfr_EastAsia = df[(df['region'] == 'Southern Asia') | df['region'].str.contains('Sub-')]

Bottom5 =  SubAfr_EastAsia.groupby(['country','region'], as_index = False)['score'].mean().groupby('region').apply(bottom, ascending = False, n = 5).reset_index(drop = True)



#plotting

fig = plt.figure(figsize = (12,10))

ax = fig.add_subplot()

sns.barplot(x = 'country', y ='score', data = Bottom5, hue = 'region', ax = ax)

plt.title('Bottom 5 countries in Southern Asia and Sub-Suharan Africa')

ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right');
region_score = df.groupby(['region','year'], as_index = False).agg({'score':['mean',np.std]}).rename(columns = {'mean':'mean_score', 'std':'std_score'})

region_score.columns = region_score.columns.droplevel(1)

region_score.columns = ['region','year','mean_score','std_score']
c = ['red','blue','green','cyan','magenta','yellow','black','brown','gray','pink']
def year_plot(df, title, col = ['region'], ylim = [3.5,7]):

    fig = plt.figure(figsize = (12,10))

    ax = plt.subplot()

    for i in df.groupby(col):

        plt.plot(i[1].year, i[1].mean_score, label = i[0], marker = 'o')

    plt.legend(ncol = 2, fontsize = 8, columnspacing = .5)

    plt.ylim(ylim)

    plt.title(title)

    plt.locator_params(axis='x', nbins=6)

    

year_plot(region_score, title = 'Happiness Score by region')
def topN(df, col,  title = 'Please insert title', n=10):

    plt.figure(figsize = (10,8))

    df.groupby('country')[col].mean().sort_values(ascending = False)[:n].plot(kind = 'bar', title = title) 
topN(df, col = 'generosity', title = 'Top10 most generous country')
topN(df, col = 'freedom', title = 'Top10 most free country')
topN(df, col = 'gov_trust', n = 10, title = 'Top 10 countries with highest trust on its government')
topN(df, col = 'family', n = 10, title = 'Top 10 countries with highest family/social support')
topN(df, col = 'gdp_per_capita', n = 10, title = 'Top 10 countries with highest trust on its government')
Top10GDP_rank = df.groupby('country')['gdp_per_capita','rank'].mean().nlargest(10, 'gdp_per_capita').loc[:,'rank']

fig = plt.figure(figsize = (10,8))

ax = fig.add_subplot()

ax.plot(Top10GDP_rank.index,Top10GDP_rank, linestyle = '', marker = 'o')

ax.set_xticklabels(Top10GDP_rank.index, rotation = 60)

ax.set_ylim(0, 80)

for country, rank in zip(Top10GDP_rank.index,Top10GDP_rank):

    ax.annotate(rank,(country, rank + 3))
ph = df[df['country'] == 'Philippines']

ph
average = df.describe().loc['mean','score':'generosity']

average.name = 'Global Average'
ph_mean = ph.describe().loc['mean','score':'generosity']

ph_mean.name = 'Philippine'
fig = plt.figure(figsize = (10,8))

ax = fig.add_subplot()

pd.DataFrame([average, ph_mean]).T.plot.barh(ax = ax)

ax.set_ylabel('Indices')

ax.set_xlabel('Value')

ax.set_title('Philippine happiness indices compared to global average')
fig = plt.figure(figsize = (10,8))

ax = fig.add_subplot()

ax.plot(ph['year'], ph['rank'])

ax.locator_params(axis='x', nbins=6)

ax.invert_yaxis()

ax.set_ylabel('Rank')

ax.set_xlabel('year')

ax.set_title('Philippine ranking on happiness (2015 - 2019)')



for year, rank in zip(ph['year'],ph['rank']):

    ax.annotate(rank,(year, rank + 1))
fig = plt.figure(figsize = (10,8))

ax = fig.add_subplot()

ph.set_index('year').loc[:,'gdp_per_capita':'generosity'].plot(kind = 'line', ax = ax, marker = 'o')

plt.legend(ncol = 2)

ax.locator_params(axis='x', nbins=6)

ax.set_ylim(0, 1.4)

plt.title('Happiness index in the Philippines')
fig = plt.figure(figsize = (10,10))

ax = fig.add_subplot()

df[df['region'] == 'Southeastern Asia'].groupby('country').agg('mean').drop(['rank','year'],1).sort_values(by = 'score', ascending = False).T.plot(kind = 'bar', ax = ax, width = .8)

ax.set_title('Philippine compared to other regions')

ax.set_ylabel('Value')

ax.set_xlabel('Variables')

plt.legend(ncol = 3)

ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right');
df.head()
indices = df.groupby('country').agg('mean').loc[:,'gdp_per_capita':'generosity']

plt.figure(figsize = (10,8))

for i in indices.columns:

    sns.distplot(df[i], label = i)

plt.legend()

plt.title('Distribution of happiness indices')
regions = df['region'].unique()

regions
fig = plt.figure(figsize = (12,10))

ax = fig.add_subplot()

sns.scatterplot(data = df, x = 'score', y ='gdp_per_capita', hue = 'region', ax = ax)

ax.set_title('Scatterplot of GDP and score');
sns.lmplot(data = df, x = 'gdp_per_capita', y='generosity', height = 10)

plt.title('GDP per capita vs generosity')
plt.figure(figsize = (12,10))

sns.scatterplot(data= df, x = 'family', y = 'life_expectancy', size = 'gdp_per_capita')

plt.legend(fontsize = 'large')

plt.title('Social Support and Life expectancy')
sns.lmplot(data = df, x = 'gov_trust', y = 'freedom', hue = 'region', height = 10, legend_out = False)

plt.title('Government Trust vs Freedom')
plt.figure(figsize = (10,8))

sns.kdeplot(df['score'], shade = True)
import scipy.stats as st

mean = df['score'].mean()

std = df['score'].std()



xs = np.linspace(0,9, 10000)

ps = st.norm.pdf(xs, mean, std)



plt.figure(figsize = (10,8))

sns.kdeplot(df['score'], shade = True)

plt.plot(xs, ps)
mean = df['score'].mean()

std = df['score'].std()



xs = np.linspace(0,9, 10000)

ps = st.norm.pdf(xs, mean, std)



p_lognorm = st.lognorm.fit(df['score'])

pdf_lognorm = st.lognorm.pdf(xs, *p_lognorm)



p_skewnorm = st.skewnorm.fit(df['score'])

pdf_skewnorm = st.skewnorm.pdf(xs, *p_skewnorm)



plt.figure(figsize = (10,8))

sns.kdeplot((df['score']), shade = True)

plt.plot(xs, ps, label = 'normal')

plt.plot(xs, pdf_lognorm, label = 'log norm')

plt.plot(xs, pdf_skewnorm, label = 'skew norm')

plt.legend()
import scipy.stats as st

def get_best_distribution(data):

    dist_names = ["norm", "exponweib", "weibull_max", "weibull_min", "pareto", "genextreme"]

    dist_results = []

    params = {}

    for dist_name in dist_names:

        dist = getattr(st, dist_name)

        param = dist.fit(data)



        params[dist_name] = param

        # Applying the Kolmogorov-Smirnov test

        D, p = st.kstest(data, dist_name, args=param)

        print("p value for "+dist_name+" = "+str(p))

        dist_results.append((dist_name, p))



    # select the best fitted distribution

    best_dist, best_p = (max(dist_results, key=lambda item: item[1]))

    # store the name of the best fit and its p value



    print("Best fitting distribution: "+str(best_dist))

    print("Best p value: "+ str(best_p))

    print("Parameters for the best fit: "+ str(params[best_dist]))



    return best_dist, best_p, params[best_dist]
best_dist, best_p, params = get_best_distribution(df['score'])