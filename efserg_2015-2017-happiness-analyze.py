import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import warnings
import plotly.graph_objs as go
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

warnings.filterwarnings('ignore')
init_notebook_mode(connected=True)
%matplotlib inline
happy2015 = pd.read_csv("../input/2015.csv").rename(columns=
                             {'Country':'Country','Region':'Region','Happiness Rank':'Happiness.Rank', 'Happiness Score':'Happiness.Score','Standard Error':'Standard.Error', 'Economy (GDP per Capita)':'Economy','Family':'Family','Health (Life Expectancy)':'Health', 'Freedom':'Freedom','Trust (Government Corruption)':'Trust','Generosity':'Generosity', 'Dystopia Residual':'Dystopia.Residual'})

happy2016 = pd.read_csv("../input/2016.csv").rename(columns=
                             {'Country':'Country','Region':'Region','Happiness Rank':'Happiness.Rank', 'Happiness Score':'Happiness.Score','Lower Confidence Interval':'Lower.Confidence.Interval', 'Upper Confidence Interval':'Upper.Confidence.Interval', 'Standard Error':'Standard.Error', 'Economy (GDP per Capita)':'Economy','Family':'Family','Health (Life Expectancy)':'Health', 'Freedom':'Freedom','Trust (Government Corruption)':'Trust','Generosity':'Generosity', 'Dystopia Residual':'Dystopia.Residual'})

happy2017 = pd.read_csv("../input/2017.csv").rename(columns=
                             {'Country':'Country','Region':'Region','Happiness.Rank':'Happiness.Rank', 'Happiness.Score':'Happiness.Score','Whisker.low':'Lower.Confidence.Interval', 'Whisker.high':'Upper.Confidence.Interval', 'Standard Error':'Standard.Error', 'Economy..GDP.per.Capita.':'Economy','Family':'Family','Health..Life.Expectancy.':'Health', 'Freedom':'Freedom','Trust..Government.Corruption.':'Trust','Generosity':'Generosity', 'Dystopia.Residual':'Dystopia.Residual'})

happy2015['year'] = 2015
happy2016['year'] = 2016
happy2017['year'] = 2017
happy2015.describe()
happy2016.describe()
happy2017.describe()
happy2015.info()
happy2016.info()
happy2017.info()
region2015 = dict(zip(list(happy2015['Country']), list(happy2015['Region'])))
region2016 = dict(zip(list(happy2016['Country']), list(happy2016['Region'])))
regions = dict(region2015, **region2016)

def find_region(row):
    return regions.get(row['Country'])


happy2017['Region'] = happy2017.apply(lambda row: find_region(row), axis=1)
happy2017.head()
happy2017[happy2017['Region'].isna()]['Country']
happy2017 = happy2017.fillna(value = {'Region': regions['China']})
happiness_agg = happy2015.copy()
happiness_agg = happiness_agg.append(happy2016, ignore_index=True)
happiness_agg = happiness_agg.append(happy2017, ignore_index=True)
happiness_agg.info()
happiness = happiness_agg.drop(columns=['Lower.Confidence.Interval', 'Upper.Confidence.Interval', 'Standard.Error'])
happiness.describe()
happiness.columns
key_variables = ['Economy', 'Family', 'Freedom', 'Generosity', 'Health', 'Trust']
happiness.boxplot(column = key_variables, figsize = (10,6))
fig, axes = plt.subplots(figsize = (10, 8))

sns.violinplot(data=happiness[key_variables], ax = axes, inner="points")
sns.despine()
axes.yaxis.grid(True)

plt.show()
corr = happiness[key_variables].corr()

mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

f, ax = plt.subplots(figsize=(11, 9))
cmap = sns.diverging_palette(220, 10, as_cmap=True)
sns.heatmap(corr, mask=mask, cmap=cmap, annot=True, vmax=.3, center=0, square=True, linewidths=.5, cbar_kws={"shrink": .5})
sns.set()
sns.pairplot(happiness[key_variables], size = 2.5)
plt.show();
happiness[key_variables].corrwith(happiness['Happiness.Score'], axis=0, drop=False)
sns.pairplot(happiness, x_vars=key_variables, y_vars=['Happiness.Score'], size = 3)
plt.show()
fig, axes = plt.subplots(figsize = (16, 8))
sns.swarmplot(x="Region", y="Happiness.Score",  data=happiness, ax = axes)
plt.xticks(rotation=45)
key_variables_with_dystopia = key_variables.copy()
key_variables_with_dystopia.append('Dystopia.Residual')
happiness_with_distopia = happiness.groupby(['year']).mean()[key_variables_with_dystopia]

data_perc = happiness_with_distopia.divide(happiness_with_distopia.sum(axis=1), axis=0).reset_index()
f, ax = plt.subplots(figsize=(11, 9))

pal = sns.color_palette(['#417A56', '#A6E449', '#A39994', '#D95B30', '#FDEC59', '#5EDBDD', '#E20E5F'])

plt.stackplot(data_perc['year'].values, data_perc[key_variables_with_dystopia].T,
              labels=happiness_with_distopia, colors=pal)

plt.legend(loc='upper left')
plt.margins(0,0)

plt.show() 
rank_by_year_correlation = happiness[['Country', 'Happiness.Score', 'year']] \
    .groupby(['Country']).corr().ix[0::2,'year'].reset_index(name='correlation') \
    [['Country', 'correlation']]
rank_by_year_correlation.head(5)
data = dict(type = 'choropleth', locations = rank_by_year_correlation['Country'],
           locationmode = 'country names', z = rank_by_year_correlation['correlation'], 
           text = rank_by_year_correlation['Country'], colorbar = {'title':'Change of happiness'})

layout = dict(geo = dict(showframe = False, projection = {'type': 'Mercator'}))

choromap = go.Figure(data=[data], layout=layout)
iplot(choromap)
