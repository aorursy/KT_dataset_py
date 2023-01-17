import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline
wh2015 = pd.read_csv('../input/2015.csv')
wh2015.info()
wh2016 = pd.read_csv('../input/2016.csv')
wh2016.info()
#max happiness score for 2015

wh2015[wh2015['Happiness Score'] == wh2015['Happiness Score'].max()][['Country','Happiness Score', 'Happiness Rank']]
#max happiness score for 2015

wh2016[wh2016['Happiness Score'] == wh2016['Happiness Score'].max()][['Country','Happiness Score', 'Happiness Rank']]
#min happiness score for 2015

wh2015[wh2015['Happiness Score'] == wh2015['Happiness Score'].min()][['Country','Happiness Score', 'Happiness Rank']]
#min happiness score for 2016

wh2016[wh2016['Happiness Score'] == wh2016['Happiness Score'].min()][['Country','Happiness Score', 'Happiness Rank']]
#factors for most happy country - Switzerland 2015

wh2015[wh2015['Country'] == 'Switzerland'][['Economy (GDP per Capita)', 'Family',

       'Health (Life Expectancy)', 'Freedom',

       'Trust (Government Corruption)', 'Generosity', 'Happiness Score', 'Happiness Rank']]
#factors for- Switzerland 2016

wh2016[wh2016['Country'] == 'Switzerland'][['Economy (GDP per Capita)', 'Family',

       'Health (Life Expectancy)', 'Freedom',

       'Trust (Government Corruption)', 'Generosity', 'Happiness Score', 'Happiness Rank']]
#factors for most happy country - Denmark 2016

wh2016[wh2016['Country'] == 'Denmark'][['Economy (GDP per Capita)', 'Family',

       'Health (Life Expectancy)', 'Freedom',

       'Trust (Government Corruption)', 'Generosity', 'Happiness Score', 'Happiness Rank']]
#factors for- Denmark 2015

wh2015[wh2015['Country'] == 'Denmark'][['Economy (GDP per Capita)', 'Family',

       'Health (Life Expectancy)', 'Freedom',

       'Trust (Government Corruption)', 'Generosity', 'Happiness Score', 'Happiness Rank']]
#increase in family and generosity seems major factor in affecting overall happiness score
#plot for Happiness Score vs all factors to judge if Family and generosity play major role in 2015

sns.pairplot(data = wh2015.drop(['Country', 'Region', 'Happiness Rank', 'Standard Error', 'Dystopia Residual'], axis = 1))
#plot for Happiness Score vs all factors to judge if Family and generosity play major role in 2015

sns.pairplot(data = wh2016.drop(['Country', 'Region', 'Happiness Rank', 'Dystopia Residual', 'Upper Confidence Interval', 'Lower Confidence Interval'], axis = 1))
#Region wise happiness scores for 2015 show western europe to be relatively happier than others and sub saharan africa as least happiest

plt.figure(figsize=(12,6))

sns.swarmplot('Region', 'Happiness Score', data = wh2015)

plt.xticks(rotation = 60)
#Region wise happiness scores for 2016 show western europe to be relatively happier than others and sub saharan africa as least happiest

plt.figure(figsize=(12,6))

sns.swarmplot('Region', 'Happiness Score', data = wh2016)

plt.xticks(rotation = 60)