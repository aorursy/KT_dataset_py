#for manipulating dataframes

import pandas as pd



#for visualisation

import matplotlib.pyplot as plt

import seaborn as sns

plt.style.use('fivethirtyeight')
d_2015 = pd.read_csv('../input/world-happiness/2015.csv')

d_2016 = pd.read_csv('../input/world-happiness/2016.csv')

d_2017 = pd.read_csv('../input/world-happiness/2017.csv')
plt.rcParams['figure.figsize'] = (16, 10)

sns.boxplot(d_2016['Region'], d_2016['Happiness Score'])

plt.xticks(rotation = 90)

plt.show()
sns.heatmap(d_2016.corr(), cmap = 'copper', annot = True)

plt.show()
D = d_2016.drop(['Happiness Rank', 'Lower Confidence Interval', 'Upper Confidence Interval'], axis = 1)

d = D.corr()['Happiness Score']

d = pd.DataFrame(d)

row_names = d.index

row_names = pd.DataFrame(row_names)

plt.plot(row_names[0], d['Happiness Score'])

plt.xticks(rotation = 90)

plt.show()
d = d_2016.groupby('Region')['Happiness Score', 'Economy (GDP per Capita)', 'Region'].median()

sns.scatterplot(d['Happiness Score'], d['Economy (GDP per Capita)'], hue = d.index, s = 500)

plt.show()
d1 = d_2016.loc[lambda d_2016: d_2016['Region'] == 'Western Europe']

d2 = d_2016.loc[lambda d_2016: d_2016['Region'] == 'North America']

d = pd.concat([d1, d2], axis = 0)

sns.heatmap(d.corr(), cmap = 'copper', annot = True)

plt.show()
d1 = d_2016.loc[lambda d_2016: d_2016['Region'] == 'Western Europe']

d2 = d_2016.loc[lambda d_2016: d_2016['Region'] == 'North America']

new_data = pd.concat([d1, d2], axis = 0)

D = new_data.drop(['Happiness Rank', 'Lower Confidence Interval', 'Upper Confidence Interval'], axis = 1)

d = D.corr()['Happiness Score']

d = pd.DataFrame(d)

row_names = d.index

row_names = pd.DataFrame(row_names)

plt.plot(row_names[0], d['Happiness Score'])

plt.xticks(rotation = 90)

plt.show()
sns.scatterplot(d_2016['Health (Life Expectancy)'], d_2016['Economy (GDP per Capita)'])

plt.show()
plt.subplot(1, 2, 1)

sns.scatterplot(d_2016['Health (Life Expectancy)'],  d_2016['Happiness Score'])

plt.subplot(1, 2, 2)

d = d_2016.loc[lambda d_2016: d_2016['Health (Life Expectancy)'] > .65]

sns.scatterplot(d['Health (Life Expectancy)'],  d['Happiness Score'])

plt.show()
d1 = d_2016.loc[lambda d_2016: d_2016['Region'] == 'Eastern Asia']

# d2 = d_2016.loc[lambda d_2016: d_2016['Region'] == 'SouthEastern Asia']

d2 = d_2016.loc[lambda d_2016: d_2016['Region'] == 'Central and Eastern Europe']



d = pd.concat([d1, d2], axis = 0)

sns.heatmap(d.corr(), cmap = 'copper', annot = True)

plt.show()
d1 = d_2016.loc[lambda d_2016: d_2016['Region'] == 'Eastern Asia']

# d2 = d_2016.loc[lambda d_2016: d_2016['Region'] == 'SouthEastern Asia']

d2 = d_2016.loc[lambda d_2016: d_2016['Region'] == 'Central and Eastern Europe']



new_data = pd.concat([d1, d2], axis = 0)

D = new_data.drop(['Happiness Rank', 'Lower Confidence Interval', 'Upper Confidence Interval'], axis = 1)

d = D.corr()['Happiness Score']

d = pd.DataFrame(d)

row_names = d.index

row_names = pd.DataFrame(row_names)

plt.plot(row_names[0], d['Happiness Score'])

plt.xticks(rotation = 90)

plt.show()
d1 = d_2016.loc[lambda d_2016: d_2016['Region'] == 'Eastern Asia']

# d2 = d_2016.loc[lambda d_2016: d_2016['Region'] == 'SouthEastern Asia']

d2 = d_2016.loc[lambda d_2016: d_2016['Region'] == 'Central and Eastern Europe']



d3 = d_2016.loc[lambda d_2016: d_2016['Region'] == 'Western Europe']

# d2 = d_2016.loc[lambda d_2016: d_2016['Region'] == 'SouthEastern Asia']

d4 = d_2016.loc[lambda d_2016: d_2016['Region'] == 'North America']



mid = pd.concat([d1, d2], axis = 0)

rich = pd.concat([d3, d4], axis = 0)

plt.subplot(3,2,1)

plt.title('Mid Happiness Region')

sns.scatterplot(mid['Freedom'], mid['Happiness Score'])

plt.subplot(3,2,2)

plt.title('High Happiness Region')

sns.scatterplot(rich['Freedom'], rich['Happiness Score'])

plt.subplot(3,2,3)

sns.scatterplot(mid['Trust (Government Corruption)'], mid['Happiness Score'])

plt.subplot(3,2,4)

sns.scatterplot(rich['Trust (Government Corruption)'], rich['Happiness Score'])

plt.subplot(3,2,5)

sns.scatterplot(mid['Generosity'], mid['Happiness Score'])

plt.subplot(3,2,6)

sns.scatterplot(rich['Generosity'], rich['Happiness Score'])



plt.show()
d1 = d_2016.loc[lambda d_2016: d_2016['Region'] == 'Eastern Asia']

# d2 = d_2016.loc[lambda d_2016: d_2016['Region'] == 'SouthEastern Asia']

d2 = d_2016.loc[lambda d_2016: d_2016['Region'] == 'Central and Eastern Europe']



d3 = d_2016.loc[lambda d_2016: d_2016['Region'] == 'Western Europe']

# d2 = d_2016.loc[lambda d_2016: d_2016['Region'] == 'SouthEastern Asia']

d4 = d_2016.loc[lambda d_2016: d_2016['Region'] == 'North America']



mid = pd.concat([d1, d2], axis = 0)

rich = pd.concat([d3, d4], axis = 0)

plt.subplot(3,2,1)

plt.title('Mid Happiness Region')

sns.scatterplot(mid['Economy (GDP per Capita)'], mid['Freedom'])

plt.subplot(3,2,2)

plt.title('High Happiness Region')

sns.scatterplot(rich['Economy (GDP per Capita)'], rich['Freedom'])

plt.subplot(3,2,3)

sns.scatterplot(mid['Economy (GDP per Capita)'], mid['Trust (Government Corruption)'])

plt.subplot(3,2,4)

sns.scatterplot(rich['Economy (GDP per Capita)'], rich['Trust (Government Corruption)'])

plt.subplot(3,2,5)

sns.scatterplot(mid['Economy (GDP per Capita)'], mid['Generosity'])

plt.subplot(3,2,6)

sns.scatterplot(rich['Economy (GDP per Capita)'], rich['Generosity'])



plt.show()
d1 = d_2016.loc[lambda d_2016: d_2016['Region'] == 'Sub-Saharan Africa']

d2 = d_2016.loc[lambda d_2016: d_2016['Region'] == 'Southern Asia']

d = pd.concat([d1, d2], axis = 0)

sns.heatmap(d.corr(), cmap = 'copper', annot = True)

plt.show()
d1 = d_2016.loc[lambda d_2016: d_2016['Region'] == 'Sub-Saharan Africa']

d2 = d_2016.loc[lambda d_2016: d_2016['Region'] == 'Southern Asia']

new_data = pd.concat([d1, d2], axis = 0)

D = new_data.drop(['Happiness Rank', 'Lower Confidence Interval', 'Upper Confidence Interval'], axis = 1)

d = D.corr()['Happiness Score']

d = pd.DataFrame(d)

row_names = d.index

row_names = pd.DataFrame(row_names)

plt.plot(row_names[0], d['Happiness Score'])

plt.xticks(rotation = 90)

plt.show()