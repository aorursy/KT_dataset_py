import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sb

from numpy import mean

plt.style.use('ggplot')

data = pd.read_csv('../input/suicide-rates-overview-1985-to-2016/master.csv')

data.columns
data=data.rename(columns={'country':'Country',

                          'year':'Year',

                          'sex':'Sex',

                          'age':'Age',

                          'suicides_no':'SuicidesNo',

                          'population':'Population',

                          'suicides/100k pop':'Suicides100kPop',

                          'country-year':'CountryYear',

                          'HDI for year':'HDIForYear',

                          ' gdp_for_year ($) ':'GdpForYear',

                          'gdp_per_capita ($)':'GdpPerCapita',

                          'generation':'Generation'})
data.isnull().sum()
df = data.groupby(['Country'])

df.SuicidesNo.mean().nlargest(10).plot(kind='barh')

plt.xlabel("Average Number of Suicides (1985-2015)")

plt.title("Top 10 Countries by No. of Suicides")
df = data.groupby(['Country'])

df.Suicides100kPop.mean().nlargest(10).plot(kind='barh')

plt.xlabel("Avg. Number of Suicides per 100k (1985-2015)")

plt.title("Top 10 Countries by Prop. of Suicides per 100k")
data['Sex'].value_counts().plot.bar()

plt.xlabel('Sex')

plt.ylabel('# of Observations')

plt.title('Number of Observations by Sex')
df = data.groupby(['Sex'])

df.SuicidesNo.sum().plot(kind='pie', autopct='%1.1f%%', label='World Suicides by Sex')


sb.catplot(x='Sex', y='SuicidesNo', col='Age', data=data, estimator=mean, kind='bar', col_wrap=3, ci=False, col_order=['5-14 years', '15-24 years', '25-34 years', '35-54 years', '55-74 years', '75+ years'])
plt.figure(figsize=(35,15))

sb.set_context("paper", 2.5, {"lines.linewidth": 4})

sb.barplot(data=data,x='Year',y='SuicidesNo',hue='Sex', ci=False)
plt.figure(figsize=(30,10))

sb.lineplot(x='Year', y='SuicidesNo', data=data, hue='Sex', estimator=mean)
sb.lineplot(x='Year', y='Population', data=data, hue='Sex', estimator=mean, ci=False)