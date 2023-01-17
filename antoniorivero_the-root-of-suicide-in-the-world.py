import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
data = pd.read_csv('../input/master.csv')
data.head()
data.info()
# HDI data was collected every 5 years at the beginning. Last 5 years are yearly values. 

data[['year','HDI for year']].dropna().drop_duplicates('year').sort_values('year')
sns.heatmap(data.corr())
data['HDI for year'] = data.groupby('country')['HDI for year'].transform(lambda x: x.fillna(method = 'bfill'))

data['HDI for year'] = data.groupby('country')['HDI for year'].transform(lambda x: x.fillna(method = 'ffill'))
data.info()
data.groupby('country')['HDI for year'].count()[lambda x: x == 0]
plt.figure(figsize = (15,7))

ax = plt.subplot(121)

data.groupby(['country-year','sex','generation'])['suicides/100k pop'].mean().unstack(1).nlargest(10, 'male').plot.barh(ax = ax)

plt.title('Top Suicide Rate for Males')

ax = plt.subplot(122)

data.groupby(['country-year','sex','generation'])['suicides/100k pop'].mean().unstack(1).nlargest(10, 'female').plot.barh(ax = ax)

plt.title('Top Suicide Rate for Females')

plt.tight_layout()
hung = data.query('country == "Hungary"')
sns.pairplot(hung)
hung.head()
hung.loc[:,'age'] = pd.Categorical(hung['age'], categories = hung['age'].unique()[[0,2,1,3,4,5]], ordered = True)
hung.info()
sns.barplot(data = hung, x = 'sex', y = 'suicides/100k pop', hue = 'age')
sns.barplot(data = hung, x = 'sex', y = 'population', hue = 'age', estimator = np.sum)
sns.barplot(data = hung, x = 'sex', y = 'suicides_no', hue = 'age', estimator = np.sum)
# sns.catplot(kind = 'bar', 

#             data = hung, 

#             x = 'sex', 

#             y = 'suicides_no', 

#             hue = 'age', 

#             col = 'year', col_wrap = 3)

print('The above plot summarizes the data pretty well. The pattern is present throughout the years')
hung.groupby('year')['gdp_per_capita ($)'].std().head()

# No variation through the year, can take first value of every year.
gdp_capita = hung.groupby('year')[['gdp_per_capita ($)']].first()
gdp_capita = gdp_capita.join(hung.groupby('year')['suicides_no'].sum())
gdp_capita.plot(kind = 'scatter', x = 'gdp_per_capita ($)', y = 'suicides_no')

for i in range(len(gdp_capita)):

    plt.annotate(gdp_capita.index[i], (gdp_capita.iloc[i,0],gdp_capita.iloc[i,1]))
gdp_capita.corr()