import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
beers = pd.read_csv('../input/craft-cans/beers.csv')
beers.head()
breweries = pd.read_csv('../input/craft-cans/breweries.csv')
breweries.head()
beers.rename(columns={'id': 'beer_id', 'name': 'beer_name'}, inplace=True)
beers.head()
# Create a column for the brewery index
breweries['brewery_id'] = breweries.index

# Merge the two dataframes together based on brewery ID
df = beers.merge(breweries, on='brewery_id')
df.head()
df.drop(labels=['Unnamed: 0_x', 'Unnamed: 0_y'], axis=1, inplace=True)
df.head()
df.head(10)
breweries.isnull().sum()
print('Number of records:', breweries.shape[0])
# Plot a bar chart with the number of breweries in each state
breweries_by_state = breweries['state'].value_counts()
plt.figure(figsize=(10,8))
sns.barplot(x=breweries_by_state.index, y=breweries_by_state.values)
plt.title('Number of Breweries by State')
plt.ylabel('Number of Breweries')
plt.xlabel('State')
plt.xticks(rotation='vertical')
# Create a series that contains the number of breweries in the 20 cities with the highest brewery count
breweries_by_city = breweries.groupby('city')['name'].count().nlargest(20)
plt.figure(figsize=(10,8))
sns.barplot(x=breweries_by_city.index, y=breweries_by_city.values)
plt.title('Top 20 Cities by Number of Breweries')
plt.ylabel('Number of Breweries')
plt.xlabel('State')
plt.xticks(rotation=45)
breweries.head()
# Create a new dataframe with only the washington breweries
washington_breweries = breweries[breweries['state']==' WA']
print('Number of breweries located in Washington:', washington_breweries.shape[0])
# Get the value counts of each washington city featured in the dataset and plot
wa_breweries_by_city = washington_breweries['city'].value_counts()
plt.figure(figsize=(8,6))
sns.barplot(x=wa_breweries_by_city.index, y=wa_breweries_by_city.values)
plt.title('Number of Breweries in Washington Cities')
plt.ylabel('Number of Breweries')
plt.xlabel('City')
plt.xticks(rotation=45)
beers.head()
print('Total number of induvidual beers in the dataset:', beers.shape[0])
beers.isnull().sum()
print('Average ABV of all beers:', beers['abv'].mean())
print('Average IBU of all beers:', beers['ibu'].mean())
# Create a histogram of ABV
plt.figure(figsize=(8,8))
sns.distplot(a=beers['abv'])
plt.title('Histogram of Beer ABV')
plt.ylabel('Frequency')
plt.xlabel('ABV')
# Create a histogram of IBU
plt.figure(figsize=(8,8))
sns.distplot(a=beers['ibu'])
plt.title('Histogram of Beer IBU')
plt.ylabel('Frequency')
plt.xlabel('IBU')
# Create a scatter plot comparing ABV and IBU and plot a regression line
plt.figure(figsize=(10,8))
sns.regplot(x=beers['abv'], y=beers['ibu'])
plt.title('ABV vs. IBU')
plt.xlabel('ABV')
plt.ylabel('IBU')
# Create a joint kde plot of ABV and IBU
sns.jointplot(data=beers, x='abv', y='ibu', kind='kde')
# Create a series with the styles that have the top 20 most records in the dataframe
beers_by_type = beers['style'].value_counts().nlargest(20)
beers_by_type
# Plot the top 20 most popular beer styles
plt.figure(figsize=(8,6))
sns.barplot(x=beers_by_type.index, y=beers_by_type.values)
plt.title('Top 20 Styles of Beer that Appear in Dataset')
plt.xlabel('Beer Style')
plt.ylabel('Number of Beers')
plt.xticks(rotation='vertical')
print('The highest ABV beer:', (beers['abv'].max() * 100), '%')
print('The lowest ABV beer:', (beers['abv'].min() * 100), '%')
print('The highest IBU beer:', beers['ibu'].max())
print('The lowest IBU beer:', beers['ibu'].min())
highest_avg_abv = beers.groupby('style')[['abv']].mean().nlargest(10, columns='abv')
highest_avg_abv.sort_values(by='abv', ascending=False)
highest_avg_ibu = beers.groupby('style')[['ibu']].mean().nlargest(10, columns='ibu')
highest_avg_ibu.sort_values(by='ibu', ascending=False)
lowest_avg_ibu = beers.groupby('style')[['ibu']].mean().nsmallest(10, columns='ibu')
lowest_avg_ibu.sort_values(by='ibu', ascending=True)
df.head()
avg_abv_by_state = df.groupby('state')[['abv']].mean().sort_values(by='abv', ascending=False)
plt.figure(figsize=(8,11))
sns.barplot(y=avg_abv_by_state.index, x=avg_abv_by_state['abv'], orient='horizontal')
plt.title('States Ranked by Average ABV')
plt.ylabel('State')
plt.xlabel('ABV')
avg_ibu_by_state = df.groupby('state')[['ibu']].mean().sort_values(by='ibu', ascending=False)

plt.figure(figsize=(8,11))
sns.barplot(y=avg_ibu_by_state.index, x=avg_ibu_by_state['ibu'], orient='horizontal')
plt.title('States Ranked by Average IBU')
plt.ylabel('State')
plt.xlabel('IBU')