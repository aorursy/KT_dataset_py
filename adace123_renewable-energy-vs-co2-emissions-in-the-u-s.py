import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
from sklearn.linear_model import LinearRegression
%matplotlib inline
co2_per_cap_indicator = 'CO2 emissions (metric tons per capita)'
renewable_output_indicator = 'Renewable electricity output (% of total electricity output)'
renewable_consumption_indicator = 'Renewable energy consumption (% of total final energy consumption)'
total_energy_use_indicator = 'Energy use (kg of oil equivalent per capita)'
total_electricity_consumption_indicator = 'Electric power consumption (kWh per capita)'
indicators_of_interest = [co2_per_cap_indicator, renewable_consumption_indicator, \
                          renewable_output_indicator, total_energy_use_indicator, total_electricity_consumption_indicator]
indicator_data = pd.read_csv('../input/Indicators.csv')
countries = pd.read_csv('https://raw.githubusercontent.com/lukes/ISO-3166-Countries-with-Regional-Codes/master/all/all.csv')
indicator_data = indicator_data[indicator_data['CountryCode'].isin(countries['alpha-3'].values)]
co2_emissions_per_cap = indicator_data[indicator_data['IndicatorName'] == co2_per_cap_indicator]
us_data = indicator_data[indicator_data['CountryName'] == 'United States']
indicator_data.shape
# Total number of unique countries in the dataset
len(indicator_data['CountryName'].unique())
avg_co2_by_country = co2_emissions_per_cap.groupby('CountryName')['Value'].mean().sort_values(ascending=False)
avg_co2_by_country = avg_co2_by_country.to_frame().reset_index()
total_top_10 = avg_co2_by_country['Value'][:10].sum()
world_avg = avg_co2_by_country['Value'].sum()
us_avg = avg_co2_by_country[avg_co2_by_country['CountryName'] == 'United States']['Value']
avg_co2_by_country[:10].plot.bar(x='CountryName', y='Value', figsize=(9, 6), legend=False)
ax = plt.subplot()
ax.text(0.18, 0.9, "The top 10 make up {}% of the world's total CO2 emissions per capita.\nThe U.S. accounts for {}% total.".format(int((total_top_10 / world_avg) * 100), round(us_avg / world_avg, 3)), transform=ax.transAxes)
plt.xlabel('Country')
plt.ylabel(co2_per_cap_indicator)
plt.title('Average Emissions for the World\'s Top 10 Per Capita Polluters')
plt.show()
co2_emissions_per_cap[co2_emissions_per_cap['CountryName'] == 'United States'].plot('Year', 'Value', figsize=(13, 5))
plt.show()
co2_emissions_per_cap.loc[co2_emissions_per_cap['Value'].idxmax()][['CountryName', 'Value', 'Year']]
co2_emissions_per_cap.loc[co2_emissions_per_cap[co2_emissions_per_cap['CountryName'] == 'United States']['Value'].idxmax()]
pct_changes = co2_emissions_per_cap.groupby('CountryName').apply(lambda g: g['Value'].pct_change().mean()).sort_values(ascending=False)
rate_ratio = np.divide(pct_changes['United States'], pct_changes.mean())
fig, axes = plt.subplots(3, 1, figsize=(13, 18))
ax1 = pct_changes.head(5).plot.bar(ax=axes[0], fontsize=14)
ax2 = pct_changes.tail(5).plot.bar(ax=axes[1], fontsize=14)

pct = co2_emissions_per_cap.pivot(index='Year', columns='CountryName', values='Value')['United States'].pct_change().to_frame().reset_index().dropna()
reg = LinearRegression()
reg.fit(pct[['Year']], pct[['United States']])
pct.plot.scatter(x='Year', y='United States', legend=False, ax=axes[2])
axes[2].plot(pct['Year'], reg.predict(pct[['Year']]))

ax1.set_title('CO2 Per Capita Change: Top 5 Fastest Growing Countries', fontsize=14)
ax2.set_title('CO2 Per Capita Change: Top 5 Fastest Shrinking Countries', fontsize=14)
ax1.set_ylabel('Average % of CO2 Per Capita Change', fontsize=14)
ax2.set_ylabel('Average % of CO2 Per Capita Change', fontsize=14)
axes[2].set_ylabel('Average % of CO2 Per Capita Change', fontsize=14)
axes[2].set_title('US CO2 Per Capita Percent Changes Over Time', fontsize=14)

plt.tight_layout()
plt.show()
def plot_indicators(indicators, labels):
    dfs = [us_data[(us_data['IndicatorName'] == i)] for i in indicators]
    data = [df.pivot(columns='CountryName', index='Year', values='Value') for df in dfs]
    fig, axes = plt.subplots(len(indicators) // 2, len(indicators), figsize=(15,5))
    for i, df in enumerate(dfs):
        axes[i].plot(data[i])
        axes[i].set_title(labels[i])
        axes[i].set_xlabel('Year')
        axes[i].set_ylabel(indicators[i][indicators[i].index('('):])
        print("Average {} is {}".format(indicators[i], round(df['Value'].mean(), 2)))
    plt.show()
renewable_labels = ['Renewable Energy Consumption for the US', 'Renewable Energy Output for the US']
plot_indicators([renewable_output_indicator, renewable_consumption_indicator], renewable_labels)
labels = ['Total Energy Use for the U.S.', 'Total Electricity Consumption for the U.S.']
plot_indicators([total_energy_use_indicator, total_electricity_consumption_indicator], labels)
corr_matrix = us_data.pivot_table(index=['CountryName', 'Year'], columns='IndicatorName', values='Value').corr().loc[indicators_of_interest]
corr_matrix = corr_matrix.dropna(axis=1)
corr_matrix_fig = sns.heatmap(corr_matrix.loc[:, indicators_of_interest])
plt.show()
