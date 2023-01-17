import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt
covid_dataset = pd.read_csv('../input/confirmed-covid-cases/covid_confirmed.csv')

covid_dataset.head()
covid_dataset.shape
covid_dataset.drop(['Lat','Long'], axis=1, inplace=True)

covid_dataset.head()
covid_aggregated = covid_dataset.groupby('Country/Region').sum()
covid_aggregated.head()
covid_aggregated.shape
covid_aggregated.loc['India'].plot()

covid_aggregated.loc['China'].plot()

covid_aggregated.loc['US'].plot()

plt.legend()
covid_aggregated.loc['India'].plot()
covid_aggregated.loc['India'].diff().plot()
covid_aggregated.loc['India'].diff().max()
covid_aggregated.loc['China'].diff().max()
covid_aggregated.loc['US'].diff().max()
countries = list(covid_aggregated.index)

max_infection_rates = []

for country in countries :

    max_infection_rates.append(covid_aggregated.loc[country].diff().max())

covid_aggregated['Maximum Infection Rate'] = max_infection_rates
covid_aggregated.head()
covid_data = pd.DataFrame(covid_aggregated['Maximum Infection Rate'])

covid_data
happiness_report = pd.read_csv('../input/world-happiness/2019.csv')

happiness_report.head()
happiness_report.shape
columns_to_drop = ['Overall rank','Score','Generosity','Perceptions of corruption']

happiness_report.drop(columns_to_drop, axis=1, inplace=True)

happiness_report.head()
happiness_report.set_index('Country or region', inplace=True)

happiness_report.head()
covid_data.head()
happiness_report.head()
data = happiness_report.join(covid_data).copy()

data.head()
data.corr()

# it is representing the currelation between every two columns of our dataset
data.head()
x = data['GDP per capita']

y = data['Maximum Infection Rate']

sns.scatterplot(x,np.log(y))
sns.regplot(x,np.log(y))
x = data['Social support']

y = data['Maximum Infection Rate']

sns.scatterplot(x,np.log(y))
sns.regplot(x,np.log(y))
x = data['Healthy life expectancy']

y = data['Maximum Infection Rate']

sns.scatterplot(x,np.log(y))
sns.regplot(x,np.log(y))
x = data['Freedom to make life choices']

y = data['Maximum Infection Rate']

sns.scatterplot(x,np.log(y))
sns.regplot(x,np.log(y))