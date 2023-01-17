# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
data = pd.read_csv('/kaggle/input/ds4g-environmental-insights-explorer/eie_data/gppd/gppd_120_pr.csv')
data.columns
data = data.drop(['system:index','country','wepp_id','url','.geo','country_long','geolocation_source','name','gppd_idnr'],axis = 1)
data.tail()
data['total_generation'] = data['generation_gwh_2013'] + data['generation_gwh_2014'] + data['generation_gwh_2015'] + data['generation_gwh_2016'] + data['generation_gwh_2017'] 
data = data.drop(['generation_gwh_2013','generation_gwh_2014','generation_gwh_2015','generation_gwh_2016','generation_gwh_2017'],axis = 1)
data
import seaborn as sns
ax = sns.barplot(x="primary_fuel", y="estimated_generation_gwh", data=data)
ax = sns.barplot(x="estimated_generation_gwh", y="source", data=data)
ax = sns.scatterplot(x="commissioning_year", y="estimated_generation_gwh", data=data)
ax = sns.scatterplot(x="capacity_mw", y="estimated_generation_gwh", data=data)
the_mean = data.groupby('primary_fuel').mean()
from sklearn.linear_model import LinearRegression

import matplotlib.pyplot as plot
X = the_mean['capacity_mw']

y = the_mean['estimated_generation_gwh']
reg = LinearRegression()

reg = reg.fit(X.values.reshape(-1,1),y)

predictions = reg.predict(X.values.reshape(-1,1))
plot.scatter(X, y, color = 'red')

plot.plot(X, predictions, color = 'blue')

plot.title('RelationShip between Capasity and expected gwh')

plot.xlabel('Capasity')

plot.ylabel('Expected gwh')

plot.show()
from sklearn.metrics import r2_score

r2_score(y,predictions)
the_mean = the_mean.sort_values('capacity_mw')
the_mean['weights'] = list(the_mean['estimated_generation_gwh'] / the_mean['capacity_mw'])
the_mean['weights'] = the_mean['weights'] / the_mean['weights'].sum() * 1000
the_mean['the_adjusted_ratio'] = the_mean['weights'] * the_mean['capacity_mw']
the_mean
X = the_mean['the_adjusted_ratio']

y = the_mean['estimated_generation_gwh']
reg = LinearRegression()

reg = reg.fit(X.values.reshape(-1,1),y)

predictions = reg.predict(X.values.reshape(-1,1))
plot.scatter(X, y, color = 'red')

plot.plot(X, predictions, color = 'blue')

plot.title('RelationShip between Capasity and expected gwh')

plot.xlabel('Capasity')

plot.ylabel('Expected gwh')

plot.show()
from sklearn.metrics import r2_score

r2_score(y,predictions)
dictionary_of_coefficients = dict(zip(the_mean.index, the_mean['the_adjusted_ratio']))
dictionary_of_coefficients
data
for index,row in data.iterrows():

    for name in data.primary_fuel.unique():

        if row['primary_fuel'] == name:

            data.loc[index,'predicted_gwh'] = (dictionary_of_coefficients[name] * row['capacity_mw'])

            

from sklearn.metrics import mean_squared_error

mean_squared_error(data['estimated_generation_gwh'], data['predicted_gwh'])
ax = sns.scatterplot(x="estimated_generation_gwh", y="predicted_gwh", data=data)
from sklearn.metrics import r2_score

r2_score(data['estimated_generation_gwh'], data['predicted_gwh'])
dictionary_of_coefficients
data
the_mean = data.groupby('primary_fuel').mean()
the_mean['new_ratios'] = the_mean['predicted_gwh'] / the_mean['estimated_generation_gwh']
the_mean
dictionary_of_coefficients
for name in the_mean.index:

    dictionary_of_coefficients[name] =  dictionary_of_coefficients[name] / the_mean.loc[name,'new_ratios']
dictionary_of_coefficients
for index,row in data.iterrows():

    for name in data.primary_fuel.unique():

        if row['primary_fuel'] == name:

            data.loc[index,'predicted_gwh'] = (dictionary_of_coefficients[name] * row['capacity_mw'])

            

from sklearn.metrics import mean_squared_error

mean_squared_error(data['estimated_generation_gwh'], data['predicted_gwh'])
from sklearn.metrics import r2_score

r2_score(data['estimated_generation_gwh'], data['predicted_gwh'])
data
import pandas as pd

global_power_plant_database = pd.read_csv("../input/global-power-plant-database/global_power_plant_database.csv")
emmission_annual = pd.read_excel('/kaggle/input/emission-annual/emission_annual.xls')
emmission_mean = emmission_annual.groupby('Energy Source').mean()
data = pd.concat([global_power_plant_database[['capacity_mw','fuel1']],data])
for name in dictionary_of_coefficients:

    data.loc[data['fuel1'] == name,'predicted_gwh'] = (dictionary_of_coefficients[name] * data.loc[data['fuel1'] == name,'capacity_mw'])

    
dictionary_of_coefficients
average_co2_emissions = {'Coal':909,'Hydro':4,'Solar':105,'Wind':13,'Gas':465,'Oil':821}
for name in dictionary_of_coefficients:

    data.loc[data['fuel1'] == name,'co2_emissions'] = (average_co2_emissions[name] * data.loc[data['fuel1'] == name,'predicted_gwh']) * 1000000
data.head(30)
data['co2_emissions'].plot(figsize = (15,10))
data.sort_values('co2_emissions')['co2_emissions'].plot(use_index=False,figsize=(15,10))
data['co2_emissions'].sum() / 1000000