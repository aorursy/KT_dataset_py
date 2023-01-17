import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.cm as cm
%matplotlib inline
dataset = pd.read_csv('../input/worlddevelpmentindicator/dataset.csv', skiprows=4)
dataset.head(5)
dataset.drop(['Country Code', 'Indicator Name', 'Indicator Code', '1960', '2016', 'Unnamed: 61'],
             axis = 1, inplace = True)
dataset.head(5)
dataset.dropna(how = 'any', axis = 0, inplace = True)
dataset.isnull().sum()
x = dataset.columns[1:]
y = dataset.iloc[0][1:]
country = dataset.iloc[0][0]
plt.plot(x, y)
plt.rcParams['figure.figsize'] = (30, 30)
plt.rcParams['font.size'] = '20'
plt.title('Population density of ' + country + ' over the years')
plt.xlabel('Years')
plt.xticks(rotation = '90')
plt.ylabel('Population density (People per sq. Km)')
plt.plot(x, y, linewidth = 4)
x = dataset.columns[1:]
colors = cm.rainbow(np.linspace(0, 1, 5))

for index in range(5):
    y = dataset.iloc[index][1:]
    plt.plot(x, 
             y, 
             c = colors[index],
             label = dataset.iloc[index][0],
             linewidth = 4)
    plt.title('Comparing population density of various conutries')
    plt.xlabel('Years')
    plt.xticks(rotation = '90')
    plt.ylabel('Population Density')
    plt.legend(prop = {'size': 24})
countries = dataset['Country Name']
populationDensity2015 = dataset['2015']
plt.xticks(rotation = '90')
plt.bar(countries, populationDensity2015, color = cm.rainbow(np.linspace(0, 1, len(countries))))
top10 = dataset.sort_values('2015', ascending = False).head(10)
plt.xticks(rotation = '45')
plt.title('Population Density of 10 most densely populated countries for the year 2015')
plt.xlabel('Countires')
plt.ylabel('Population Density')
plt.bar(top10['Country Name'],
        top10['2015'],
        color = cm.rainbow(np.linspace(0, 1, len(top10))))
total_columns = dataset.shape[1]
selected_data = dataset[dataset.sum(axis = 1).apply(lambda x: x/total_columns) <= 10]
consolidated_data = selected_data.sum(axis = 1).apply(lambda x: x/total_columns)
countries = selected_data['Country Name']
plt.rcParams['figure.figsize'] = (20, 20)
plt.rcParams['font.size'] = 14
plt.title('Average Population density for various countries')
plt.xlabel('Countries')
plt.ylabel('Average Population Density')
plt.xticks(rotation = '90')
plt.scatter(countries, 
            consolidated_data, 
            s = consolidated_data*20, 
            c = cm.rainbow(np.linspace(0, 1, len(countries))))
minimum = dataset.loc[:, dataset.columns != 'Country Name'].min()
maximum = dataset.loc[:, dataset.columns != 'Country Name'].max()
diff = maximum - minimum
minOfMax = maximum.min()
plt.title('Range of Population Density for years 1962-2015')
plt.xticks(rotation = '90')
plt.xlabel('Years')
plt.ylabel('Population Density')
plt.bar(dataset.columns[1:], diff.apply(lambda x: x-minOfMax), color = cm.rainbow(np.linspace(0, 1, dataset.shape[1])))
from urllib.request import urlopen
from bs4 import BeautifulSoup

content = BeautifulSoup(urlopen('https://en.wikipedia.org/wiki/List_of_countries_and_dependencies_by_area'),
                        'html.parser')
dataset['Area'] = 0.0
table = content.find_all('table')[0]
rows = table.find_all('tr')
for tr in rows:
    td = tr.find_all('td')
    a = tr.find('a')
    try:
        area = td[3].text.split('♠')[1].split('(')[0].replace(',', '')
        dataset.loc[dataset['Country Name'] == a.text, 'Area'] = int(area)
    except Exception:
        continue
dataset = dataset[dataset['Area'] != 0]
dataset.shape
population = dataset['Area'].multiply(dataset['2015'], axis = 0)
countries = dataset['Country Name']

plt.subplot(2,2,1)
plt.title('Population of various countries for year 2015')
plt.xlabel('Countries')
plt.ylabel('Population')
plt.xticks(rotation = '90')
plt.bar(countries[:20], population[:20], color = 'b')

plt.subplot(2,2,2)
plt.title('Population Density of various countries for year 2015')
plt.xlabel('Countries')
plt.ylabel('Population Density')
plt.xticks(rotation = '90')
plt.bar(countries[:20], dataset['2015'][:20], color = 'r')