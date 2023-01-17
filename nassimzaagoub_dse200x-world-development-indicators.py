import pandas as pd

import numpy as np

import random

import matplotlib.pyplot as plt
d = pd.read_csv('../input/Indicators.csv')

data = pd.DataFrame(d)
data.shape
data.head()
pd.unique(data[['CountryCode']].values.ravel())
countryFilter = ['ARB', 'CSS', 'CEB', 'EAS', 'EAP', 'EMU', 'ECS', 'ECA', 'EUU',

       'FCS', 'HPC', 'HIC', 'NOC', 'OEC', 'LCN', 'LAC', 'LDC', 'LMY',

       'LIC', 'LMC', 'MEA', 'MNA', 'MIC', 'NAC', 'OED', 'OSS', 'PSS',

       'SST', 'SAS', 'SSF', 'SSA', 'UMC', 'WLD']
data[data['IndicatorName'].str.contains('GDP', na=False)].head()
data[data['IndicatorName'].str.contains('Population', na=False)].head(10)
data[data['IndicatorName'].str.contains('GINI', na=False)].head()
indicatorFilter = ['SP.POP.TOTL', 'SI.POV.GINI', 'NY.GDP.PCAP.CD']
data[data['IndicatorName'].str.contains('GINI', na=False)].groupby('Year').count()['CountryName']
yearsFilter = [2010]
filterMesh = (data['IndicatorCode'].isin(indicatorFilter)) & (~data['CountryCode'].isin (countryFilter)) & (data['Year'].isin(yearsFilter))

country_data = data.loc[filterMesh]

country_data.head()
p_data = country_data.pivot(index = 'CountryName', columns = 'IndicatorName', values = 'Value')

p_data.head()
p_data = p_data.dropna()

p_data.columns = [['GDP per capita (current US$)', 'GINI index (World Bank estimate)', 'Total Population']]

p_data.head()



#p_data.shape
%matplotlib inline

import matplotlib.pyplot as plt



fig, axis = plt.subplots()

# Grid lines, Xticks, Xlabel, Ylabel



axis.yaxis.grid(True)

axis.set_title('2010 - Income Inequality',fontsize=10)

axis.set_xlabel('GDP per capita (current US$)',fontsize=10)

axis.set_ylabel('GINI index (World Bank estimate)',fontsize=10)



X = p_data['GDP per capita (current US$)']

Y = p_data['GINI index (World Bank estimate)']



axis.scatter(X, Y)

plt.show()
p_data['GDP per capita (current US$)'].corr(p_data['GINI index (World Bank estimate)'])
%matplotlib inline

p_data.plot.scatter(x='GDP per capita (current US$)', y='GINI index (World Bank estimate)', 

                 s=p_data['Total Population']*.00001, grid = True,

                 yticks = (20, 30, 40, 50, 60),

                title = '2010 - Income Inequality', figsize=(15,10), fontsize=10)