import matplotlib.pyplot as plt

import numpy as np

import sqlite3

import pandas as pd 

import folium

%matplotlib inline
Indicators = pd.read_csv('/kaggle/input/world-development-indicators/Indicators.csv')
Indicators.head(5)
hist_country = 'USA'

mortality_stage = []

mask3 = Indicators['IndicatorCode'].str.contains('SP.DYN.IMRT.IN') 

mask4 = Indicators['CountryCode'].str.contains(hist_country)



mortality_stage = Indicators[mask3 & mask4]
hist_indicator1 = 'GDP per capita \(constant 2005'

hist_country1 = 'USA'



mask1 = Indicators['IndicatorName'].str.contains(hist_indicator1) 

mask2 = Indicators['CountryCode'].str.contains(hist_country1)



gdp_stage1 = Indicators[mask1 & mask2]

mortality_stage.head(5)
gdp_stage1.head(5)
print("GDP Min Year = ", gdp_stage1['Year'].min(), "max: ", gdp_stage1['Year'].max())

print("Mortality min Year = ", mortality_stage['Year'].min(), "max: ", mortality_stage['Year'].max())
mortality_stage_trunc = mortality_stage[mortality_stage['Year'] < 2015]
print("Mortality min Year = ", mortality_stage_trunc['Year'].min(), "max: ", mortality_stage_trunc['Year'].max())
fig, axis = plt.subplots()

# Grid lines, Xticks, Xlabel, Ylabel



axis.yaxis.grid(True)

axis.set_title('Infant mortality rate vs. GDP \(per capita\)',fontsize=10)

axis.set_xlabel(gdp_stage1['IndicatorName'].iloc[0],fontsize=10)

axis.set_ylabel(mortality_stage_trunc['IndicatorName'].iloc[0],fontsize=10)



X = gdp_stage1['Value']

Y = mortality_stage_trunc['Value']



axis.scatter(X, Y)

plt.show()
np.corrcoef(gdp_stage1['Value'],mortality_stage_trunc['Value'])
hist_year = 2010

mask5 = Indicators['IndicatorCode'].str.contains('SP.DYN.IMRT.IN') 

mask6 = Indicators['Year'].isin([hist_year])



mortality_stage1 = Indicators[mask5 & mask6]
mortality_stage1.head()
len(mortality_stage1)
hist_indicator2 = 'GDP per capita \(constant 2005'

hist_year = 2010



mask7 = Indicators['IndicatorName'].str.contains(hist_indicator2) 

mask8 = Indicators['Year'].isin([hist_year])

gdp_stage2 = Indicators[mask7 & mask8]
gdp_stage2.head(5)
len(gdp_stage2)
gdp_mortality= mortality_stage1.merge(gdp_stage2, on='CountryCode', how='inner')
gdp_mortality.head()
len(gdp_mortality)
fig, axis = plt.subplots()



axis.yaxis.grid(True)

axis.set_title('Infant mortality vs. GDP \(per capita\)',fontsize=10)

axis.set_xlabel(gdp_mortality['IndicatorName_y'].iloc[0],fontsize=10)

axis.set_ylabel(gdp_mortality['IndicatorName_x'].iloc[0],fontsize=10)



X = gdp_mortality['Value_y']

Y = gdp_mortality['Value_x']



axis.scatter(X, Y)

plt.show()
plot_data = mortality_stage1[['CountryCode','Value']]

plot_data.head()
hist_indicator = mortality_stage1.iloc[0]['IndicatorName']
country_geo = 'https://raw.githubusercontent.com/python-visualization/folium/588670cf1e9518f159b0eee02f75185301327342/examples/data/world-countries.json'
map = folium.Map(location=[100, 0], zoom_start=1.5)
folium.Choropleth(geo_data=country_geo, data=plot_data,

             columns=['CountryCode', 'Value'],

             key_on='feature.id',

             fill_color='YlGnBu', fill_opacity=0.7, line_opacity=0.2,

             legend_name=hist_indicator).add_to(map)
map.save('plot_data.html')
from IPython.display import HTML

HTML('<iframe src=plot_data.html width=700 height=450></iframe>')