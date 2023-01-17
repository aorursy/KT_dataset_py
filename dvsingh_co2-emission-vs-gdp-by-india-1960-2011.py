import pandas as pd

import numpy as np

import random

import matplotlib.pyplot as plt



# Reading the data from kaggle dataset

data = pd.read_csv('../input/Indicators.csv')

#data.describe()



hist_indicator = 'CO2 emissions \(metric'

stage = data[data['IndicatorName'].str.contains(hist_indicator)  & data['CountryCode'].str.contains('IND')]

#stage.head(10)





mask1 = data['IndicatorName'].str.contains(hist_indicator) 

mask2 = data['Year'].isin([2011])



# apply our mask

co2_2011 = data[mask1 & mask2]

co2_2011.head()



ig, ax = plt.subplots()



ax.annotate("IND",

            xy=(18, 5), xycoords='data',

            xytext=(18, 30), textcoords='data',

            arrowprops=dict(arrowstyle="->",

                            connectionstyle="arc3"),

            )

            

plt.hist(co2_2011['Value'], 10, normed=False, facecolor='green')



plt.xlabel(stage['IndicatorName'].iloc[0])

plt.ylabel('# of Countries')

plt.title('Histogram of CO2 Emissions Per Capita')



#plt.axis([10, 22, 0, 14])

plt.grid(True)

plt.show()



# GDP Data



mask1 = data['IndicatorName'].str.contains(hist_indicator) 

mask2 = data['CountryCode'].str.contains('IND')



# stage is just those indicators matching the USA for country code and CO2 emissions over time.

gdp_stage = data[mask1 & mask2]

#gdp_stage.head(2)



fig, axis = plt.subplots()

# Grid lines, Xticks, Xlabel, Ylabel



axis.yaxis.grid(True)

axis.set_title('CO2 Emissions vs. GDP \(per capita\)',fontsize=10)

axis.set_xlabel(gdp_stage['IndicatorName'].iloc[0],fontsize=10)

axis.set_ylabel(stage['IndicatorName'].iloc[0],fontsize=10)



X = gdp_stage['Value']

Y = stage['Value']



axis.scatter(X, Y)

plt.show()




