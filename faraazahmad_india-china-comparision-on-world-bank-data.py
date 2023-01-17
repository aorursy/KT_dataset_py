import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
population = pd.read_csv('../input/country_population.csv')
fertility_rate = pd.read_csv('../input/fertility_rate.csv')
expectancy = pd.read_csv('../input/life_expectancy.csv')
population.head()
india = population.loc[population['Country Name'] == 'India']
china = population.loc[population['Country Name'] == 'China']
x = range(1959, 2016)
size = india.size
y_india = india.iloc[0, 4:size]
y_china = china.iloc[0, 4:size]

plt.plot(x, y_india)
plt.plot(x, y_china)
plt.xlabel('Year (1960 - 2016)')
plt.ylabel('population (billions)')
plt.legend(['India', 'China'])
plt.show()
india_growth = []
for i in range(56):
    india_growth.append(y_india[i + 1] - y_india[i])

china_growth = []
for i in range(56):
    china_growth.append(y_china[i + 1] - y_china[i])
    
plt.plot(x[1:], india_growth)
plt.plot(x[1:], china_growth)
plt.legend(["India", "China"])
plt.show()
fertility_rate.head()
# fertility rate of china
china_fer = fertility_rate.loc[fertility_rate['Country Name'] == "China"].iloc[0, 4:size]
# fertility rate of India
india_fer = fertility_rate.loc[fertility_rate['Country Name'] == "India"].iloc[0, 4:size]
# comparing fertility rate of both countries
plt.plot(x, india_fer)
plt.plot(x, china_fer)
plt.xlabel('Year (1960 - 2016)')
plt.ylabel('child per woman')
plt.legend(['India', 'China'])
plt.show()
china_area = 9326410.0
india_area = 2973193.0
plt.plot(x, y_india/india_area)
plt.plot(x, y_china/china_area)
plt.xlabel('Year (1960 - 2016)')
plt.ylabel('population density (people per sq. km.)')
plt.legend(['India', 'China'])
plt.show()