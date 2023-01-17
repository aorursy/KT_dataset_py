import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

from scipy import stats
import os

print(os.listdir('../input/'))

os.chdir('../input/')
waste_diversion_rate = pd.read_csv('waste_diversion_rate.txt')

citywide_energy_usage = pd.read_csv('citywide_energy_usage.txt')

low_carbon_commute = pd.read_csv('low_carbon_commute.txt')

outdoor_air_quality = pd.read_csv('outdoor_air_quality.txt')

planted_trees = pd.read_csv('planted_trees.txt')
waste_city_merge = waste_diversion_rate.merge(citywide_energy_usage, left_on='Year', right_on = 'Year')
waste_city_trees_merge = waste_city_merge.merge(planted_trees, left_on='Year', right_on = 'Year')
waste_city_trees_air_merge = waste_city_trees_merge.merge(outdoor_air_quality, left_on='Year', right_on = 'Year')
waste_city_trees_air_commute_merge = waste_city_trees_air_merge.merge(low_carbon_commute, left_on='Year', right_on = 'Year')
waste_city_trees_air_commute_merge.shape
waste_city_trees_air_merge.shape
waste_city_trees_merge.shape
waste_city_trees_air_commute_merge.plot(kind = 'scatter', x='Citywide Energy Usage', y = 'Yard Trees Planted')

x = waste_city_trees_air_commute_merge['Citywide Energy Usage']

y = waste_city_trees_air_commute_merge['Yard Trees Planted']





slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)

plt.plot(x, y, 'o', label='original data')

plt.plot(x, intercept + slope*x, 'r', label='fitted line')

plt.title('City-Wide Energy Usage vs. Yard Trees Planted')

plt.legend()

plt.show()

print("R-squared: %f" % r_value**2)







#plt.plot(np.unique(waste_city_trees_merge['Citywide Energy Usage']), np.poly1d(np.polyfit(waste_city_trees_merge['Citywide Energy Usage'], waste_city_trees_merge['Yard Trees Planted'], 1))(np.unique(waste_city_trees_merge['Citywide Energy Usage'])))
waste_city_trees_air_commute_merge.plot(kind = 'scatter', x='Citywide Energy Usage', y = 'Street Trees Planted')

x = waste_city_trees_air_commute_merge['Citywide Energy Usage']

y = waste_city_trees_air_commute_merge['Street Trees Planted']





slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)

plt.plot(x, y, 'o', label='original data')

plt.plot(x, intercept + slope*x, 'r', label='fitted line')

plt.title('City-wide Energy Usage vs. Street Trees Planted')

plt.legend()

plt.show()

print("R-squared: %f" % r_value**2)
waste_city_trees_air_commute_merge.plot(kind = 'scatter', x='Recycling', y = 'Yard Trees Planted')

x = waste_city_trees_air_commute_merge['Recycling']

y = waste_city_trees_air_commute_merge['Yard Trees Planted']





slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)

plt.plot(x, y, 'o', label='original data')

plt.plot(x, intercept + slope*x, 'r', label='fitted line')

plt.title('Recycling vs. Yard Trees Planted')

plt.legend()

plt.show()

print("R-squared: %f" % r_value**2)
waste_city_trees_air_commute_merge.plot(kind = 'scatter', x='Recycling', y = 'Street Trees Planted')

x = waste_city_trees_air_commute_merge['Recycling']

y = waste_city_trees_air_commute_merge['Street Trees Planted']





slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)

plt.plot(x, y, 'o', label='original data')

plt.plot(x, intercept + slope*x, 'r', label='fitted line')

plt.title('Recycling vs. Street Trees Planted')

plt.legend()

plt.show()

print("R-squared: %f" % r_value**2)
waste_city_trees_air_commute_merge.plot(kind = 'scatter', x='Good', y = 'Street Trees Planted')

x = waste_city_trees_air_commute_merge['Good']

y = waste_city_trees_air_commute_merge['Street Trees Planted']





slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)

plt.plot(x, y, 'o', label='original data')

plt.plot(x, intercept + slope*x, 'r', label='fitted line')

plt.title('Number of days with Good Air Quality vs. Street Trees Planted')

plt.legend()

plt.show()

print("R-squared: %f" % r_value**2)
waste_city_trees_air_commute_merge.plot(kind = 'scatter', x='Good', y = 'Yard Trees Planted')

x = waste_city_trees_air_commute_merge['Good']

y = waste_city_trees_air_commute_merge['Yard Trees Planted']





slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)

plt.plot(x, y, 'o', label='original data')

plt.plot(x, intercept + slope*x, 'r', label='fitted line')

plt.title('Number of days with Good Air Quality vs. Yard Trees Planted')

plt.legend()

plt.show()

print("R-squared: %f" % r_value**2)
waste_city_trees_air_commute_merge.plot(kind = 'scatter', x='Low Carbon Commute Percentage', y = 'Good')

x = waste_city_trees_air_commute_merge['Low Carbon Commute Percentage']

y = waste_city_trees_air_commute_merge['Good']





slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)

plt.plot(x, y, 'o', label='original data')

plt.plot(x, intercept + slope*x, 'r', label='fitted line')

plt.title('Low Carbon Commute Percentage vs. Number of days with Good Air Quality')

plt.legend()

plt.show()

print("R-squared: %f" % r_value**2)
waste_city_trees_air_commute_merge.plot(kind = 'scatter', x='Low Carbon Commute Percentage', y = 'Yard Trees Planted')

x = waste_city_trees_air_commute_merge['Low Carbon Commute Percentage']

y = waste_city_trees_air_commute_merge['Yard Trees Planted']





slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)

plt.plot(x, y, 'o', label='original data')

plt.plot(x, intercept + slope*x, 'r', label='fitted line')

plt.title('Low Carbon Commute Percentage vs. Yard Trees Planted')

plt.legend()

plt.show()

print("R-squared: %f" % r_value**2)
waste_city_trees_air_commute_merge.plot(kind = 'scatter', x='Low Carbon Commute Percentage', y = 'Recycling')

x = waste_city_trees_air_commute_merge['Low Carbon Commute Percentage']

y = waste_city_trees_air_commute_merge['Recycling']





slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)

plt.plot(x, y, 'o', label='original data')

plt.plot(x, intercept + slope*x, 'r', label='fitted line')

plt.title('Low Carbon Commute Percentage vs. Recycling')

plt.legend()

plt.show()

print("R-squared: %f" % r_value**2)
waste_city_trees_air_commute_merge.plot(kind = 'scatter', x='Landfill', y = 'Yard Trees Planted')

x = waste_city_trees_air_commute_merge['Landfill']

y = waste_city_trees_air_commute_merge['Yard Trees Planted']





slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)

plt.plot(x, y, 'o', label='original data')

plt.plot(x, intercept + slope*x, 'r', label='fitted line')

plt.title('Landfill Amount vs. Yard Trees Planted')

plt.legend()

plt.show()

print("R-squared: %f" % r_value**2)
waste_city_trees_merge.plot(kind = 'scatter', x='Citywide Energy Usage', y = 'Street Trees Planted')

x = waste_city_trees_merge['Citywide Energy Usage']

y = waste_city_trees_merge['Street Trees Planted']



slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)

plt.plot(x, y, 'o', label='original data')

plt.plot(x, intercept + slope*x, 'r', label='fitted line')

plt.title('City-wide Energy Usage vs. Street Trees Planted')

plt.legend()

plt.show()

print("R-squared: %f" % r_value**2)

print("p-value: %f" % p_value)

#Remove outlier datapoint

waste_city_trees_merge_2 = waste_city_trees_merge

waste_city_trees_merge_2 = waste_city_trees_merge_2.drop([3],axis = 0)
waste_city_trees_merge_2.plot(kind = 'scatter', x='Citywide Energy Usage', y = 'Street Trees Planted')

x = waste_city_trees_merge_2['Citywide Energy Usage']

y = waste_city_trees_merge_2['Street Trees Planted']



slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)

plt.plot(x, y, 'o', label='original data')

plt.plot(x, intercept + slope*x, 'r', label='fitted line')

plt.title('City-wide Energy Usage vs. Street Trees Planted (Without Outlier)')

plt.legend()

plt.show()

print("R-squared: %f" % r_value**2)

print("p-value: %f" % p_value)

waste_city_trees_air_commute_merge.plot(kind = 'scatter', x='Low Carbon Commute Percentage', y = 'Yard Trees Planted')

x = waste_city_trees_air_commute_merge['Low Carbon Commute Percentage']

y = waste_city_trees_air_commute_merge['Yard Trees Planted']





slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)

plt.plot(x, y, 'o', label='original data')

plt.plot(x, intercept + slope*x, 'r', label='fitted line')

plt.title('Low Carbon Commute Percentage vs. Yard Trees Planted')

plt.legend()

plt.show()

print("R-squared: %f" % r_value**2)

print("p-value: %f" % p_value)

#drop outlier datapoint

waste_city_trees_air_commute_merge_2 = waste_city_trees_air_commute_merge

waste_city_trees_air_commute_merge_2 = waste_city_trees_air_commute_merge_2.drop([6], axis = 0)
waste_city_trees_air_commute_merge_2.plot(kind = 'scatter', x='Low Carbon Commute Percentage', y = 'Yard Trees Planted')

x = waste_city_trees_air_commute_merge_2['Low Carbon Commute Percentage']

y = waste_city_trees_air_commute_merge_2['Yard Trees Planted']





slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)

plt.plot(x, y, 'o', label='original data')

plt.plot(x, intercept + slope*x, 'r', label='fitted line')

plt.title('Low Carbon Commute Percentage vs. Yard Trees Planted (Without Outlier)')

plt.legend()

plt.show()

print("R-squared: %f" % r_value**2)

print("p-value: %f" % p_value)
waste_city_trees_air_commute_merge.plot(kind = 'scatter', x='Landfill', y = 'Yard Trees Planted')

x = waste_city_trees_air_commute_merge['Landfill']

y = waste_city_trees_air_commute_merge['Yard Trees Planted']





slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)

plt.plot(x, y, 'o', label='original data')

plt.plot(x, intercept + slope*x, 'r', label='fitted line')

plt.title('Landfill Mass Added vs. Yard Trees Planted')

plt.legend()

plt.show()

print("R-squared: %f" % r_value**2)

print("p-value: %f" % p_value)