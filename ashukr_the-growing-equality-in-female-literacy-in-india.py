import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
data = pd.read_csv('../input/Indicators.csv')
data.head(10)
countries = data['CountryName'].unique().tolist()
"India" in countries
data_India=data.loc[data['CountryName'] == "India"]
data_India.head()
literacy_indicator = 'Youth literacy rate'
literacy_mask = data_India["IndicatorName"].str.contains(literacy_indicator)
female_indicator = 'female'

male_indicator = 'male'
stage_literacy = data_India[literacy_mask]
female_mask = stage_literacy['IndicatorName'].str.contains(female_indicator)

male_mask = stage_literacy['IndicatorName'].str.contains(male_indicator)

stage_literacy_female = stage_literacy[female_mask]

stage_literacy_male = stage_literacy[male_mask& ~female_mask]
print(stage_literacy_female.shape)

print(stage_literacy_male.shape)
print(stage_literacy_female['Value'].min())

print(stage_literacy_male['Value'].min())
print(stage_literacy_female['Value'].max())

print(stage_literacy_male['Value'].max())
print(stage_literacy_female['Year'].min())

print(stage_literacy_male['Year'].min())
print(stage_literacy_female['Year'].max())

print(stage_literacy_male['Year'].max())
female_literacy_year = stage_literacy_female['Year'].values

female_literacy_percentage = stage_literacy_female['Value'].values

male_literacy_year = stage_literacy_male['Year'].values

male_literacy_percentage = stage_literacy_male['Value'].values
plt.plot(female_literacy_year,female_literacy_percentage)

plt.axis([1981,2012,40,95])

plt.xlabel('Year')

plt.ylabel(stage_literacy_female['IndicatorName'].iloc[0])

plt.title('Growth of female literacy rate in India')

plt.show()
plt.plot(male_literacy_year,male_literacy_percentage)

plt.axis([1981,2012,40,95])

plt.xlabel('Year')

plt.ylabel(stage_literacy_male['IndicatorName'].iloc[0])

plt.title('Growth of male literacy rate in India')

plt.show()
plt.plot(male_literacy_year,male_literacy_percentage,label='male literacy')

#plt.legend(['male literacy'])

plt.plot(female_literacy_year,female_literacy_percentage,label = 'female literacy')

#plt.legend(['female literacy'])

plt.axis([1981,2012,40,95])

plt.xlabel('Year')

plt.ylabel('youth literacy rate')

plt.title('Comparision of male and female Youth literacy rate in India')



legend = plt.legend(loc='upper left', shadow=True, fontsize='x-large')

plt.show()