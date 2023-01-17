import pandas as pd

import seaborn as sns

import numpy as np



import matplotlib.pyplot as plt

%matplotlib inline



import warnings

warnings.filterwarnings('ignore')
global_temp = pd.read_csv('../input/GlobalTemperatures.csv')
# drop unnecessary columns

global_temp = global_temp[['dt', 'LandAverageTemperature']]



global_temp['dt'] = pd.to_datetime(global_temp['dt'])

global_temp['year'] = global_temp['dt'].map(lambda x: x.year)

global_temp['month'] = global_temp['dt'].map(lambda x: x.month)



def get_season(month):

    if month >= 3 and month <= 5:

        return 'spring'

    elif month >= 6 and month <= 8:

        return 'summer'

    elif month >= 9 and month <= 11:

        return 'autumn'

    else:

        return 'winter'

    

min_year = global_temp['year'].min()

max_year = global_temp['year'].max()

years = range(min_year, max_year + 1)



global_temp['season'] = global_temp['month'].apply(get_season)



spring_temps = []

summer_temps = []

autumn_temps = []

winter_temps = []



for year in years:

    curr_years_data = global_temp[global_temp['year'] == year]

    spring_temps.append(curr_years_data[curr_years_data['season'] == 'spring']['LandAverageTemperature'].mean())

    summer_temps.append(curr_years_data[curr_years_data['season'] == 'summer']['LandAverageTemperature'].mean())

    autumn_temps.append(curr_years_data[curr_years_data['season'] == 'autumn']['LandAverageTemperature'].mean())

    winter_temps.append(curr_years_data[curr_years_data['season'] == 'winter']['LandAverageTemperature'].mean())
sns.set(style="whitegrid")

sns.set_color_codes("pastel")

f, ax = plt.subplots(figsize=(10, 6))



plt.plot(years, summer_temps, label='Summers average temperature', color='orange')

plt.plot(years, autumn_temps, label='Autumns average temperature', color='r')

plt.plot(years, spring_temps, label='Springs average temperature', color='g')

plt.plot(years, winter_temps, label='Winters average temperature', color='b')



plt.xlim(min_year, max_year)



ax.set_ylabel('Average temperature')

ax.set_xlabel('Year')

ax.set_title('Average temperature in each season')

legend = plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), frameon=True, borderpad=1, borderaxespad=1)
temp_by_country = pd.read_csv('../input/GlobalLandTemperaturesByCountry.csv')

countries = temp_by_country['Country'].unique()
max_min_list = []



# getting max and min temps

for country in countries:

    curr_temps = temp_by_country[temp_by_country['Country'] == country]['AverageTemperature']

    max_min_list.append((curr_temps.max(), curr_temps.min()))

    

# nan cleaning

res_max_min_list = []

res_countries = []



for i in range(len(max_min_list)):

    if not np.isnan(max_min_list[i][0]):

        res_max_min_list.append(max_min_list[i])

        res_countries.append(countries[i])



# calc differences        

differences = []



for tpl in res_max_min_list:

    differences.append(tpl[0] - tpl[1])

    

# sorting

differences, res_countries = (list(x) for x in zip(*sorted(zip(differences, res_countries), key=lambda pair: pair[0], reverse=True)))



# ploting

f, ax = plt.subplots(figsize=(8, 8))

sns.barplot(x=differences[:15], y=res_countries[:15], palette=sns.color_palette("coolwarm", 25), ax=ax)



texts = ax.set(ylabel="", xlabel="Temperature difference", title="Countries with the highest temperature differences")