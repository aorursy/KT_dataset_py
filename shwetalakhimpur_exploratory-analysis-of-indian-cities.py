import pandas as pd

import seaborn as sns

import numpy as np



import matplotlib.pyplot as plt

%matplotlib inline



import warnings

warnings.filterwarnings('ignore')



indian_cities = {'Ahmadabad', 'Bangalore' , 'Bombay' , 'Kanpur', 'Lakhnau', 'Nagpur', 'Madras','Pune', 'Calcutta' , 'Surat', 'New Delhi', 'Jaipur', 'Hyderabad'}
global_temp = pd.read_csv('../input/GlobalLandTemperaturesByMajorCity.csv')
# drop unnecessary columns

global_temp = global_temp[['dt', 'City', 'AverageTemperature']]



global_temp['dt'] = pd.to_datetime(global_temp['dt'])

global_temp['year'] = global_temp['dt'].map(lambda x: x.year)

global_temp['month'] = global_temp['dt'].map(lambda x: x.month)

global_temp['City'] = global_temp['City']



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

    spring_temps.append(curr_years_data[curr_years_data['season'] == 'spring']['AverageTemperature'].mean())

    summer_temps.append(curr_years_data[curr_years_data['season'] == 'summer']['AverageTemperature'].mean())

    autumn_temps.append(curr_years_data[curr_years_data['season'] == 'autumn']['AverageTemperature'].mean())

    winter_temps.append(curr_years_data[curr_years_data['season'] == 'winter']['AverageTemperature'].mean())
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
temp_by_majorcity = pd.read_csv('../input/GlobalLandTemperaturesByMajorCity.csv')

cities = temp_by_majorcity['City'].unique()



indian_cities = {'Ahmadabad', 'Bangalore' , 'Bombay' , 'Kanpur', 'Lakhnau', 'Nagpur', 'Madras','Pune', 'Calcutta' , 'Surat', 'New Delhi', 'Jaipur', 'Hyderabad'}
city_list = []

min_max_list = []



# getting max and min temps

for city in cities:

    if (city in indian_cities):

        ##print(city)

        curr_temps = temp_by_majorcity[temp_by_majorcity['City'] == city]['AverageTemperature']

        curr_temps_uncertain = temp_by_majorcity[temp_by_majorcity['City'] == city]['AverageTemperatureUncertainty']

        min_max_list.append((curr_temps.max(), curr_temps.min()))

        city_list.append(city)

    

# nan cleaning

res_min_max_list = []

res_cities = []



for i in range(len(min_max_list)):

    if not np.isnan(min_max_list[i][0]):

        res_min_max_list.append(min_max_list[i])

        res_cities.append(city_list[i])



# calc differences        

differences = []



for tpl in res_min_max_list:

    differences.append(tpl[0] - tpl[1])

    

# sorting

differences, res_cities = (list(x) for x in zip(*sorted(zip(differences, res_cities), key=lambda pair: pair[0], reverse=True)))



# ploting cities with temperature difference

f, ax = plt.subplots(figsize=(8, 8))

sns.barplot(x=differences[:15], y=res_cities[:14], palette=sns.color_palette("coolwarm", 25), ax=ax)



texts = ax.set(ylabel="", xlabel="Temperature difference", title="Cities with the highest temperature differences")
min_max_majorcity = pd.DataFrame(temp_by_majorcity)



max_majorcity = min_max_majorcity.groupby(['City']).max()['AverageTemperature']

max_majorcity = max_majorcity.to_frame().reset_index()

max_majorcity.columns = ['City','City Maximum']

max_majorcity = max_majorcity.loc[max_majorcity['City'].isin(indian_cities)]



f, ax = plt.subplots(figsize=(15, 8))

sns.barplot(x='City',y='City Maximum', data=max_majorcity.reset_index())

texts = ax.set(ylabel="MAX Temperature", xlabel="City", title="Cities maximum temperatures")
min_majorcity = min_max_majorcity.groupby(['City'],as_index=True).min()['AverageTemperature']

min_majorcity = min_majorcity.to_frame().reset_index()

min_majorcity.columns = ['City','City Minimum']

##print(min_majorcity)

min_majorcity = min_majorcity.loc[min_majorcity['City'].isin(indian_cities)]



f, ax = plt.subplots(figsize=(15, 8))

sns.barplot(x='City',y='City Minimum', data=min_majorcity.reset_index())

texts = ax.set(ylabel="MIN Temperature", xlabel="City", title="Cities minimum temperatures")
##Merging the two data sets

min_max_majorcity_merge = pd.merge(min_majorcity , max_majorcity)

min_max_majorcity_merge = min_max_majorcity_merge.set_index('City')

##print(min_max_majorcity_merge)



ax = min_max_majorcity_merge.plot(kind='bar',stacked=False, figsize=(15, 8))



texts = ax.set(ylabel="City Max-Min Temperature", xlabel="City", title="Cities Min MAX temperatures")
df = pd.read_csv('../input/GlobalLandTemperaturesByMajorCity.csv')



df_allIndianMajorCities = df.loc[df['City'].isin(indian_cities)]

df_allIndianMajorCities = df_allIndianMajorCities.ix[:, :4]

del df_allIndianMajorCities['AverageTemperatureUncertainty']

df_allIndianMajorCities
temp_year = df_allIndianMajorCities['dt'].apply(lambda x: int(x[0:4]))

##df_allIndianMajorCities

##df_allIndianMajorCities = df_allIndianMajorCities.iloc[:1900]##WORK HERE

##df_allIndianMajorCities.drop(df_allIndianMajorCities.loc[1:100],inplace=True)



group_by_year = df_allIndianMajorCities.groupby([temp_year,'City']).mean()

group_by_year
## Display bar chart with the data for different cities over the years

## This is taking a lot of time - make it more efficient
##group_by_year = group_by_year.set_index('dt')

##group_by_year



ax = group_by_year.plot(kind='bar',stacked=False, figsize=(20, 8))

texts = ax.set(ylabel="City Average Temperature", xlabel="Year", title="Cities Avg temperatures")