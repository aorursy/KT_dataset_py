# import libraries for data analysis

import numpy as np

import pandas as pd



# import libraries for data visualization

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline

sns.set_style(style = 'whitegrid', rc = {'xtick.bottom': True, 'ytick.left': True})
# read data from the dataset into dataframes

age_group_details = pd.read_csv('/kaggle/input/covid19-in-india/AgeGroupDetails.csv')

covid_19_india = pd.read_csv('/kaggle/input/covid19-in-india/covid_19_india.csv')

hospital_beds_india = pd.read_csv('/kaggle/input/covid19-in-india/HospitalBedsIndia.csv')

icmr_testing_details = pd.read_csv('/kaggle/input/covid19-in-india/ICMRTestingDetails.csv')

individual_details = pd.read_csv('/kaggle/input/covid19-in-india/IndividualDetails.csv')

population_india_census_2011 = pd.read_csv('/kaggle/input/covid19-in-india/population_india_census2011.csv')

statewise_testing_details = pd.read_csv('/kaggle/input/covid19-in-india/StatewiseTestingDetails.csv')



# explore each of the dataframes (below)
age_group_details.head()
covid_19_india.head()
hospital_beds_india.head()
icmr_testing_details.head()
individual_details.head()
population_india_census_2011.head()
statewise_testing_details.head()
# update the state name for Telangana

population_india_census_2011['State / Union Territory'] = population_india_census_2011['State / Union Territory'].apply(lambda name : 'Telangana' if name == 'Telengana' else name)



# update the district name for Ahmedabad

individual_details['detected_district'] = individual_details['detected_district'].apply(lambda name : 'Ahmedabad' if name == 'Ahmadabad' else name)



# udpate the city name for Ahmedabad

individual_details['detected_city'] = individual_details['detected_city'].apply(lambda name : 'Ahmedabad' if name == 'Ahmadabad' else name)
date_cumulative = covid_19_india.groupby('Date').sum()

date_cumulative.reset_index(inplace = True)



# change the date format to 'YYYY-MM-DD'

date_cumulative['Date'] = date_cumulative['Date'].apply(lambda date : '20' + '-'.join(date.split('/')[::-1]))



# sort the rows by date (in ascending order)

date_cumulative.sort_values('Date', inplace = True)



# calculate the number of active cases

date_cumulative['Active'] = date_cumulative['Confirmed'] - (date_cumulative['Cured'] + date_cumulative['Deaths'])



date_cumulative = date_cumulative[['Date', 'Confirmed', 'Cured', 'Deaths', 'Active']]



# dataframe with number of cases (categorised by current health status) grouped by date [CUMULATIVE]

date_cumulative.head()
plt.figure(figsize=(20, 8), dpi = 100)

sns.lineplot(x = 'Date', y = 'Confirmed', data = date_cumulative, label = 'Confirmed', color = 'blue', marker = 'o')

sns.lineplot(x = 'Date', y = 'Cured', data = date_cumulative, label = 'Cured', color = 'green')

sns.lineplot(x = 'Date', y = 'Deaths', data = date_cumulative, label = 'Deaths', color = 'black') 

sns.lineplot(x = 'Date', y = 'Active', data = date_cumulative, label = 'Active', color = 'red')

plt.title('Cumulative number of cases (categorised by current health status) grouped by date')

plt.ylabel('Number of cases')

plt.legend(loc = 0)

plt.xticks(rotation = 90)

plt.tight_layout()

plt.show()
plt.figure(figsize=(8, 5), dpi = 100)

sns.barplot(x = 'AgeGroup', y = 'TotalCases', data = age_group_details, palette = 'icefire')

plt.title('Total positive cases in each age group')

plt.xlabel('Age group')

plt.ylabel('Total positive cases')

plt.tight_layout()

plt.show()
temp_state = covid_19_india.groupby('State/UnionTerritory').max()

temp_state.reset_index(inplace = True)

temp_state['Active'] = temp_state['Confirmed'] - (temp_state['Cured'] + temp_state['Deaths'])

temp_state = temp_state[['State/UnionTerritory', 'Confirmed', 'Cured', 'Deaths', 'Active']]

temp_state.rename(columns = {'State/UnionTerritory': 'State / Union Territory'}, inplace = True)

temp_state['State / Union Territory'] = temp_state['State / Union Territory'].apply(lambda name : 'Telangana' if name == 'Telengana' else name)



# calculate the number of cases (categorised by current health status) for India and add them to the dataframe

# index number 41 is chosen to avoid any loss of data (as of now, the total number of states and UTs in India is 28 + 8 = 36)

temp_state.loc[41] = ['India', temp_state['Confirmed'].sum(), temp_state['Cured'].sum(), temp_state['Deaths'].sum(), temp_state['Active'].sum()]

 

temp_state.head()
# calculate India's total area

total_area_india_km2 = population_india_census_2011['Area'].apply(lambda area : float(area.split('\xa0')[0].replace(',', ''))).sum()



temp_population = population_india_census_2011[['State / Union Territory', 'Population', 'Density']]

temp_population['Population density (per km2)'] = temp_population['Density'].apply(lambda density : float(density.split('/')[0].replace(',', '')))

temp_population.drop(columns = ['Density'], inplace = True)



# calculate India's total population

total_population_india = temp_population['Population'].sum()



# calculate India's total population density

density_india = total_population_india / total_area_india_km2



temp_population.loc[41] = ['India', total_population_india, density_india]

temp_population.head()
statewise_data = pd.merge(left = temp_state, right = temp_population, on = 'State / Union Territory', how = 'inner')



# select India's data from the merged dataframe and storing it as a series in a variable

india_data = statewise_data.iloc[-1]



# drop India's data stored in the last row of the dataframe

statewise_data.drop(statewise_data.tail(1).index, inplace = True)



# store India's data in a row with index number 41 in the dataframe

statewise_data.loc[41] = india_data



# dataframe with important data from each state/union territory as well as India

statewise_data.head()
statewise_data.drop(41)[['State / Union Territory', 'Cured', 'Deaths', 'Active']].plot.bar(x = 'State / Union Territory', stacked = True, figsize = (15, 9))

plt.title('Total cases in Indian states and UTs')

plt.ylabel('Total cases')

plt.tight_layout()

plt.show()
total_cases_national = statewise_data.loc[41]['Confirmed']

active_national = statewise_data.loc[41]['Active']

cured_national = statewise_data.loc[41]['Cured']

deaths_national = statewise_data.loc[41]['Deaths']

print('NUMBER OF CASES IN INDIA\n')

print(f'Total: {total_cases_national}')

print(f'Active: {active_national}')

print(f'Cured: {cured_national}')

print(f'Deaths: {deaths_national}')
datewise_count = individual_details[individual_details['current_status'] != 'Migrated']

datewise_count = datewise_count.groupby(['diagnosed_date', 'current_status']).count()['id'].unstack()

datewise_count.fillna(0, inplace = True)

datewise_count.reset_index(inplace = True)



# change the date format to 'YYYY-MM-DD'

datewise_count['diagnosed_date'] = datewise_count['diagnosed_date'].apply(lambda date : '-'.join(date.split('/')[::-1]))



datewise_count.sort_values('diagnosed_date', inplace = True)

datewise_count.rename(columns = {'Deceased': 'Deaths', 'Hospitalized': 'Active', 'Recovered': 'Cured'}, inplace = True)



# calculate the total number of confirmed cases

datewise_count['Confirmed'] = datewise_count['Active'] + datewise_count['Cured'] + datewise_count['Deaths']



datewise_count['Deaths'] = datewise_count['Deaths'].apply(lambda num : int(num))

datewise_count['Active'] = datewise_count['Active'].apply(lambda num : int(num))

datewise_count['Cured'] = datewise_count['Cured'].apply(lambda num : int(num))

datewise_count['Confirmed'] = datewise_count['Confirmed'].apply(lambda num : int(num))

datewise_count = datewise_count[['diagnosed_date', 'Confirmed', 'Deaths', 'Cured', 'Active']]



# dataframe with number of cases (categorised by current health status) grouped by date [NON-CUMULATIVE]

datewise_count.head()
datewise_count.drop(columns = ['Confirmed']).plot.bar(x = 'diagnosed_date', stacked = True, figsize = (18, 8))

plt.title('Non-cumulative number of cases (categorised by current health status) grouped by date')

plt.xlabel('Date')

plt.ylabel('Number of cases')

plt.tight_layout()

plt.show()
# calculate total cases per million people for each state/union territory and India

statewise_data['Total cases per million people'] = statewise_data['Confirmed'] / statewise_data['Population'] * pow(10, 6)



# calculate recovery rate (%) for each state/union territory and India

statewise_data['Recovery rate (%)'] = statewise_data['Cured'] / statewise_data['Confirmed'] * 100



# calculate deaths per million people for each state/union territory and India

statewise_data['Deaths per million people'] = statewise_data['Deaths'] / statewise_data['Population'] * pow(10, 6)



# calculate fatality rate (%) for each state/union territory and India

statewise_data['Fatality rate (%)'] = statewise_data['Deaths'] / statewise_data['Confirmed'] * 100



statewise_data.head()
plt.figure(figsize=(11, 7), dpi = 100)

sns.barplot(x = 'State / Union Territory', y = 'Total cases per million people', data = statewise_data, palette = 'viridis')

plt.title('Total cases per million people in India and each of its states and UTs')

plt.xticks(rotation = 90)

plt.tight_layout()

plt.show()
plt.figure(figsize=(11, 7), dpi = 100)

sns.barplot(x = 'State / Union Territory', y = 'Recovery rate (%)', data = statewise_data, palette = 'viridis')

plt.title('Recovery rate (%) in India and each of its states and UTs')

plt.xticks(rotation = 90)

plt.tight_layout()

plt.show()
plt.figure(figsize=(11, 7), dpi = 100)

sns.barplot(x = 'State / Union Territory', y = 'Deaths per million people', data = statewise_data, palette = 'viridis')

plt.title('Deaths per million people in India and each of its states and UTs')

plt.xticks(rotation = 90)

plt.tight_layout()

plt.show()
plt.figure(figsize=(11, 7), dpi = 100)

sns.barplot(x = 'State / Union Territory', y = 'Fatality rate (%)', data = statewise_data, palette = 'viridis')

plt.title('Fatality rate (%) in India and each of its states and UTs')

plt.xticks(rotation = 90)

plt.tight_layout()

plt.show()
# create a list of the Indian states where positive COVID-19 cases have been reported

list_of_states = list(individual_details['detected_state'].unique())



# create a list of nationalities of all the positive COVID-19 cases reported in India

list_of_nations = list(individual_details[individual_details['nationality'].isnull() == False]['nationality'].unique())



temp_age = individual_details[['age', 'detected_state', 'nationality', 'current_status']]

for i in temp_age.index:

    if temp_age.loc[i]['detected_state'] in list_of_states and temp_age.loc[i]['nationality'] not in list_of_nations:

        # assign nationality as 'India', if detected state is in the list of Indian states but nationality is not in the list of reported nationalities

        temp_age.loc[i]['nationality'] = 'India'

        

# select only those cases whose nationality is mentioned as 'India'

temp_age = temp_age[temp_age['nationality'] == 'India']



# drop the rows where age is mentioned as 'F' or 'M'

temp_age = temp_age[(temp_age['age'] != 'F') & (temp_age['age'] != 'M')]



# drop the rows where age is not mentioned

temp_age.dropna(subset = ['age'], inplace = True)



# if age is mentioned as '28-35', update it with the mean value, i.e. 31 (the actual mean value is 31.5, but since we want to assign a value for age, we use 31)

# typecast age given as a string into a floating point number first, and then into an integer - for instances where age is mentioned as a floating-point string

temp_age['age'] = temp_age['age'].apply(lambda age : 31 if age == '28-35' else int(float(age)))



temp_age.head()
plt.figure(figsize=(13, 7), dpi = 100)

sns.boxplot(x = 'detected_state', y = 'age', data = temp_age, palette = 'Set1')

plt.title('Age distribution of positive cases in Indian states and UTs')

plt.xlabel('State / Union Territory')

plt.ylabel('Age')

plt.xticks(rotation = 90)

plt.tight_layout()

plt.show()
plt.figure(figsize=(5, 5), dpi = 100)

sns.boxplot(x = 'nationality', y = 'age', data = temp_age, palette = 'viridis')

plt.title('Age distribution of positive cases in India')

plt.xlabel('')

plt.ylabel('Age')

plt.tight_layout()

plt.show()
print('NATIONAL DESCRIPTIVE STATISTICS (Ages of all positive cases in India)\n')

print(temp_age.describe()['age'])
print('NATIONAL DESCRIPTIVE STATISTICS (Ages of positive cases in India, grouped by current health status)\n')

print(temp_age.groupby('current_status').describe()['age'])
plt.figure(figsize=(6, 5), dpi = 100)

sns.heatmap(statewise_data[['Confirmed', 'Population', 'Population density (per km2)']].corr(), cmap = 'viridis_r')

plt.yticks(rotation = 0)

plt.tight_layout()

plt.show()
sns.jointplot(x = 'Population', y = 'Confirmed', data = statewise_data.drop(41), kind = 'reg', color = 'crimson')

plt.ylabel('Total positive cases')

plt.tight_layout()

plt.show()
sns.jointplot(x = 'Population density (per km2)', y = 'Confirmed', data = statewise_data.drop(41), kind = 'reg', color = 'navy')

plt.ylabel('Total positive cases')

plt.tight_layout()

plt.show()
hospital_details = hospital_beds_india[['State/UT', 'TotalPublicHealthFacilities_HMIS', 'NumPublicBeds_HMIS']]

hospital_details.rename(columns = {'State/UT': 'State / Union Territory'}, inplace = True)

hospital_details['State / Union Territory'] = hospital_details['State / Union Territory'].apply(lambda name : str(name).replace('&', 'and'))

hospital_details.dropna(inplace = True)

hospital_details['TotalPublicHealthFacilities_HMIS'] = hospital_details['TotalPublicHealthFacilities_HMIS'].apply(lambda count : int(str(count).replace(',', '')))

hospital_details['NumPublicBeds_HMIS'] = hospital_details['NumPublicBeds_HMIS'].apply(lambda count : int(str(count).replace(',', '')))



# merge (or, add) the details for Dadra and Nagar Haveli & Daman and Diu (as the former UTs have been merged into a single UT) and update the dataframe

dnhdd = hospital_details[(hospital_details['State / Union Territory'] == 'Dadra and Nagar Haveli') | (hospital_details['State / Union Territory'] == 'Daman and Diu')].sum()

hospital_details.drop(index = [7, 8, 36], inplace = True)

hospital_details.loc[36] = ['Dadra and Nagar Haveli and Daman and Diu', dnhdd['TotalPublicHealthFacilities_HMIS'], dnhdd['NumPublicBeds_HMIS']]



hospital_details.sort_values('State / Union Territory', inplace = True)

hospital_details.loc[37] = ['India', hospital_details['TotalPublicHealthFacilities_HMIS'].sum(), hospital_details['NumPublicBeds_HMIS'].sum()]

hospital_details = pd.merge(left = hospital_details, right = temp_population.drop(columns = ['Population density (per km2)']), on = 'State / Union Territory', how = 'inner')



# calculate public health facility details per 1000 people for each state/UT as well as India

hospital_details['TotalPublicHealthFacilities/1000 people'] = (hospital_details['TotalPublicHealthFacilities_HMIS'] / hospital_details['Population']) * 1000

hospital_details['NumPublicBeds/1000 people'] = (hospital_details['NumPublicBeds_HMIS'] / hospital_details['Population']) * 1000



# dataframe with public health facility details for each state/UT and India

hospital_details.head()
plt.figure(figsize=(13, 8), dpi = 100)

sns.barplot(x = 'State / Union Territory', y = 'Population', data = hospital_details.drop(35), palette = 'viridis')

plt.title('Population of each state/UT')

plt.xticks(rotation = 90)

plt.tight_layout()

plt.show()
plt.figure(figsize=(13, 8), dpi = 100)

sns.barplot(x = 'State / Union Territory', y = 'TotalPublicHealthFacilities_HMIS', data = hospital_details.drop(35), palette = 'viridis')

plt.title('Total public health facilities in each state/UT')

plt.ylabel('Total public health facilities')

plt.xticks(rotation = 90)

plt.tight_layout()

plt.show()
plt.figure(figsize=(13, 8), dpi = 100)

sns.barplot(x = 'State / Union Territory', y = 'NumPublicBeds_HMIS', data = hospital_details.drop(35), palette = 'viridis')

plt.title('Number of public beds in each state/UT')

plt.ylabel('Number of public beds')

plt.xticks(rotation = 90)

plt.tight_layout()

plt.show()
plt.figure(figsize=(13, 8), dpi = 100)

sns.barplot(x = 'State / Union Territory', y = 'TotalPublicHealthFacilities/1000 people', data = hospital_details, palette = 'viridis')

plt.title('Total public health facilities per 1000 people in each state/UT and India')

plt.ylabel('Total public health facilities per 1000 people')

plt.xticks(rotation = 90)

plt.tight_layout()

plt.show()
plt.figure(figsize=(13, 8), dpi = 100)

sns.barplot(x = 'State / Union Territory', y = 'NumPublicBeds/1000 people', data = hospital_details, palette = 'viridis')

plt.title('Number of public beds per 1000 people in each state/UT and India')

plt.ylabel('Number of public beds per 1000 people')

plt.xticks(rotation = 90)

plt.tight_layout()

plt.show()
plt.figure(figsize=(8, 7), dpi = 100)

sns.heatmap(hospital_details.corr(), cmap = 'viridis_r')

plt.tight_layout()

plt.show()
testing_details = icmr_testing_details[['TotalSamplesTested', 'TotalPositiveCases']]



# change the date format to 'YYYY-MM-DD'

testing_details['Date'] = icmr_testing_details['DateTime'].apply(lambda dt : '20' + '-'.join(dt.split(' ')[0].split('/')[::-1]))



testing_details = testing_details[['Date', 'TotalSamplesTested', 'TotalPositiveCases']]



# dataframe with date-wise testing details provided by the Indian Council of Medical Research (ICMR)

testing_details.head() 
plt.figure(figsize=(16, 6), dpi = 100)

sns.lineplot(x = 'Date', y = 'TotalSamplesTested', data = testing_details, label = 'Total samples tested', color = 'navy', marker = 'o') 

sns.lineplot(x = 'Date', y = 'TotalPositiveCases', data = testing_details, label = 'Total positive cases', color = 'crimson', marker = 'o')

plt.title('Indian Council of Medical Research (ICMR) testing details')

plt.ylabel('Count')

plt.legend(loc = 0)

plt.xticks(rotation = 90)

plt.tight_layout()

plt.show()
sns.jointplot(x = 'TotalSamplesTested', y = 'TotalPositiveCases', data = testing_details, kind = 'reg', color = 'r')

plt.xlabel('Total samples tested')

plt.ylabel('Total positive cases')

plt.xticks(rotation = 90)

plt.tight_layout()

plt.show()
# series: states/UTs (highest number of confirmed COVID-19 cases)

states_highest = statewise_data.groupby('State / Union Territory').mean()['Confirmed'].sort_values(ascending = False).drop(labels = ['India']).head(10)



# series: states/UTs (lowest number of confirmed COVID-19 cases)

states_lowest = statewise_data.groupby('State / Union Territory').mean()['Confirmed'].sort_values().head(10)



# series: districts (highest number of confirmed COVID-19 cases)

districts_highest = individual_details['detected_district'].value_counts().head(10)
plt.figure(figsize=(8, 5), dpi = 100)

sns.barplot(x = states_highest.values, y = states_highest.index, palette = 'viridis')

plt.title('TOP 10 STATES/UTs WITH THE HIGHEST NUMBER OF CONFIRMED COVID-19 CASES')

plt.xlabel('Total number of confirmed cases')

plt.ylabel('State/Union territory')

plt.tight_layout()

plt.show()
print('TOP 10 STATES/UTs WITH THE HIGHEST NUMBER OF CONFIRMED COVID-19 CASES\n')

print(states_highest)
plt.figure(figsize=(8, 5), dpi = 100)

sns.barplot(x = states_lowest.values, y = states_lowest.index, palette = 'viridis')

plt.title('TOP 10 STATES/UTs WITH THE LOWEST NUMBER OF CONFIRMED COVID-19 CASES')

plt.xlabel('Total number of confirmed cases')

plt.ylabel('State/Union territory')

plt.tight_layout()

plt.show()
print('TOP 10 STATES/UTs WITH THE LOWEST NUMBER OF CONFIRMED COVID-19 CASES\n')

print(states_lowest)
plt.figure(figsize=(8, 5), dpi = 100)

sns.barplot(x = districts_highest.values, y = districts_highest.index, palette = 'viridis')

plt.title('TOP 10 DISTRICTS WITH THE HIGHEST NUMBER OF CONFIRMED COVID-19 CASES')

plt.xlabel('Total number of confirmed cases')

plt.ylabel('District')

plt.tight_layout()

plt.show()
print('TOP 10 DISTRICTS WITH THE HIGHEST NUMBER OF CONFIRMED COVID-19 CASES\n')

print(districts_highest)
print('DISTRICTS WITH ONLY 1 CONFIRMED COVID-19 CASE\n')

for dist, cnt in zip(individual_details['detected_district'].value_counts().index, individual_details['detected_district'].value_counts().values):

    # check if there's only 1 positive case in the district and the district name doesn't contain '*', such as 'Other Region*', 'Other States*' and 'Italians*' 

    if cnt == 1 and '*' not in dist:

        print(dist)
date_cumulative.reset_index(inplace = True)

date_cumulative.drop(columns = ['index'], inplace = True)



# create a list to contain the date-wise growth rate (%) of confirmed cases; the growth rate for the first date available in the dataframe is asssigned a null value

growth_rate = [np.nan]



# calculate the growth rate for each date (except the first date) given in the dataframe

for i in range(1, len(date_cumulative['Date'])):

    growth_rate.append((date_cumulative.iloc[i]['Confirmed'] - date_cumulative.iloc[i - 1]['Confirmed']) / date_cumulative.iloc[i - 1]['Confirmed'] * 100)

    

# create a new column to store the growth rate for each date given in the dataframe

date_cumulative['Growth rate (%)'] = growth_rate



date_cumulative.head()
plt.figure(figsize=(20, 8), dpi = 100)

sns.lineplot(x = 'Date', y = 'Growth rate (%)', data = date_cumulative, color = 'navy', marker = 'o')

plt.title('Date-wise growth rate (%) of confirmed cases in India')

plt.ylabel('Growth rate (%) of confirmed cases')

plt.xticks(rotation = 90)

plt.tight_layout()

plt.show()
statewise_testing_details.rename(columns = {'State': 'State / Union Territory'}, inplace = True)

statewise_testing_details = statewise_testing_details.groupby('State / Union Territory').max()

statewise_testing_details.reset_index(inplace = True)



# add population of each state/UT to the existing dataframe to create a new merged dataframe

statewise_testing_details = pd.merge(left = statewise_testing_details[['State / Union Territory', 'TotalSamples', 'Positive']], 

                                     right = population_india_census_2011[['State / Union Territory', 'Population']], on = 'State / Union Territory', how = 'inner')



# calculate and add India's testing details to the dataframe

statewise_testing_details.iloc[-1] = ['India', statewise_testing_details['TotalSamples'].sum(), 

                                      statewise_testing_details['Positive'].sum(), statewise_testing_details['Population'].sum()]



# calculate total samples per million people for each state/union territory and India

statewise_testing_details['Total samples per million people'] = statewise_testing_details['TotalSamples'] / statewise_testing_details['Population'] * pow(10, 6)



# calculate positive cases per 1000 samples for each state/union territory and India

statewise_testing_details['Positive cases per 1000 samples'] = statewise_testing_details['Positive'] / statewise_testing_details['TotalSamples'] * 1000



statewise_testing_details.head()
plt.figure(figsize=(11, 7), dpi = 100)

sns.barplot(x = 'State / Union Territory', y = 'Total samples per million people', data = statewise_testing_details, palette = 'viridis')

plt.title('Total samples per million people in India and each of its states and UTs')

plt.xticks(rotation = 90)

plt.tight_layout()

plt.show()
plt.figure(figsize=(11, 7), dpi = 100)

sns.barplot(x = 'State / Union Territory', y = 'Positive cases per 1000 samples', data = statewise_testing_details, palette = 'viridis')

plt.title('Positive cases per 1000 samples in India and each of its states and UTs')

plt.xticks(rotation = 90)

plt.tight_layout()

plt.show()
# create a dataframe with important COVID-19 details for each Indian state/UT as well as India



# sort by total number of confirmed cases in non-ascending order

covid_19_details = statewise_data.sort_values('Confirmed', ascending = False)



# add serial number (as index) to the dataframe

covid_19_details['Sno'] = range(len(covid_19_details['State / Union Territory']))

covid_19_details.set_index('Sno', inplace = True)



# rearrange the columns in the dataframe

covid_19_details = covid_19_details[['State / Union Territory', 'Confirmed', 'Cured', 'Deaths', 'Active', 'Recovery rate (%)', 'Fatality rate (%)', 

                                     'Total cases per million people', 'Deaths per million people', 'Population', 'Population density (per km2)']]



print('IMPORTANT COVID-19 DETAILS OF INDIA AND EACH OF ITS STATES AND UNION TERRITORIES')

covid_19_details