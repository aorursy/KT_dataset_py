import pandas as pd 

import numpy as np

import seaborn as sns 

import matplotlib.pyplot as plt

import plotly.express as px

import pandas.util.testing as tm

import datetime



print('modules are imported.')
symptoms={'symptom':['Fever',

        'Dry cough',

        'Fatigue',

        'Sputum production',

        'Shortness of breath',

        'Muscle pain',

        'Sore throat',

        'Headache',

        'Chills',

        'Nausea or vomiting',

        'Nasal congestion',

        'Diarrhoea',

        'Haemoptysis',

        'Conjunctival congestion'],'percentage':[87.9,67.7,38.1,33.4,18.6,14.8,13.9,13.6,11.4,5.0,4.8,3.7,0.9,0.8]}



symptoms=pd.DataFrame(data=symptoms,index=range(14))

symptoms
fig = px.bar(symptoms[['symptom', 'percentage']].sort_values('percentage', ascending=False), 

             y="percentage", x="symptom", color='symptom', log_y=True, title='Symptom of  Coronavirus')

fig.show()
pd.read_csv('../input/coronavirusdataset/Case.csv')
patient = pd.read_csv('../input/coronavirusdataset/PatientInfo.csv')



patient.head()
patient.contact_number = patient.contact_number.replace('-', None)

patient_contact = patient[ ~patient.contact_number.isna() ]



# Convert numeric but str type values into int type 

patient_contact.contact_number = list(map(int, patient_contact.contact_number))



# 3) Drop unreasonably large values

patient_contact = patient_contact[ patient_contact.contact_number < 10000 ]



display(patient_contact.contact_number.describe())



fig, ax = plt.subplots(figsize=(13, 7))

plt.title('Patient Distribution by Number of Contacts Before Confirmed', fontsize=17)



sns.swarmplot(patient_contact.contact_number)



ax.set_xlabel('Number of contacts', size=13)

ax.set_ylabel('Patients', size=13)

plt.show()
age_raw = pd.read_csv('../input/coronavirusdataset/TimeAge.csv')



age_raw


age_list = age_raw.age.unique()

# Plot cases by age

fig, ax = plt.subplots(figsize=(13, 7))



sns.barplot(age_list, age_raw.confirmed[-9:])

ax.set_xlabel('age', size=13)

ax.set_ylabel('number of cases', size=13)



plt.show()
def plot_groupby(data, groupby, column, title, ylabel=None, axis=None):

    

    fig, ax = plt.subplots(figsize=(13, 7))

    plt.title(f'{title}', fontsize=17)

    ax.set_xlabel('Date', size=13)

    ax.set_ylabel('Number of cases', size=13)



    

    group_list = data.groupby(groupby)

    for group in group_list:

        if axis == None:

            sns.lineplot(group[1].date.values

                     , group[1][column].values

                     , label=group[0])

        else:

            sns.lineplot(group[1].date.values

                     , group[1][column].values

                     , label=group[0])

  

    dates_num = 12

    ax.set_xticks(ax.get_xticks()[::int(len(age_raw.date.unique())/dates_num)+1])

    ax.legend()

    plt.show()
plot_groupby(age_raw, 'age', 'confirmed', 'Confirmed Cases by Age (cumulative)')
plot_groupby(age_raw, 'age', 'deceased', 'Deceased Cases by Age (cumulative)')



age_deceased = age_raw.tail(9)[['age', 'deceased']]

age_deceased.set_index(np.arange(0, len(age_raw.age.unique())), inplace=True)

ax.legend()



print('Latest deceased cases')

display(age_deceased)
gender = pd.read_csv('../input/coronavirusdataset/TimeGender.csv')



gender
gender['sex'].value_counts()
fig, ax = plt.subplots(figsize=(13, 7))

plt.title('Confirmed Cases by Sex (cumulative)', fontsize=17)



sex_confirmed = (gender[gender.sex=='male'].confirmed, gender[gender.sex=='female'].confirmed)



for sex_each, sex_label in zip(sex_confirmed, ['male', 'female']):

    sns.lineplot(gender.date.unique(), sex_each, label=sex_label)

    

ax.set_xticks(ax.get_xticks()[::int(len(gender.date.unique())/8)])

plt.xlabel('Date')

plt.ylabel('Number of cases')



plt.show()
fig, ax = plt.subplots(figsize=(13, 7))

plt.title('Deceased cases by sex (cumulative)', fontsize=17)

sex_deceased = (gender[gender.sex=='male'].deceased, gender[gender.sex=='female'].deceased)



for sex_each, sex_label in zip(sex_deceased, ['male', 'female']):

    sns.lineplot(gender.date.unique(), sex_each, label=sex_label)

ax.set_xticks(ax.get_xticks()[::int(len(gender.date.unique())/8)])

plt.xlabel('Date')

plt.ylabel('Number of cases')



plt.show()
region = pd.read_csv('../input/coronavirusdataset/Region.csv')



region
print('Number of regions:', len(region.province.unique()))
region.describe()
print(region.province.unique())
elderly_pop = pd.DataFrame(region[region.province!='Korea'].groupby('province').mean().elderly_population_ratio

                           .sort_values(ascending=False))

elderly_pop
fig, ax = plt.subplots(figsize=(13, 7))

plt.title('Elderly population ratio by region', fontsize=17)



plt.xticks(rotation=30)

sns.barplot(elderly_pop.index, elderly_pop.elderly_population_ratio)



plt.xlabel('region')

plt.ylabel('Total population ratio (%)')



plt.show()
search = pd.read_csv('../input/coronavirusdataset/SearchTrend.csv')



search
fig, ax = plt.subplots(figsize=(13, 7))



plt.title('Search Trends Related to Respiratory Diseases (since 1st case worldwide)', size=17)

ax.set_xlabel('Date', size=13)

ax.set_ylabel('Relative in time range (%)', size=13)



for col in search.columns[1:]:

    sns.lineplot(search.date[search.date >= '2019-11-17']

             , search[search.date >= '2019-11-17'][col])

    

ax.set_xticks(ax.get_xticks()[::int(len(search.date[search.date >= '2019-11-17'])/8)])

ax.legend()



plt.show()
fig, ax1 = plt.subplots(figsize=(13, 7))

plt.title('Search Trends Related to Respiratory Diseases (since 1st case in S.Korea)', size=17)

ax1.set_xlabel('Date', size=13)

ax1.set_ylabel('Relative interests in time range (%)', size=13)

for column in search.columns[1:]:

    sns.lineplot(search[search.date >= '2020-01-20'].date

             , search[search.date >= '2020-01-20'][column])

ax1.set_xticks(ax1.get_xticks()[::int(len(search[search.date >= '2020-01-20'].date)/8)])

ax1.legend()



plt.show()
weather = pd.read_csv('../input/coronavirusdataset/Weather.csv')



weather.head()
print('Number of regions in weather data:', len(weather.province.unique()))

print(sorted(weather.province.unique()))

print('-----------------------------------------------------')

print('Number of regions in region data:', len(region.province.unique()))

print(sorted(region.province.unique()))
print('Average weather by region')

weather_status = weather.loc[:, 'province':].groupby('province').mean()

weather_status.tail(3)