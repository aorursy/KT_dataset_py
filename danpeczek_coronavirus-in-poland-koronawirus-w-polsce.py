import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns



ecdc_infections = pd.read_csv('../input/uncover/UNCOVER_v4/UNCOVER/ECDC/current-data-on-the-geographic-distribution-of-covid-19-cases-worldwide.csv')

ecdc_infections["daterep"] = pd.to_datetime(ecdc_infections[["year","month","day"]])

ecdc_latest_date = ecdc_infections.sort_values(by="daterep", ascending=False)["daterep"].values[0]

ecdc_infections["cum_cases"] = 0

ecdc_infections["cum_deaths"] = 0

codes = ecdc_infections["geoid"].drop_duplicates().values



for code in codes:

    infections = ecdc_infections[ecdc_infections["geoid"]==code].sort_values("daterep")["cases"].cumsum()

    deaths = ecdc_infections[ecdc_infections["geoid"]==code].sort_values("daterep")["deaths"].cumsum()

    indexes = infections.index

    inf_index=0

    for inf in infections:

        ecdc_infections.loc[indexes[inf_index], "cum_cases"] = inf

        inf_index+=1

    death_index=0

    death_indexes = deaths.index

    for death in deaths:

        ecdc_infections.loc[death_indexes[death_index], "cum_deaths"] = death

        death_index+=1 

ecdc_infections["active_cases"] = ecdc_infections["cum_cases"] - ecdc_infections["cum_deaths"]



countries_list = ecdc_infections[['countriesandterritories', 'continentexp']].drop_duplicates(subset='countriesandterritories')



ecdc_infections.head()
def getPatientZeroDateForCountry(countryName, data=ecdc_infections):

    return data[(data['countriesandterritories'] == countryName) & (data['cases'] > 0)].sort_values('daterep', ascending=True).head(1)['daterep'].values[0]
goverment_measures = pd.read_csv('../input/uncover/UNCOVER_v4/UNCOVER/HDE/HDE/acaps-covid-19-government-measures-dataset.csv')

goverment_measures.head()
testing_data = pd.read_csv('../input/uncover/UNCOVER_v4/UNCOVER/our_world_in_data/covid-19-testing-all-observations.csv')

poland_tests = testing_data[testing_data.entity == 'Poland - samples tested'].copy().reset_index()



poland_tests['daily_change_in_cumulative_total'].fillna(0, inplace=True)

poland_tests['daily_change_in_cumulative_total_per_thousand'].fillna(0, inplace=True)

poland_tests.head()
poland_patient_zero = getPatientZeroDateForCountry("Poland")

poland_patient_zero
patient_zero_countries = []

for index, row in countries_list.iterrows():

    date = getPatientZeroDateForCountry(row['countriesandterritories'])

    if date >= poland_patient_zero-np.timedelta64(2,'D') and date <= poland_patient_zero+np.timedelta64(2,'D') and row['continentexp'] == 'Europe' and row['countriesandterritories'] != 'Poland':

        patient_zero_countries.append(row['countriesandterritories'])

patient_zero_countries
plt.clf()

plt.figure(figsize=(15,10))

ax = sns.lineplot(x = 'daterep', y = 'cum_cases', data=ecdc_infections[ecdc_infections['countriesandterritories'] == 'Poland'])

ax.set_title('Infection curve for a Poland')

ax.set_xlabel('Date')

ax.set_ylabel('Cumulative number of active cases')

plt.show()
plt.clf()

f, axes = plt.subplots(len(patient_zero_countries), 2, figsize=(40, len(patient_zero_countries)*10))

index = 0

for country in patient_zero_countries:

    plt.figure(figsize=(20,10))

    ax = sns.lineplot(x = 'daterep', y = 'cum_cases', data=ecdc_infections[ecdc_infections['countriesandterritories'] == 'Poland'], ax=axes[index, 0])

    ax.set_title('Linear infection curve for a Poland', fontsize=24)

    ax.set_xlabel('Date')

    ax.set_ylabel('Cumulative number of active cases')

    

    plt.figure(figsize=(20,10))

    bx = sns.lineplot(x = 'daterep', y = 'cum_cases', data=ecdc_infections[ecdc_infections['countriesandterritories'] == country], ax=axes[index, 1])

    bx.set_title('Linear infection curve for '+country, fontsize=24)

    bx.set_xlabel('Date')

    bx.set_ylabel('Cumulative number of active cases')

    

    index+=1

plt.tight_layout()

plt.show()
ecdc_infections["log_cum_cases"] = np.log(ecdc_infections["cum_cases"])

plt.clf()

plt.figure(figsize=(15,10))

ax = sns.lineplot(x = 'daterep', y = 'log_cum_cases', data=ecdc_infections[ecdc_infections['countriesandterritories'] == 'Poland'])

ax.set_title('Logarythmic infection curve for a Poland')

ax.set_xlabel('Date')

ax.set_ylabel('Cumulative number of active cases')

plt.show()
plt.clf()

plt.figure(figsize=(15,10))

ax = sns.lineplot(x = 'daterep', y = 'log_cum_cases', data=ecdc_infections[ecdc_infections['countriesandterritories'] == 'Slovenia'])

ax.set_title('Logarythmic infection curve for a Slovenia')

ax.set_xlabel('Date')

ax.set_ylabel('Cumulative number of active cases')

plt.show()
def getNumberOfTests(date, df=poland_tests):

    tests = df[df['date']==date]['daily_change_in_cumulative_total'].values

    if len(tests) != 0:

        return tests[0]

    return 0



ecdc_infections['daterep'] = ecdc_infections['daterep'].dt.strftime('%Y-%m-%d')

tests_and_infections = ecdc_infections[ecdc_infections['countriesandterritories'] == 'Poland'].copy()

tests_and_infections['tests'] = 0

tests_and_infections['tests'] = tests_and_infections['daterep'].apply(getNumberOfTests)



plt.clf()

plt.figure(figsize=(20,10))

ax = sns.lineplot(x = 'daterep', y = 'cases', data=tests_and_infections[tests_and_infections['countriesandterritories'] == 'Poland'], label='Cases')

ax.axhline(240, ls='--')

i=0

for label in ax.get_xticklabels():

    if i%10!=0:

        label.set_visible(False)

    i+=1

ax.set_title('Daily number of infections')
plt.clf()

plt.figure(figsize=(20,10))

ax = sns.lineplot(x = 'daterep', y = 'cases', data=tests_and_infections[tests_and_infections['countriesandterritories'] == 'Poland'], label='Cases')

ax = sns.lineplot(x = 'daterep', y = 'tests', data=tests_and_infections[tests_and_infections['countriesandterritories'] == 'Poland'], label='Tests')

i=0

for label in ax.get_xticklabels():

    if i%10!=0:

        label.set_visible(False)

    i+=1

ax.set_title('Combined number of cases with tests performed in Poland.')
testing_data['daily_change_in_cumulative_total'].fillna(0, inplace=True)

testing_data['daily_change_in_cumulative_total_per_thousand'].fillna(0, inplace=True)



testing_data_window = testing_data[(testing_data['date'] >= '2020-03-04') & (testing_data['date'] <= '2020-05-20')]



plt.clf()

plt.figure(figsize=(20,10))



ax = sns.lineplot(x = 'date', y = 'daily_change_in_cumulative_total_per_thousand', data=testing_data_window[testing_data_window['entity'] == 'Poland - samples tested'], label='Poland')

ax = sns.lineplot(x = 'date', y = 'daily_change_in_cumulative_total_per_thousand', data=testing_data_window[testing_data_window['entity'] == 'Czech Republic - tests performed'], label='Czech Republic')

ax = sns.lineplot(x = 'date', y = 'daily_change_in_cumulative_total_per_thousand', data=testing_data_window[testing_data_window['entity'] == 'Slovenia - tests performed'], label='Slovenia')

ax = sns.lineplot(x = 'date', y = 'daily_change_in_cumulative_total_per_thousand', data=testing_data_window[testing_data_window['entity'] == 'Hungary - tests performed'], label='Hungary')

ax = sns.lineplot(x = 'date', y = 'daily_change_in_cumulative_total_per_thousand', data=testing_data_window[testing_data_window['entity'] == 'Latvia - tests performed'], label='Latvia')



i=0

for label in ax.get_xticklabels():

    if i%10!=0:

        label.set_visible(False)

    i+=1

ax.set_title('Tests per 1000 people in different countries.')

ax.set_ylabel('Tests performed daily')

plt.show()