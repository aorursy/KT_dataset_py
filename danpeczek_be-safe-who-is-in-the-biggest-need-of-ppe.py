import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns



sns.set_style('dark')
ecdc_infections = pd.read_csv('/kaggle/input/uncover/ECDC/current-data-on-the-geographic-distribution-of-covid-19-cases-worldwide.csv')

ecdc_infections["daterep"] = pd.to_datetime(ecdc_infections[["year","month","day"]])

ecdc_latest_date = ecdc_infections.sort_values(by="daterep", ascending=False)["daterep"].values[0]

ecdc_infections.head()
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

ecdc_infections.head()
infections = pd.read_csv('/kaggle/input/uncover/johns_hopkins_csse/johns-hopkins-covid-19-daily-dashboard-cases-by-country.csv')

infections.head()
country_information = pd.read_csv('/kaggle/input/covidindicators/inform-covid-indicators.csv')

country_information["population_living_in_urban_areas"] = country_information["population_living_in_urban_areas"].replace('No data', None)

country_information["population_living_in_urban_areas"] = country_information["population_living_in_urban_areas"].astype('float64')

country_information.head()
country_testing = pd.read_csv('/kaggle/input/uncover/our_world_in_data/covid-19-testing-all-observations.csv')

country_testing['entity'] = country_testing.entity.str.split('-').str[0]

country_testing['entity'] = country_testing.entity.str.rstrip()

country_testing.head()
goverment_measures = pd.read_csv('/kaggle/input/uncover/HDE/acaps-covid-19-government-measures-dataset.csv')

goverment_measures.head()
d = ecdc_infections.copy()

d = d.drop(["day", "month", "year", "geoid", "countryterritorycode", "popdata2018"], axis=1)

d = d.groupby(["countriesandterritories","daterep"]).sum().unstack(level=0)

most_infected_countries = d[d.index==ecdc_latest_date].xs("active_cases", axis=1).sort_values(by=ecdc_latest_date, axis=1, ascending=False).iloc[:, 0:10].columns
plt.clf()

plt.figure(figsize=(25,15))

ax = sns.lineplot(data=d.xs("active_cases", axis=1)[most_infected_countries], dashes=False)

i=0

for label in ax.get_xticklabels():

    if i%10!=0:

        label.set_visible(False)

    i+=1

plt.show()
def getPatientZeroDate(country, df=d):

    return df.xs("cases", axis=1)[df.xs("cases", axis=1)[country]>=1][country].index[0]
plt.clf()

f, axes = plt.subplots(len(most_infected_countries), 1, figsize=(20, 100))

# sns.despine(left=True)

index = 0

for country in most_infected_countries:

    plt.figure(figsize=(20,10))

    ax=sns.lineplot(data=d[d.index>=getPatientZeroDate(country)].xs("active_cases", axis=1)[country], ax=axes[index])

    i = 0

    for label in ax.get_xticklabels():

        if i%3!=0:

            label.set_visible(False)

        i+=1

    index+=1

    ax.set_title("Infections over time in the "+country)

plt.tight_layout()
def getDensity(countryName, country_info = country_information):

    density = country_info[country_info['country'] == countryName]['population_density'].values

    if len(density) == 0:

        return None

    return density[0]



def getPercentageOfUrbanArea(countryName, country_info = country_information):

    urban_areas = country_info[country_info['country'] == countryName]['population_living_in_urban_areas'].values

    if len(urban_areas) == 0:

        return None

    return urban_areas[0]



def getPopulation(countryName, country_info = ecdc_infections):

    population = country_info[country_info['countriesandterritories'] == countryName]['popdata2018'].values

    if len(population) == 0:

        return None

    return population[0]
country_data = infections[['country_region', 'last_update', 'confirmed', 'deaths','recovered','active']].copy()

country_data['density'] = country_data['country_region'].apply(lambda x: getDensity(x))

country_data['pop2018'] = country_data['country_region'].apply(lambda x: getPopulation(x))

country_data['perc_of_population_in_urban_area'] = country_data['country_region'].apply(lambda x: getPercentageOfUrbanArea(x))

country_data['perc_of_population_in_urban_area'] = np.around(country_data['perc_of_population_in_urban_area'].values, decimals=2)

country_data.head()
country_data['percentage'] = country_data['active']/country_data['pop2018']*100

plt.clf()

plt.figure(figsize=(15,10))

ax = sns.barplot(x="country_region", y="percentage", data=country_data.sort_values(by='percentage', ascending=False).head(10))

ax.set_title("Top 10 countries with the highest percentage of active cases in comparison to population")

plt.show()
goverment_measures[(goverment_measures["country"] == "Qatar") | (goverment_measures["country"] == "Andorra")][["country", "measure", "date_implemented"]]
goverment_measures[(goverment_measures["country"] == "Spain") | (goverment_measures["country"] == "Italy")].drop_duplicates(subset=["country", "measure"])[["country", "measure", "date_implemented"]]
plt.clf()

plt.figure(figsize=(15,10))

ax = sns.barplot(x="density", y="percentage", data=country_data.sort_values(by="percentage", ascending=False).head(15))

ax.set_title("Top 10 highest percentage rate in countries compared to the density")

plt.show()
plt.clf()

plt.figure(figsize=(15,5))

ax = sns.barplot(x="perc_of_population_in_urban_area", y="percentage", data=country_data.sort_values(by="percentage", ascending=False).head(15))

ax.set_title("Top 10 countries with the highest percentage of active cases with the percentage of people living in urban areas.")

plt.show()
def getLatestCountryDate(entity, df=country_testing):

    return df[df['entity'] == entity].sort_values(by='cumulative_total', ascending=False)['date'].values[0]



def getInfections(entity, df=country_data):

    infections = country_data[country_data['country_region'] == entity]['active'].values

    if entity == 'United States':

        infections = country_data[country_data['country_region'] == 'US']['active'].values

    if len(infections) == 0:

        return 0

    return infections[0]



entities = country_testing['entity'].drop_duplicates().values

tests_total = pd.DataFrame(columns = ['entity', 'total_tests', 'tests_per_thousand'])



for entity in entities:

    tests_total = tests_total.append({'entity' : entity ,

                                      'total_tests' : country_testing[(country_testing['entity']==entity) & (country_testing['date']==getLatestCountryDate(entity))]['cumulative_total'].values[0],

                                      'tests_per_thousand' : country_testing[(country_testing['entity']==entity) & (country_testing['date']==getLatestCountryDate(entity))]['cumulative_total_per_thousand'].values[0]

                                     } , ignore_index=True)

tests_total['population'] = tests_total['entity'].apply(lambda x: getPopulation(x))

tests_total['active_infections'] = tests_total['entity'].apply(lambda x: getInfections(x)) 

tests_total.head()
plt.clf()

plt.figure(figsize=(15,10))

ax = sns.barplot(x='total_tests', y='entity', data=tests_total.sort_values('active_infections', ascending=False).head(25))

ax.set_title('Number of tests performed by top 25 countries with the biggest number of active cases')

ax.set_xlabel('Total number of tests')

ax.set_ylabel('Country Name')

plt.show()
plt.clf()

plt.figure(figsize=(15,10))

ax = sns.barplot(x='tests_per_thousand', y='entity', data=tests_total.sort_values('active_infections', ascending=False).head(25))

ax.set_title('Number of tests per 1000 citizens by top 25 countries with the biggest number of active cases')

ax.set_xlabel('Total number of tests')

ax.set_ylabel('Country Name')

plt.show()
hospitals = pd.read_csv('/kaggle/input/uncover/esri_covid-19/esri_covid-19/definitive-healthcare-usa-hospital-beds.csv')

hospitals['num_licens'] = hospitals['num_licens'].replace("****", 0)



hospitals['num_licens'] = hospitals['num_licens'].astype('int32')

hospitals['num_icu_be'] = hospitals['num_icu_be'].fillna(0)

hospitals['num_icu_be'] = hospitals['num_icu_be'].astype('int32')



covid_us = pd.read_csv('/kaggle/input/uncover/covid_tracking_project/covid-statistics-by-us-states-totals.csv')

covid_us['hospitalized'] = covid_us['hospitalized'].fillna(0)

covid_us['hospitalized'] = covid_us['hospitalized'].astype('int32')



hospitals.head()
def getNumberOfStaff(state_code, hospitals = hospitals):

    h = hospitals.groupby('hq_state')['num_licens'].sum().reset_index()

    return h[h['hq_state'] == state_code]['num_licens'].values[0]



def getNumberOfIcuBeds(state_code, hospitals = hospitals):

    h = hospitals.groupby('hq_state')['num_icu_be'].sum().reset_index()

    return h[h['hq_state'] == state_code]['num_icu_be'].values[0]
df_state = covid_us[['state', 'datemodified', 'hospitalized']].copy()

df_state['active'] = covid_us['positive'].fillna(0) - covid_us['death'].fillna(0)

df_state['active'] = df_state['active'].astype('int32')

df_state['licensed_staff'] = df_state['state'].apply(lambda state: getNumberOfStaff(state))

df_state['icu_beds'] = df_state['state'].apply(lambda state: getNumberOfIcuBeds(state))

df_state['active_per_icu_beds'] = df_state['active']/df_state['icu_beds']

df_state['hospitalized_per_icu_beds'] = (df_state['hospitalized']/df_state['icu_beds'])*100

df_state['active_per_staff'] = df_state['active']/df_state['licensed_staff']

df_state['hospitalized_per_staff'] = df_state['hospitalized']/df_state['licensed_staff']

# df_state.head()
plt.clf()

plt.figure(figsize=(25,10))

ax = sns.barplot(x='state', y='hospitalized_per_icu_beds', data=df_state.sort_values(by='hospitalized_per_icu_beds', ascending=False).head(10))

ax.set_title("Ratio of the hospitalized patients to the available ICU beds.")

ax.set_ylabel("Percentage of hospitalized patients to the number of all ICU beds in the state")

plt.show()
plt.clf()

plt.figure(figsize=(25,10))

ax = sns.barplot(x='state', y='hospitalized_per_staff', data=df_state.sort_values(by='hospitalized_per_staff', ascending=False).head(10))

ax.set_title("Ratio of the hospitalized patients to the licensed hospital staff.")

ax.set_ylabel("Ratio")

plt.show()
plt.clf()

plt.figure(figsize=(25,10))

ax = sns.barplot(x='state', y='active_per_staff', data=df_state.sort_values(by='active_per_staff', ascending=False).head(10))

ax.set_title("Ratio of the active patients to the licensed hospital staff.")

ax.set_ylabel("Ratio")

plt.show()