#Data Analyses Libraries

import pandas as pd                

import numpy as np    

from urllib.request import urlopen

import json

import glob

import os



#Importing Data plotting libraries

import matplotlib.pyplot as plt     

import plotly.express as px       

import plotly.offline as py       

import seaborn as sns             

import plotly.graph_objects as go 

from plotly.subplots import make_subplots

import matplotlib.ticker as ticker

import matplotlib.animation as animation

from matplotlib.ticker import MaxNLocator



#Other Miscallaneous Libraries

import warnings

warnings.filterwarnings('ignore')

from IPython.display import HTML

import matplotlib.colors as mc

import colorsys

from random import randint

import re
#Uploading the dataset

us_tracker1 = pd.read_csv('../input/uncover/UNCOVER/covid_tracking_project/covid-statistics-for-all-us-totals.csv')



#Getting the values from the dataset

us_tests_conducted = int(us_tracker1['totaltestresults'])

us_positive_tests = int(us_tracker1['positive'])

us_hospitalizations = int(us_tracker1['hospitalized'])



#Printing the values(

print('Out of total {} Covid-19 tests conducted in US, {} were positive and {} were hospitalized which is {}% of total confirmed cases'

      .format(us_tests_conducted,us_positive_tests,us_hospitalizations,(us_hospitalizations/us_positive_tests)*100))
#Getting the total cases (State wise) in USA

usa_covid_cases = pd.read_csv('../input/covid19-in-usa/us_states_covid19_daily.csv')



#Filter to get the recent data

condition = usa_covid_cases['date'] == 20200621

usa_covid_cases = usa_covid_cases[condition]



#Getting the dataset

usa_covid_cases.tail()
#Adding a column to calculate rate of Admission 

usa_covid_cases['Hospitalization_rate'] = usa_covid_cases['hospitalizedCumulative']/usa_covid_cases['positive']*100



#Sorting the dataframe in descending order

usa_covid_cases.sort_values(by=['Hospitalization_rate'], inplace=True, ascending=False)
#Fetching the data for Spain

spain_data = pd.read_csv('../input/covcsd-covid19-countries-statistical-dataset/covid19-spain-cases.csv')



#Getting the latest data and hospital admission rate in Spain.

latest = spain_data['fecha'] == '2020-04-01'

spain_cases = spain_data[latest]



#Obtaining the values

spain_cases['Rate_general'] = (spain_cases['hospitalizados']/spain_cases['casos'])*100

spain_cases['Rate_icu'] = (spain_cases['uci']/spain_cases['casos'])*100



#Printing the values

spain_cases.head()
#Reading the COVID-19 NET Hospitalization rates dataset

net_data = pd.read_csv('/kaggle/input/covidnet-hospitalization-rates/COVID-NET_Surveillance_03-28-2020.csv')



#Getting to display the dataset

net_data.head()
#Dropping the null values in the dataset

net_data = net_data.dropna()



#Plotting the dataset



ages = net_data.AGE_CATEGORY.unique()



ages = ages[ages != '65+ yr']

ages = ages[ages != '85+']

ages = ages[ages != 'Overall']

df_ages = net_data[net_data.AGE_CATEGORY.isin(ages)]



df_ages = df_ages[df_ages.CATCHMENT == 'Entire Network']



fig, ax2 = plt.subplots(figsize=(20, 5))

ax2.xaxis.set_major_locator(MaxNLocator(integer=True))

sns.lineplot(x=df_ages['MMWR-WEEK'], y=df_ages.WEEKLY_RATE, hue=df_ages.AGE_CATEGORY, ax=ax2)

fig.show()
# Creating the dataset from the URL



ailments_data = {'Medical_condition' : ['Asthma','Autoimmune diseases','Cardiovascualr disease','Chronic lung disease','Liver disease','Hypertension','Immune Suppression','Metabolic disease','Neurological Disease','Obesity','Renal Disease','Other Disease','No medical condition'],

                'Hospitalized_adults' : [738,179,2023,1263,309,3386,582,2509,1404,2735,964,334,504]}

ailments_data = pd.DataFrame(ailments_data)



#Plotting the dataset

fig = px.bar(ailments_data, x='Medical_condition', y='Hospitalized_adults')

fig.show()
#Importing the clinical spectrum data

clinical_spectrum = pd.read_csv('../input/uncover/UNCOVER/einstein/diagnosis-of-covid-19-and-its-clinical-spectrum.csv')



#Filtering the data to contain the values only for the confirmed COVID-19 Tests

confirmed = clinical_spectrum['sars_cov_2_exam_result'] == 'positive'

clinical_spectrum = clinical_spectrum[confirmed]



#Viewing the dataset statistics

clinical_spectrum.tail()
#Filetering the datasets

hospitalized_condtion = clinical_spectrum['patient_addmited_to_regular_ward_1_yes_0_no'] == 't'

us_hospitalized_spectra = clinical_spectrum[hospitalized_condtion]





unhospitalized_condtion = clinical_spectrum['patient_addmited_to_regular_ward_1_yes_0_no'] == 'f'

us_unhospitalized_spectra = clinical_spectrum[unhospitalized_condtion]



#Taking mean value of the spectra conditions

hospitalized_mean = us_hospitalized_spectra.mean(axis = 0, skipna = True) 

unhospitalized_mean = us_unhospitalized_spectra.mean(axis = 0, skipna = True) 



#Making columns for the dataset

hospitalized_mean = hospitalized_mean.to_frame()

hospitalized_mean = hospitalized_mean.reset_index()

hospitalized_mean.columns = ['Parameter','Hospitalized_figures']



unhospitalized_mean = unhospitalized_mean.to_frame()

unhospitalized_mean = unhospitalized_mean.reset_index()

unhospitalized_mean.columns = ['Parameter','Unhospitalized_figures']



#Merging both the dataframes together

hospitalized_mean['Unhospitalized_figures'] = unhospitalized_mean['Unhospitalized_figures']



#Viewing the dataset

hospitalized_mean.dropna()



#The most important clinical factors

hospitalized_mean['Change'] =  hospitalized_mean['Hospitalized_figures'] - hospitalized_mean['Unhospitalized_figures']

hospitalized_mean.sort_values(['Change'], axis=0, ascending=True, inplace=True) 



#Getting to know the health factors that define HCP Requirement for a patient

lower = hospitalized_mean.head(10)

higher = hospitalized_mean.tail(10)
#Printing the values

for i in lower['Parameter']:

    print('For lower value of {}, the patient may require HCP'.format(i))

    

for i in higher['Parameter']:

    print('For higher value of {}, the patient may require HCP'.format(i))
#Reading the dataset

usa_cases_tot = pd.read_csv('../input/covcsd-covid19-countries-statistical-dataset/USA_COVID_19.csv',dtype={"fips": str})



#Getting the GeoJSON File for plotting the data

with urlopen('https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json') as response:

    counties = json.load(response)



#Plotting the data for confirmed Cases of COVID-19    

py.init_notebook_mode(connected=True)



usa_cases_tot['log_ConfirmedCases'] = np.log(usa_cases_tot.Cases + 1)

usa_cases_tot['fips'] = usa_cases_tot['fips'].astype(str).str.rjust(5,'0')

 

fig = px.choropleth(usa_cases_tot, geojson=counties, locations='fips', color='log_ConfirmedCases',

                           color_continuous_scale="Viridis",

                           range_color=(0, 12),

                           scope="usa")



fig.update_layout(title_text="Confirmed Cases of COVID-19 in USA - June 09, 2020")

py.offline.iplot(fig)
#Importing the dataset and Removing NaN Values.

health_indices = pd.read_csv('../input/covcsd-covid19-countries-statistical-dataset/Hospitalization Causes.csv')

health_indices.dropna(subset=['Hospitalization Rates'], axis=0, inplace=True)



#Finding the correaltion Matrix

corr_matrix = health_indices.corr()



#Printing the Correaltion Matrix with just the correlation column

corr_matrix = corr_matrix['Hospitalization Rates']

corr_matrix = corr_matrix.to_frame()



#Getting the correlation matrix values in ascending order

corr_matrix.dropna(subset=['Hospitalization Rates'], axis=0, inplace=True)



#Getting Highest correalted and least correalted values

corr_matrix.sort_values(by=['Hospitalization Rates'], inplace=True, ascending=False)

corr_matrix.drop(corr_matrix.index[0],inplace=True)

most_correlated = corr_matrix.head(15)

least_correlated = corr_matrix.tail(15)



#Printing the heatmap

fig, ax = plt.subplots(figsize=(10,10))  

ax.set_title('Most Correlated Demographic Values with Hospitalization Rates (Positive Correlation) \n\n')

sns.heatmap(most_correlated)

#Printing the heatmap

fig, ax = plt.subplots(figsize=(10,10))  

ax.set_title('Least Correlated Demographic Values with Hospitalization Rates\n\n')

sns.heatmap(least_correlated)
#Creating a dictonary for USA States and Codes



states = {"AL":"Alabama","AK":"Alaska","AZ":"Arizona","AR":"Arkansas","CA":"California","CO":"Colorado","CT":"Connecticut","DE":"Delaware","FL":"Florida","GA":"Georgia","HI":"Hawaii","ID":"Idaho","IL":"Illinois","IN":"Indiana","IA":"Iowa","KS":"Kansas","KY":"Kentucky","LA":"Louisiana","ME":"Maine","MD":"Maryland","MA":"Massachusetts","MI":"Michigan","MN":"Minnesota","MS":"Mississippi","MO":"Missouri","MT":"Montana","NE":"Nebraska","NV":"Nevada","NH":"New Hampshire","NJ":"New Jersey","NM":"New Mexico","NY":"New York","NC":"North Carolina","ND":"North Dakota","OH":"Ohio","OK":"Oklahoma","OR":"Oregon","PA":"Pennsylvania","RI":"Rhode Island","SC":"South Carolina","SD":"South Dakota","TN":"Tennessee","TX":"Texas","UT":"Utah","VT":"Vermont","VA":"Virginia","WA":"Washington","WV":"West Virginia","WI":"Wisconsin","WY":"Wyoming"}

states_data = pd.DataFrame(list(states.items()),columns=['codes','state'])



#Adding the column into COVCSD Dataset to provide codes to states



blend_data = pd.merge(states_data,health_indices,on='state')

blend_data.head()
#Loading the dataset

hospital_data = pd.read_csv('../input/uncover/UNCOVER/harvard_global_health_institute/hospital-capacity-by-state-20-population-contracted.csv')

hospital_data.rename(columns = {'state':'codes'},inplace = True)

hospital_data = hospital_data[['codes','total_hospital_beds','total_icu_beds','hospital_bed_occupancy_rate','icu_bed_occupancy_rate']]



#Merging the datasets together

final_data = pd.merge(blend_data,hospital_data,on='codes')



#Viewing the dataset

final_data.head()
#Finding the correaltion Matrix

corr_matrix = final_data.corr()



#Printing the Correaltion Matrix with just the correlation column

corr_matrix = corr_matrix['total_hospital_beds']

corr_matrix = corr_matrix.to_frame()



#Getting the correlation matrix values in ascending order

corr_matrix.dropna(subset=['total_hospital_beds'], axis=0, inplace=True)



#Getting Highest correalted and least correalted values

corr_matrix.sort_values(by=['total_hospital_beds'], inplace=True, ascending=False)

corr_matrix.drop(corr_matrix.index[0],inplace=True)

most_correlated = corr_matrix.head(30)

least_correlated = corr_matrix.tail(20)



#Printing the heatmap

fig, ax = plt.subplots(figsize=(10,10))  

ax.set_title('Most Correlated Demographic Values with Hospitalization Bed Infrastructure (Positive Correlation) \n\n')

sns.heatmap(most_correlated)
#Printing the heatmap

fig, ax = plt.subplots(figsize=(10,10))  

ax.set_title('Most Correlated Demographic Values with Hospitalization Bed Infrastructure (Negative Correlation)\n\n')

sns.heatmap(least_correlated)