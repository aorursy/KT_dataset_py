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



#Other Miscallaneous Libraries

import warnings

warnings.filterwarnings('ignore')

from IPython.display import HTML

import matplotlib.colors as mc

import colorsys

from random import randint

import re
#Reading the cumulative cases dataset

covid_cases = pd.read_csv('../input/novel-corona-virus-2019-dataset/covid_19_data.csv')



#Viewing the dataset

covid_cases.head()
#Groping the same cities and countries together along with their successive dates.



country_list = covid_cases['Country/Region'].unique()



country_grouped_covid = covid_cases[0:1]



for country in country_list:

    test_data = covid_cases['Country/Region'] == country   

    test_data = covid_cases[test_data]

    country_grouped_covid = pd.concat([country_grouped_covid, test_data], axis=0)

    

country_grouped_covid.reset_index(drop=True)

country_grouped_covid.head()



#Dropping of the column Last Update

country_grouped_covid.drop('Last Update', axis=1, inplace=True)



#Replacing NaN Values in Province/State with a string "Not Reported"

country_grouped_covid['Province/State'].replace(np.nan, "Not Reported", inplace=True)



#Creating a dataset to analyze the cases country wise - As of 05/17/2020



latest_data = country_grouped_covid['ObservationDate'] == '05/17/2020'

country_data = country_grouped_covid[latest_data]



#Plotting a bar graph for confirmed cases vs deaths due to COVID-19 in World.



unique_dates = country_grouped_covid['ObservationDate'].unique()

confirmed_cases = []

recovered = []

deaths = []



for date in unique_dates:

    date_wise = country_grouped_covid['ObservationDate'] == date  

    test_data = country_grouped_covid[date_wise]

    

    confirmed_cases.append(test_data['Confirmed'].sum())

    deaths.append(test_data['Deaths'].sum())

    recovered.append(test_data['Recovered'].sum())

    

#Converting the lists to a pandas dataframe.



country_dataset = {'Date' : unique_dates, 'Confirmed' : confirmed_cases, 'Recovered' : recovered, 'Deaths' : deaths}

country_dataset = pd.DataFrame(country_dataset)



#Plotting the Graph of Cases vs Deaths Globally.



fig = go.Figure()

fig.add_trace(go.Bar(x=country_dataset['Date'], y=country_dataset['Confirmed'], name='Confirmed Cases of COVID-19', marker_color='rgb(55, 83, 109)'))

fig.add_trace(go.Bar(x=country_dataset['Date'],y=country_dataset['Deaths'],name='Total Deaths because of COVID-19',marker_color='rgb(26, 118, 255)'))



fig.update_layout(title='Confirmed Cases and Deaths from COVID-19',xaxis_tickfont_size=14,

                  yaxis=dict(title='Reported Numbers',titlefont_size=16,tickfont_size=14,),

    legend=dict(x=0,y=1.0,bgcolor='rgba(255, 255, 255, 0)',bordercolor='rgba(255, 255, 255, 0)'),barmode='group',bargap=0.15, bargroupgap=0.1)

fig.show()





fig = go.Figure()

fig.add_trace(go.Bar(x=country_dataset['Date'], y=country_dataset['Confirmed'], name='Confirmed Cases of COVID-19', marker_color='rgb(55, 83, 109)'))

fig.add_trace(go.Bar(x=country_dataset['Date'],y=country_dataset['Recovered'],name='Total Recoveries because of COVID-19',marker_color='rgb(26, 118, 255)'))



fig.update_layout(title='Confirmed Cases and Recoveries from COVID-19',xaxis_tickfont_size=14,

                  yaxis=dict(title='Reported Numbers',titlefont_size=16,tickfont_size=14,),

    legend=dict(x=0,y=1.0,bgcolor='rgba(255, 255, 255, 0)',bordercolor='rgba(255, 255, 255, 0)'),

    barmode='group',bargap=0.15, bargroupgap=0.1)

fig.show()
df1 = pd.read_csv("../input/corona-virus-capillary-and-liver-tumor-samples/both_clean_liver_capillary_CoV.csv")

df1.head().style.background_gradient(cmap='PuBuGn')
#Importing the clinical spectrum data

clinical_spectrum = pd.read_csv('../input/uncover/UNCOVER/einstein/diagnosis-of-covid-19-and-its-clinical-spectrum.csv')



#Filtering the data to contain the values only for the confirmed COVID-19 Tests

confirmed = clinical_spectrum['sars_cov_2_exam_result'] == 'positive'

clinical_spectrum = clinical_spectrum[confirmed]



#Filetering the datasets

positive_condition = clinical_spectrum['sars_cov_2_exam_result'] == 'positive'

positive_condition_spectra = clinical_spectrum[positive_condition]





negative_condition = clinical_spectrum['sars_cov_2_exam_result'] == 'negative'

negative_condition_spectra = clinical_spectrum[negative_condition]



#Taking mean value of the spectra conditions

positive_mean = clinical_spectrum.mean(axis = 0, skipna = True) 

negative_mean = clinical_spectrum.mean(axis = 0, skipna = True) 



#Making columns for the dataset

positive_mean = positive_mean.to_frame()

positive_mean = positive_mean.reset_index()

positive_mean.columns = ['Parameter','Positive_figures']



negative_mean = negative_mean.to_frame()

negative_mean = negative_mean.reset_index()

negative_mean.columns = ['Parameter','Negative_figures']



#Merging both the dataframes together

positive_mean['Negative_figures'] = negative_mean['Negative_figures']



#Viewing the dataset

positive_mean.dropna()



#The most important clinical factors

positive_mean['Change'] =  positive_mean['Positive_figures'] - positive_mean['Negative_figures']

positive_mean.sort_values(['Change'], axis=0, ascending=True, inplace=True) 



#Getting to know the health factors that define HCP Requirement for a patient

lower = positive_mean.head(15)

higher = positive_mean.tail(15)



#Printing the values

for i in lower['Parameter']:

    print('For lower value of {}, the patient is Prone to COVID-19'.format(i))

    

for i in higher['Parameter']:

    print('For higher value of {}, the patient is Prone to COVID-19'.format(i))