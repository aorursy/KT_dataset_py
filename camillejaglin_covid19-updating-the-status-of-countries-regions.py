# Imports



import numpy as np

import pandas as pd



import plotly.graph_objects as go



import plotly.express as px

import plotly.io as pio

pio.templates.default = "plotly_dark"

from plotly.subplots import make_subplots

import seaborn as sns
# Import dataset



data = pd.read_csv('/kaggle/input/novel-corona-virus-2019-dataset/covid_19_data.csv')
# Set name of country/region :



country_name = "France"
# Create new datasets name



covid_19_own = []



# Create datas lists



Confirmed = []

Deaths = []

Recovered = []

Active = []



Date = []
# Functions (dataset creation and data recovery)



def create_new_datasets(country):

    

    temp = []

    

    # Loc datas for new dataset

    

    for i in range(len(data)):

        # For country

        if data['Country/Region'][i] == country :

        

        # For region

        #if data['Province/State'][i] == country:

            temp.append(data.loc[i])

    temp = pd.DataFrame(temp).reset_index()

    

    return temp



def get_all_series_values(dataset, confirmed_list_name, deaths_list_name, recovered_list_name, active_list_name, date_name):

    

    confirmed, deaths, recovered = 0, 0, 0

    

    for i in range(len(dataset)):

        

        date = dataset['ObservationDate'][i]

        

        confirmed += dataset['Confirmed'][i]

        deaths += dataset['Deaths'][i]

        recovered += dataset['Recovered'][i]

        

        if (i < len(dataset) - 1 and dataset['ObservationDate'][i + 1] != date) or i == len(dataset) - 1:

            

            current_infected = confirmed - deaths - recovered

            confirmed_list_name.append(confirmed)

            deaths_list_name.append(deaths)

            recovered_list_name.append(recovered)

            active_list_name.append(current_infected)

            date_name.append(date)

            confirmed, deaths, recovered = 0, 0, 0
# Create new dataset



covid_19_own = create_new_datasets(country_name)
# Get series list and date list



get_all_series_values(covid_19_own, Confirmed, Deaths, Recovered, Active, Date)

Date = pd.to_datetime(pd.Series(Date), format='%m/%d/%Y')
# Plots



fig = go.Figure()



fig.add_trace(go.Scatter(x=Date, y=Confirmed, name = 'Confirmed', marker_color='rgba(236, 249, 92, 1)'))

fig.add_trace(go.Scatter(x=Date, y=Deaths, name = 'Deaths', marker_color='rgba(227, 14, 14, 1)'))

fig.add_trace(go.Scatter(x=Date, y=Recovered, name = 'Recovered', marker_color='rgba(64, 246, 44, 1)'))

fig.add_trace(go.Bar(x=Date, y=Active, name = 'Active', marker_color='rgba(243, 255, 210, .2)'))



title_status = "Status of " + country_name



fig.update_layout(title=title_status,

                   xaxis_title='Dates',

                   yaxis_title='Number of cases')



fig.show()