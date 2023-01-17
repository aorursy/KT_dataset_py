#imports

import pandas as pd 

import plotly.graph_objects as go

import os

from IPython.core.display import display, HTML

# Input data files are available in the "../input/" directory.





for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
#create dataframes

confirmed=pd.read_csv('/kaggle/input/novel-corona-virus-2019-dataset/time_series_covid_19_confirmed.csv')

recovered=pd.read_csv('/kaggle/input/novel-corona-virus-2019-dataset/time_series_covid_19_recovered.csv')

covid_19_data=pd.read_csv('/kaggle/input/novel-corona-virus-2019-dataset/covid_19_data.csv')

time_series_deaths=pd.read_csv('/kaggle/input/novel-corona-virus-2019-dataset/time_series_covid_19_deaths.csv')

line_list=pd.read_csv('/kaggle/input/novel-corona-virus-2019-dataset/COVID19_line_list_data.csv')

open_line_list=pd.read_csv('/kaggle/input/novel-corona-virus-2019-dataset/COVID19_open_line_list.csv')

display(HTML("<h1 style='text-alignment:middle'>Carona Virus Pandemic 2020</h1>"))
display(HTML("<h1>Data Last Updated: " + str(max(covid_19_data['ObservationDate'])) + "</h1>"))
#Graph the cases per Country/Region

US_Data=covid_19_data[covid_19_data['Country/Region']=='US']

Confirmed_Cases_State=pd.DataFrame(US_Data.groupby('Province/State')['Confirmed'].sum())

Confirmed_Cases_State.reset_index()

Confirmed_Cases_State=Confirmed_Cases_State.sort_values(by="Confirmed", ascending=False)



#Bar Chart

fig=go.Figure( 

    data=go.Bar(name='Rona',x=Confirmed_Cases_State.index.to_list(), y=Confirmed_Cases_State['Confirmed'])        

)

fig.update_layout(

title="Confirmed Caronavirus Cases by US State"

)

fig.show()
#Data in Table for above chart

Confirmed_Cases_State
display(HTML("<h2>Now that we've taken a look at the breakdown of cases by state, we have a grasp of overall magnitude or prevalence of the disease in the U.S. but does it matter that so many people have it? What is the general implication of the disease for a given person who has it?</h2>"))
covid_19_data[covid_19_data['Country/Region']=='US'].sort_values(by='Deaths', ascending=False)
line_list[line_list['country']=='USA']['id'].to_excel('Carona_1.xlsx')
line_list.columns