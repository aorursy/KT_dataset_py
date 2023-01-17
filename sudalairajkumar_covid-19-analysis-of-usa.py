import os

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

color = sns.color_palette()

%matplotlib inline



import plotly.offline as py

from plotly import tools

py.init_notebook_mode(connected=True)

import plotly.graph_objs as go

import plotly.express as px



pd.options.mode.chained_assignment = None



# Read the data

us_data_path = "/kaggle/input/covid19-in-usa/"

us_df = pd.read_csv(us_data_path + "us_covid19_daily.csv")

us_states_df = pd.read_csv(us_data_path + "us_states_covid19_daily.csv")

us_df["date"] = pd.to_datetime(us_df["date"], format="%Y%m%d")

us_states_df = us_states_df.reindex(index=us_states_df.index[::-1])

us_states_df["date"] = pd.to_datetime(us_states_df["date"], format="%Y%m%d").dt.date.astype(str)

#us_states_df.head()



# US state code to name mapping

state_map_dict = {'AL': 'Alabama',

 'AK': 'Alaska',

 'AS': 'American Samoa',

 'AZ': 'Arizona',

 'AR': 'Arkansas',

 'CA': 'California',

 'CO': 'Colorado',

 'CT': 'Connecticut',

 'DE': 'Delaware',

 'DC': 'District of Columbia',

 'D.C.': 'District of Columbia',

 'FM': 'Federated States of Micronesia',

 'FL': 'Florida',

 'GA': 'Georgia',

 'GU': 'Guam',

 'HI': 'Hawaii',

 'ID': 'Idaho',

 'IL': 'Illinois',

 'IN': 'Indiana',

 'IA': 'Iowa',

 'KS': 'Kansas',

 'KY': 'Kentucky',

 'LA': 'Louisiana',

 'ME': 'Maine',

 'MH': 'Marshall Islands',

 'MD': 'Maryland',

 'MA': 'Massachusetts',

 'MI': 'Michigan',

 'MN': 'Minnesota',

 'MS': 'Mississippi',

 'MO': 'Missouri',

 'MT': 'Montana',

 'NE': 'Nebraska',

 'NV': 'Nevada',

 'NH': 'New Hampshire',

 'NJ': 'New Jersey',

 'NM': 'New Mexico',

 'NY': 'New York',

 'NC': 'North Carolina',

 'ND': 'North Dakota',

 'MP': 'Northern Mariana Islands',

 'OH': 'Ohio',

 'OK': 'Oklahoma',

 'OR': 'Oregon',

 'PW': 'Palau',

 'PA': 'Pennsylvania',

 'PR': 'Puerto Rico',

 'RI': 'Rhode Island',

 'SC': 'South Carolina',

 'SD': 'South Dakota',

 'TN': 'Tennessee',

 'TX': 'Texas',

 'UT': 'Utah',

 'VT': 'Vermont',

 'VI': 'Virgin Islands',

 'VA': 'Virginia',

 'WA': 'Washington',

 'WV': 'West Virginia',

 'WI': 'Wisconsin',

 'WY': 'Wyoming'}



state_code_dict = {v:k for k, v in state_map_dict.items()}

state_code_dict["Chicago"] = 'Illinois'



def correct_state_names(x):

    try:

        return state_map_dict[x.split(",")[-1].strip()]

    except:

        return x.strip()

    

def get_state_codes(x):

    try:

        return state_code_dict[x]

    except:

        return "Others"



covid_19_df = pd.read_csv("../input/novel-corona-virus-2019-dataset/covid_19_data.csv")

us_covid_df = covid_19_df[covid_19_df["Country/Region"]=="US"]

us_covid_df["Province/State"] = us_covid_df["Province/State"].apply(correct_state_names)

us_covid_df["StateCode"] = us_covid_df["Province/State"].apply(lambda x: get_state_codes(x))



cumulative_df = us_covid_df.groupby("ObservationDate")["Confirmed", "Deaths", "Recovered"].sum().reset_index()



### Plot for number of cumulative covid cases over time

fig = px.bar(cumulative_df, x="ObservationDate", y="Confirmed")

layout = go.Layout(

    title=go.layout.Title(

        text="Daily cumulative count of confirmed COVID-19 cases in US",

        x=0.5

    ),

    font=dict(size=14),

    width=800,

    height=500,

    xaxis_title = "Date of observation",

    yaxis_title = "Number of confirmed cases"

)



fig.update_layout(layout)

fig.show()



### Plot for number of cumulative covid cases over time

fig = px.bar(cumulative_df, x="ObservationDate", y="Deaths")

layout = go.Layout(

    title=go.layout.Title(

        text="Daily cumulative count of deaths due to COVID-19 in US",

        x=0.5

    ),

    font=dict(size=14),

    width=800,

    height=500,

    xaxis_title = "Date of observation",

    yaxis_title = "Number of death cases"

)



fig.update_layout(layout)

fig.show()



### Plot for number of cumulative covid cases over time

cumulative_df["ConfirmedNew"] = cumulative_df["Confirmed"].diff() 

fig = px.bar(cumulative_df, x="ObservationDate", y="ConfirmedNew")

layout = go.Layout(

    title=go.layout.Title(

        text="Daily count of new confirmed COVID-19 cases in US",

        x=0.5

    ),

    font=dict(size=14),

    width=800,

    height=500,

    xaxis_title = "Date of observation",

    yaxis_title = "Number of confirmed cases"

)



fig.update_layout(layout)

fig.show()
import datetime



cumulative_df = us_covid_df.groupby(["StateCode", "ObservationDate"])["Confirmed", "Deaths", "Recovered"].sum().reset_index()

cumulative_df["ObservationDate"] = pd.to_datetime(cumulative_df["ObservationDate"] , format="%m/%d/%Y").dt.date

cumulative_df = cumulative_df.sort_values(by="ObservationDate").reset_index(drop=True)

start_date = datetime.date(2020, 2, 25)

cumulative_df = cumulative_df[cumulative_df["ObservationDate"]>=start_date]

cumulative_df["ObservationDate"] = cumulative_df["ObservationDate"].astype(str)



fig = px.choropleth(locations=cumulative_df["StateCode"],

                    color=cumulative_df["Confirmed"], 

                    locationmode="USA-states",

                    scope="usa",

                    animation_frame=cumulative_df["ObservationDate"],

                    color_continuous_scale='Reds',

                    range_color=[0,600000]

                    #autocolorscale=False,

                   )



layout = go.Layout(

    title=go.layout.Title(

        text="Cumulative count of COVID-19 cases in US states",

        x=0.5

    ),

    font=dict(size=14),

)



fig.update_layout(layout)

fig.show()
cumulative_df = us_covid_df.groupby(["StateCode", "ObservationDate"])["Confirmed", "Deaths", "Recovered"].sum().reset_index()

cumulative_df["ObservationDate"] = pd.to_datetime(cumulative_df["ObservationDate"] , format="%m/%d/%Y").dt.date

cumulative_df = cumulative_df.sort_values(by="ObservationDate").reset_index(drop=True)

start_date = datetime.date(2020, 3, 10)

cumulative_df = cumulative_df[cumulative_df["ObservationDate"]>=start_date]

cumulative_df["ObservationDate"] = cumulative_df["ObservationDate"].astype(str)



fig = px.scatter(cumulative_df, x="Confirmed", y="Deaths", animation_frame="ObservationDate", animation_group="StateCode",

           size="Confirmed", color="StateCode", hover_name="StateCode",

           log_x=False, size_max=55, range_x=[0,400000], range_y=[0,40000])



layout = go.Layout(

    title=go.layout.Title(

        text="Changes in number of confirmed & death cases over time in US states",

        x=0.5

    ),

    font=dict(size=14),

    xaxis_title = "Total number of confirmed cases",

    yaxis_title = "Total number of death cases"

)



fig.update_layout(layout)



fig.show()
fig = px.bar(us_df, x="date", y="total")



layout = go.Layout(

    title=go.layout.Title(

        text="Cumulative number of Total COVID-19 testing over time in US",

        x=0.5

    ),

    font=dict(size=14),

    width=800,

    height=500,

    xaxis_title = "Date of observation",

    yaxis_title = "Number of covid-19 testings"

)



fig.update_layout(layout)

fig.show()
# import plotly.express as px



# us_states_df["positive"] = us_states_df["positive"].fillna(0)



# fig = px.scatter(us_states_df, x="total", y="positive", animation_frame="date", animation_group="state",

#            size="positive", color="state", hover_name="state",

#            log_x=False, size_max=55, range_x=[0,3000000], range_y=[0,600000])



# layout = go.Layout(

#     title=go.layout.Title(

#         text="Total testing counts Vs Positive Counts over time",

#         x=0.5

#     ),

#     font=dict(size=14),

# #     width=800,

# #     height=500,

#     xaxis_title = "Total number of tests",

#     yaxis_title = "Number of positive tests"

# )



# fig.update_layout(layout)

# fig.show()