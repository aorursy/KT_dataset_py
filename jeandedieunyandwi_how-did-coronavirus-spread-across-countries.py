#I will start importing all required libraries. 



import numpy as np #for numerical computation

import pandas as pd #for working with dataframes

import plotly #For interactive visualizations

import plotly.express as px #for Interactive visualizations

import plotly.graph_objects as go #world mapping

from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

init_notebook_mode(connected=True)

import os



for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
#Reading the data

corona_df=pd.read_csv('/kaggle/input/novel-corona-virus-2019-dataset/covid_19_data.csv')
corona_df.head(5)
corona_df.tail(5)
corona_df=corona_df.rename(columns={"ObservationDate": "Date", "Country/Region":"Country", "Confirmed":"Confirmed Cases", "Recovered":"Recovered Cases"})
corona_df.head()
choro_map=px.choropleth(corona_df, 

                    locations="Country", 

                    locationmode = "country names",

                    color="Confirmed Cases", 

                    hover_name="Country", 

                    animation_frame="Date"

                   )



choro_map.update_layout(

    title_text = 'How did Coronavirus Spread across countries?',

    title_x = 0.5,

    geo=dict(

        showframe = False,

        showcoastlines = False,

    ))

    

choro_map.show()
choro_map_recovered=px.choropleth(corona_df, 

                       locations='Country/Region',

                        locationmode="country names",

                        color="Recovered",

                        hover_name="Country/Region",

                        animation_frame='ObservationDate'

                               

                       )



choro_map_recovered.update_layout(

    title_text='Is the World recovering from the Coronavirus pandemic?',

    title_x=0.7,

    

    geo=dict(

        showframe=False,

        showcoastlines=False,

    

    )



)
choro_map_recovered=px.choropleth(corona_df, 

                       locations='Country',

                        locationmode="country names",

                        color="Recovered Cases",

                        hover_name="Country",

                        animation_frame='Date'

                               

                       )



choro_map_recovered.update_layout(

    title_text='Is the World recovering from the Coronavirus pandemic?',

    title_x=0.7,

    

    geo=dict(

        showframe=False,

        showcoastlines=False,

    

    )



)