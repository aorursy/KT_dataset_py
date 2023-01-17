# first Kaggle code

# inspired by the below blog:

#https://towardsdatascience.com/visualizing-the-coronavirus-pandemic-with-choropleth-maps-7f30fccaecf5
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import plotly.graph_objects as gr #for static choropleth

import plotly.express as ex #for dynamic choropleth

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input/novel-corona-virus-2019-dataset/'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.



# read data set and print it 

df= pd.read_csv('/kaggle/input/novel-corona-virus-2019-dataset/covid_19_data.csv')



#rename columns for better understanding

df = df.rename(columns={'Country/Region':'Country', 'Province/State':'State','ObservationDate':'Date'})



#sort for convenience

#df = df.sort_values(["Date", "Country"], ascending = (True, True))



#display df

display(df)
#get data at date-country level

df_countries = df.groupby(['Country', 'Date']).sum().reset_index()



#we remove granularity of state and last update columns

#resetindex numbers each row and shows in expanded format



#remove rows where confirmed is 0

df_countries = df_countries[df_countries['Confirmed']>0]



#sort for better understanding

df_countries = df_countries.sort_values(["Date", "Country"], ascending = (True, True))



print(df_countries)
#some more cleaning for static plot



#make sure there's data only for the latest data

#df_countries['Date'] = pd.to_datetime(df_countries['Date'])

max_date = max(df_countries['Date'])

print(max_date)



#keep data only for max date

sdf_countries = df_countries[df_countries['Date']==max_date]



display(sdf_countries)
#static choropleth plot :) 



fig = gr.Figure(

        data = gr.Choropleth(

                locations = sdf_countries['Country'],

                locationmode = 'country names',

                z = sdf_countries['Confirmed'],

                colorscale = 'Reds',

                marker_line_color = 'black',

                marker_line_width = 0.5

            )

      )



display(fig)



fig.update_layout(

    title_text = 'Confirmed Cases as of '+ str(max_date),

    title_x = 0.5

  #  geo=dict(

  #      showframe = False,

  #      showcoastlines = False,

  #      projection_type = 'equirectangular'

  #  )

)



display(fig)
#dynamic choropleth



comap = ex.choropleth(

            df_countries, 

            locations="Country", 

            locationmode = "country names",

            color="Confirmed", 

            hover_name="Country", 

            animation_frame="Date",

            color_continuous_scale = ex.colors.sequential.OrRd

        )



comap.update_layout(

    title_text = 'Global Spread of Coronavirus',

    title_x = 0.5,

)

    



display(comap)