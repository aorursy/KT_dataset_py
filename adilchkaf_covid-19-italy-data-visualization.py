# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import plotly.express as px

import plotly.offline as py

import plotly.graph_objs as go





# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
csv_region = pd.read_csv("../input/covid19-in-italy/covid19_italy_region.csv")

csv_region.head()
csv_region.tail()
fig = px.sunburst(csv_region.sort_values(by='NewPositiveCases', ascending=False).reset_index(drop=True), path=["Country", "RegionName"], values="NewPositiveCases", title='Confirmed Cases', color_discrete_sequence = px.colors.qualitative.Prism)

fig.data[0].textinfo = 'label+text+value'

fig.show()


py.offline.init_notebook_mode(connected=True)

cases_df = csv_region[['Date','TotalPositiveCases']].groupby('Date').sum()



trace_tot = go.Scatter(

    x=cases_df.index, 

    y=cases_df.TotalPositiveCases,

    mode="markers+lines",

    name = 'Total Positive Cases'

)







data = [trace_tot]

py.iplot({

    "data": data,

    "layout": go.Layout(title="Total positve cases ")

})
py.offline.init_notebook_mode(connected=True)

state_df = csv_region[['Date','Deaths','Recovered']].groupby('Date').sum()



trace_dea = go.Scatter(

    x=state_df.index, 

    y=state_df.Deaths,

    mode="markers+lines",

    name = 'Deaths'

)



trace_rec = go.Scatter(

    x=state_df.index, 

    y=state_df.Recovered,

    mode="markers+lines",

    name = 'Recovered'

)



data = [trace_dea, trace_rec]

py.iplot({

    "data": data,

    "layout": go.Layout(title="COVID 19 deaths with recovred")

})
py.offline.init_notebook_mode(connected=True)

hosp = csv_region[['Date','HospitalizedPatients','IntensiveCarePatients']].groupby('Date').sum()



trace_hosp = go.Scatter(

    x=hosp.index, 

    y=hosp.HospitalizedPatients,

    mode="markers+lines",

    name = 'Hospitalized Patients'

)



trace_int = go.Scatter(

    x=hosp.index, 

    y=hosp.IntensiveCarePatients,

    mode="markers+lines",

    name = 'Intensive Care Patients'

)



data = [trace_hosp, trace_int]

py.iplot({

    "data": data,

    "layout": go.Layout(title="Hospitalized and Intensive Care Patients")

})
py.init_notebook_mode(connected=True)

 



spread = csv_region.sort_values(by=['Date'])

fig = px.scatter_mapbox(spread,

                    animation_frame='Date',

                    animation_group="Country",

                    lat="Latitude", lon="Longitude", hover_name="RegionName", 

                    hover_data=["Country","RegionName","TotalPositiveCases","Recovered","Deaths"],

                    size="TotalPositiveCases",

                    color_discrete_sequence=['red'],

                    zoom=4

                    )

fig.update_layout(mapbox_style="open-street-map")

fig.update_layout(title="Spread over the time"

                 , width = 900, height = 600)

fig.show()