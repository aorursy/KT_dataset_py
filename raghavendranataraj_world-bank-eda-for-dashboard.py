import pandas as pd

import datetime

import plotly.plotly as py

import plotly.graph_objs as go



from plotly.offline import init_notebook_mode, iplot

init_notebook_mode()
data_wb = pd.read_csv("../input/procurement-notices.csv")

data_wb.columns = [col.lower().replace(' ', '_') for col in data_wb.columns]
tod = pd.Timestamp.today() - datetime.timedelta(days=7)

data_wb.deadline_date = pd.to_datetime(data_wb.deadline_date, format='%Y-%m-%dT%H:%M:%S')

final = data_wb[(data_wb.deadline_date > tod) | (data_wb.deadline_date.isnull())]

final.shape[0]
country = data_wb.groupby('country_name').count()['id']

data = [ dict(

        type='choropleth',

        autocolorscale = False,

        locations = country.index,

        z = country,

        locationmode = 'country names'

       ) ]



# chart information

layout = dict(

        title = 'Number of Open bids',

        geo = dict(

        showframe = False,

        showcoastlines = False,

        projection = dict(

            type = 'mercator'

        )

    )

)

   

# actually show our figure

fig = dict( data=data, layout=layout )

iplot( fig, filename='d3-cloropleth-map' )

due = data_wb[data_wb.deadline_date > tod]

deadline_count = due.groupby('deadline_date').count()['id']

data = [go.Scatter(x=deadline_count.index, y=deadline_count)]



layout = dict(title = "Deadlines",

              xaxis= dict(title= 'Date',ticklen= 5,zeroline= False))

fig = dict(data = data, layout = layout)

iplot(fig)
