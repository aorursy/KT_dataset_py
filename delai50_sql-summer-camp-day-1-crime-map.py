# Import packages

from google.cloud import bigquery

import plotly.graph_objs as go

from plotly.offline import iplot, init_notebook_mode

init_notebook_mode(connected=True)

from sklearn.preprocessing import LabelEncoder
# Retrieve data for Chicago crimes

client = bigquery.Client()

dataset_ref = client.dataset("chicago_crime", project="bigquery-public-data")

table_ref = dataset_ref.table('crime')

table = client.get_table(table_ref)
# Create a dataframe (with only 500 rows for speed and simplicity)

crimes_df = client.list_rows(table, max_results=500).to_dataframe()

crimes_df['primary_type_encoded'] = LabelEncoder().fit_transform(crimes_df['primary_type'])
# Plot a quick and simple map

data = [ go.Scattergeo(

        locationmode = 'USA-states',

        lon = crimes_df['longitude'],

        lat = crimes_df['latitude'],

        text = crimes_df['primary_type'],

        marker = dict(

                color = crimes_df['primary_type_encoded']

                )

        ) ]



layout = dict(

        title = 'A few Chicago crimes', 

        geo = dict(

            scope='usa',

            showland = True,

            landcolor = "rgb(250, 250, 250)",

            countrywidth = 0.5,

            subunitwidth = 0.5 

            )

        )



fig = go.Figure(data=data, layout=layout)

iplot(fig)