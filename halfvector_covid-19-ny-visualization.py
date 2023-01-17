import plotly.express as px

import plotly.graph_objects as go

import time



from datetime import datetime, timedelta



import pandas as pd

pd.set_option('display.max_columns', None)

pd.set_option('display.max_rows', None)
df = pd.read_csv('../input/covid19-ny-data/ny_confirmed_cases.csv')



# keep only one value per day

df['Timestamp'] = pd.to_datetime(df['Timestamp']).dt.tz_localize('UTC').dt.tz_convert('US/Eastern')

df['Date'] = df['Timestamp'].dt.strftime("%Y%m%d")



df = df.sort_values(by=['Location','Timestamp'])

df = df.drop_duplicates(subset=['Location', 'Count'], keep='first')



style_overrides = {

    'New York City': 'solid',

    'Total Number of Positive Cases': 'solid',

}



fig = px.line(df, x='Timestamp', y='Count', color='Location',

              height=600,

              line_dash_map=style_overrides,

              line_dash_sequence=['dot'],

              line_dash='Location',

              line_shape='spline',

              render_mode='svg'

             )

fig.update_traces(mode='lines+markers')

fig.show()