import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt #visualization library
import plotly as py
import plotly.graph_objs as go # plotly graphical object
#!pip install dash
import dash
import dash_core_components as dcc
import dash_html_components as html

# these two lines are what allow your code to show up in a notebook!
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode()

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


df = pd.read_csv("/kaggle/input/gadcovid19analysis/coronavirus-cases_latest.csv")
df.head()
df.info()
df['Specimen date'] = df['Specimen date'].astype('datetime64')
df.info()

fig = go.Figure([
    go.Bar(x=df['Specimen date'], y=df['Daily lab-confirmed cases'])
])
fig.update_layout(title_text='Daily Confirmed Case')
fig.show()
fig_2 = go.Figure([
    go.Bar(x=df['Specimen date'], y=df['Cumulative lab-confirmed cases'])
])
fig_2.update_layout(title_text='Cumulative Confirmed Case')
fig_2.show()
app = dash.Dash()
#server = app.server

cities = df['Area name'].unique()

app.layout = html.Div([
    dcc.Dropdown(
        id ='city_dropdown',
        options = [{'label': i, 'value': i} for i in cities],
        value = 'Cumulative Confirmed Case'),
    dcc.Graph(id = 'city_graph')
    ])


app = dash.Dash()
app.layout = html.Div([
    dcc.Graph(figure=fig)
])

app.run_server(debug=True, use_reloader=True)
#@app.callback(
#    dash.dependencies.Output('city_graph','fig'),[dash.dependencies.Input('city_dropdown', 'value')])

#app.css.append_css({
#    'external_url': 'https://lunch-time-python.herokuapp.com'
#})

#if __name__ == "__main__":
#    app.run_server(debug=False)