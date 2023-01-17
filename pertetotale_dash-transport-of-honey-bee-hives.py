# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import pandas as pd
import plotly.express as px  # (version 4.7.0)
import plotly.graph_objects as go
import numpy as np
!pip install dash
import dash  # (version 1.14.0) pip install dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
merged = pd.read_csv(r"../input/transportation-honey-bee-hives-top-20/transport_bee_hives_15.csv",sep=",",
                     engine='python', header=0, index_col=0,encoding="UTF-8")
merged
dfind = merged.set_index("State") # df 
# melt
new_df = dfind.melt(id_vars=[ 'abbrev','state'], value_vars=['January1_colonies','Maximum_colonies',"April1_colonies",
                             'July1_colonies'],var_name=["Date"], value_name='Hives') # 'abbrev',

new_df["log_Hives"] = new_df.Hives.apply( lambda x : np.log(x))
new_df.info()
new_df.tail(22)
#fig = go.Figure(go.Choropleth(geojson=counties, locations=new_df.abbrev, z=new_df.log_Hives,
#                                    colorscale="Viridis", zmin=10, zmax=14,
#                                    marker_opacity=0.75, marker_line_width=0))
fig = px.choropleth(new_df, locationmode='USA-states', locations=new_df.abbrev, color=new_df.log_Hives, # geojson=counties,
                           color_continuous_scale=px.colors.sequential.YlOrRd,
                           range_color=(10, 13.34),
                           scope="usa",
                           labels={'log_Hives':'log. number of hives present'},
                            hover_data=['state', "Hives"],# , 
                          )

#fig.update_layout(mapbox_style="carto-positron",
#                  mapbox_zoom=3, mapbox_center = {"lat": 37.0902, "lon": -95.7129})
fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
fig.show()
app = dash.Dash(__name__)

# ------------------------------------------------------------------------------
# Import and clean data (importing csv )
df = pd.read_csv("../input/transportation-honey-bee-hives-top-20/transport_bee_hives_15.csv",sep=",", engine='python', header=0)#/t
#df = df.groupby(['State', 'Year'])[['Pct of Colonies Impacted']].mean() #df.reset_index(inplace=True)
print(df[:5])

# ------------------------------------------------------------------------------
# App layout
app.layout = html.Div([

    html.H2("Hive relocations in USA - Dash", style={'text-align': 'center'}),

    dcc.Dropdown(id="slct_time", # quarter
                 options=[
                     {"label": "January1_colonies", "value": "January1_colonies"}, #
                     {"label": "Maximum_colonies", "value": "Maximum_colonies"},#
                     {"label": "April1_colonies", "value": "April1_colonies"},#
                     {"label": "July1_colonies", "value": "July1_colonies"}],#
                 multi=False,
                 value="January1_colonies",
                 style={'width': "40%"}
                 ),

    html.Div(id='output_container', children=[]), #, children=[] container
    html.Br(),

    dcc.Graph(id='my_bee_map', figure={})#figure={} fig

])

# ------------------------------------------------------------------------------
# Connect the Plotly graphs with Dash Components
@app.callback(
    [Output(component_id='output_container', component_property='children'),
     Output(component_id='my_bee_map', component_property='figure')],
    [Input(component_id='slct_time', component_property='value')]
)
def update_graph(option_slctd):
    print(option_slctd)
    #print(type(option_slctd))

    container = "The time chosen by user was: {}".format(option_slctd)

    dff = new_df.copy() #  CHANGED
    dff = dff[dff["Date"]== option_slctd] #["Year"]
    
    # Plotly Express
    fig = px.choropleth(
        data_frame=dff,
        locationmode='USA-states',
        locations='abbrev', #state_code
        scope="usa",
        color="log_Hives" , #'Pct of Colonies Impacted'
        hover_data=['state', "Hives"],# LIST !! , 
        color_continuous_scale=px.colors.sequential.YlOrRd,
        labels={'Hives': '% Bee Colonies present (log.)'},
        template='plotly_dark'
    )

    # Plotly Graph Objects (GO)
    # fig = go.Figure(
    #     data=[go.Choropleth(
    #         locationmode='USA-states',
    #         locations=dff['state_code'],
    #         z=dff["Colonies Impacted"].astype(float),
    #         colorscale='Reds',
    #     )]
    # )
    #
    # fig.update_layout(
    #     title_text="Bees in the USA",
    #     title_xanchor="center",
    #     title_font=dict(size=24),
    #     title_x=0.5,
    #     geo=dict(scope='usa'),
    # )

    return container, fig

# ------------------------------------------------------------------------------
if __name__ == '__main__':
    app.run_server(debug=True, use_reloader=False)# Turn off reloader if inside Jupyter  