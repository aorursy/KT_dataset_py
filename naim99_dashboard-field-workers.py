#you need to install Dash if you don't install yet 

#dash 

!pip install dash

import dash_core_components as dcc

import dash_html_components as html

import plotly.graph_objs as go

import pandas as pd

import dash_table
df = pd.read_csv(r'../input/controle.csv', sep =';')

#pv = pd.pivot_table(df, index=['Name'], columns=["Status"], values=['Quantity'], aggfunc=sum, fill_value=0)
df
df['difference uplaod-received'] = df['received'] - df['realised in the field']
df
import dash

import dash_core_components as dcc

import dash_html_components as html

import plotly.graph_objs as go

import pandas as pd

import emoji

import dash_table

import pandas as pd

import dash

import dash_core_components as dcc

import dash_html_components as html



external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']



app = dash.Dash(__name__, external_stylesheets=external_stylesheets)





df = pd.read_csv('../input/controle.csv', sep =';')

#pv = pd.pivot_table(df, index=['Name'], columns=["Status"], values=['Quantity'], aggfunc=sum, fill_value=0)



df['difference uplaod-received'] = df['received'] - df['realised in the field']

df['check_upload'] = (df['difference uplaod-received'] == 0).astype(int)

img2 = emoji.emojize(':+1:', use_aliases=True)

img1 = emoji.emojize(':question:', use_aliases=True)

df['check_upload'] = df['check_upload'].replace([0,1], [img1, img2])

df



import flask

from flask import Flask, request, render_template

import json

app.layout = html.Div([

    dash_table.DataTable(

        id='datatable-interactivity',

        columns=[

            {"name": i, "id": i, "deletable": True, "selectable": True} for i in df.columns

        ],

        data=df.to_dict('records'),

        editable=True,

        filter_action="native",

        sort_action="native",

        sort_mode="multi",

        column_selectable="single",

        row_selectable="multi",

        row_deletable=True,

        selected_columns=[],

        selected_rows=[],

        page_action="native",

        page_current= 0,

        page_size= 10,

    ),

    html.Div(id='datatable-interactivity-container')

])



@app.callback(

    Output('datatable-interactivity', 'style_data_conditional'),

    [Input('datatable-interactivity', 'selected_columns')]

)

def update_styles(selected_columns):

    return [{

        'if': { 'column_id': i },

        'background_color': '#D2F3FF'

    } for i in selected_columns]



@app.callback(

    Output('datatable-interactivity-container', "children"),

    [Input('datatable-interactivity', "derived_virtual_data"),

     Input('datatable-interactivity', "derived_virtual_selected_rows")])

def update_graphs(rows, derived_virtual_selected_rows):

    # When the table is first rendered, `derived_virtual_data` and

    # `derived_virtual_selected_rows` will be `None`. This is due to an

    # idiosyncracy in Dash (unsupplied properties are always None and Dash

    # calls the dependent callbacks when the component is first rendered).

    # So, if `rows` is `None`, then the component was just rendered

    # and its value will be the same as the component's dataframe.

    # Instead of setting `None` in here, you could also set

    # `derived_virtual_data=df.to_rows('dict')` when you initialize

    # the component.

    if derived_virtual_selected_rows is None:

        derived_virtual_selected_rows = []



    dff = df if rows is None else pd.DataFrame(rows)



    colors = ['#7FDBFF' if i in derived_virtual_selected_rows else '#0074D9'

              for i in range(len(dff))]



    return [

        dcc.Graph(

            id=column,

            figure={

                "data": [

                    {

                        "x": dff["country"],

                        "y": dff[column],

                        "type": "bar",

                        "marker": {"color": colors},

                    }

                ],

                "layout": {

                    "xaxis": {"automargin": True},

                    "yaxis": {

                        "automargin": True,

                        "title": {"text": column}

                    },

                    "height": 250,

                    "margin": {"t": 10, "l": 10, "r": 10},

                },

            },

        )

        # check if column exists - user may have deleted it

        # If `column.deletable=False`, then you don't

        # need to do this check.

        for column in ["pop", "lifeExp", "gdpPercap"] if column in dff

    ]