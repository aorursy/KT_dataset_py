import numpy as np 

import pandas as pd 

import matplotlib.pyplot as plt

import plotly.graph_objects as go

import plotly.express as px

pd.set_option('display.max_rows', None)

import datetime

from plotly.subplots import make_subplots

data = pd.read_csv('../input/novel-corona-virus-2019-dataset/covid_19_data.csv')
data[["Confirmed","Deaths","Recovered"]] =data[["Confirmed","Deaths","Recovered"]].astype(int)
data['Country/Region'] = data['Country/Region'].replace('Mainland China', 'China')
data['Active_case'] = data['Confirmed'] - data['Deaths'] - data['Recovered']

def dark_plot_confirmed(data,lockdown,month_lockdown,country):

    fig = go.Figure()

    fig.add_trace(go.Scatter(x=data["ObservationDate"], y=data['Confirmed'],

                    mode="lines+text",

                    name='Confirmed cases',

                    marker_color='orange',

                        ))



    fig.add_annotation(

            x=lockdown,

            y=data['Confirmed'].max(),

            text="COVID-19 pandemic lockdown in "+ country,

             font=dict(

            family="Courier New, monospace",

            size=16,

            color="red"

            ),

    )





    fig.add_shape(

        # Line Vertical

        dict(

            type="line",

            x0=lockdown,

            y0=data['Confirmed'].max(),

            x1=lockdown,

    

            line=dict(

                color="red",

                width=3

            )

    ))

    fig.add_annotation(

            x=month_lockdown,

            y=data['Confirmed'].min(),

            text="Month after lockdown",

             font=dict(

            family="Courier New, monospace",

            size=16,

            color="#00FE58"

            ),

    )



    fig.add_shape(

        # Line Vertical

        dict(

            type="line",

            x0=month_lockdown,

            y0=data['Confirmed'].max(),

            x1=month_lockdown,

    

            line=dict(

                color="#00FE58",

                width=3

            )

    ))

    fig

    fig.update_layout(

        title='Evolution of Confirmed cases over time in '+ country,

        template='plotly_dark'



    )



    fig.show()
def dark_plot_active(data,lockdown,month_lockdown,country):

    fig = go.Figure()

    fig.add_trace(go.Scatter(x=data["ObservationDate"], y=data['Active_case'],

                    mode="lines+text",

                    name='Active cases',

                    marker_color='#00FE58',

                        ))



    fig.add_annotation(

            x=lockdown,

            y=data['Active_case'].max(),

            text="COVID-19 pandemic lockdown in "+ country,

             font=dict(

            family="Courier New, monospace",

            size=16,

            color="red"

            ),

    )





    fig.add_shape(

        # Line Vertical

        dict(

            type="line",

            x0=lockdown,

            y0=data['Active_case'].max(),

            x1=lockdown,

    

            line=dict(

                color="red",

                width=3

            )

    ))

    fig.add_annotation(

            x=month_lockdown,

            y=data['Active_case'].min(),

            text="Month after lockdown",

             font=dict(

            family="Courier New, monospace",

            size=16,

            color="rgb(255,217,47)"

            ),

    )



    fig.add_shape(

        # Line Vertical

        dict(

            type="line",

            x0=month_lockdown,

            y0=data['Active_case'].max(),

            x1=month_lockdown,

    

            line=dict(

                color="rgb(255,217,47)",

                width=3

            )

    ))

    fig

    fig.update_layout(

        title='Evolution of Active cases over time in '+ country,

        template='plotly_dark'



    )



    fig.show()
def dark_plot_recovered(data,lockdown,month_lockdown,country):

    fig = go.Figure()

    fig.add_trace(go.Scatter(x=data["ObservationDate"], y=data['Recovered'],

                    mode="lines+text",

                    name='Recovered cases',

                    marker_color='rgb(192,229,232)',

                        ))



    fig.add_annotation(

            x=lockdown,

            y=data['Recovered'].max(),

            text="COVID-19 pandemic lockdown in "+ country,

             font=dict(

            family="Courier New, monospace",

            size=16,

            color="red"

            ),

    )





    fig.add_shape(

        # Line Vertical

        dict(

            type="line",

            x0=lockdown,

            y0=data['Recovered'].max(),

            x1=lockdown,

    

            line=dict(

                color="red",

                width=3

            )

    ))

    fig.add_annotation(

            x=month_lockdown,

            y=data['Active_case'].min(),

            text="Month after lockdown",

             font=dict(

            family="Courier New, monospace",

            size=16,

            color="rgb(103,219,165)"

            ),

    )



    fig.add_shape(

        # Line Vertical

        dict(

            type="line",

            x0=month_lockdown,

            y0=data['Recovered'].max(),

            x1=month_lockdown,

    

            line=dict(

                color="rgb(103,219,165)",

                width=3

            )

    ))

    fig

    fig.update_layout(

        title='Evolution of Recovered cases over time in '+ country,

        template='plotly_dark'



    )



    fig.show()
def white_plot_confirmed(data,lockdown,month_lockdown,country):

    fig = go.Figure()

    fig.add_trace(go.Scatter(x=data["ObservationDate"], y=data['Confirmed'],

                    mode="lines+text",

                    name='Confirmed cases',

                    marker_color='black',

                        ))



    fig.add_annotation(

            x=lockdown,

            y=data['Confirmed'].max(),

            text="COVID-19 pandemic lockdown in "+ country,

             font=dict(

            family="Courier New, monospace",

            size=16,

            color="red"

            ),

    )





    fig.add_shape(

        # Line Vertical

        dict(

            type="line",

            x0=lockdown,

            y0=data['Confirmed'].max(),

            x1=lockdown,

    

            line=dict(

                color="red",

                width=3

            )

    ))

    fig.add_annotation(

            x=month_lockdown,

            y=data['Confirmed'].min(),

            text="Month after lockdown",

             font=dict(

            family="Courier New, monospace",

            size=16,

            color="rgb(209,175,232)"

            ),

    )



    fig.add_shape(

        # Line Vertical

        dict(

            type="line",

            x0=month_lockdown,

            y0=data['Confirmed'].max(),

            x1=month_lockdown,

    

            line=dict(

                color="rgb(209,175,232)",

                width=3

            )

    ))

    fig

    fig.update_layout(

        title='Evolution of Confirmed cases over time in '+ country,

        template='plotly_white'



    )



    fig.show()

    

def white_plot_active(data,lockdown,month_lockdown,country):

    fig = go.Figure()

    fig.add_trace(go.Scatter(x=data["ObservationDate"], y=data['Active_case'],

                    mode="lines+text",

                    name='Active cases',

                    marker_color='rgb(79,144,166)',

                        ))



    fig.add_annotation(

            x=lockdown,

            y=data['Active_case'].max(),

            text="COVID-19 pandemic lockdown in "+ country,

             font=dict(

            family="Courier New, monospace",

            size=16,

            color="red"

            ),

    )





    fig.add_shape(

        # Line Vertical

        dict(

            type="line",

            x0=lockdown,

            y0=data['Active_case'].max(),

            x1=lockdown,

    

            line=dict(

                color="red",

                width=3

            )

    ))

    fig.add_annotation(

            x=month_lockdown,

            y=data['Active_case'].min(),

            text="Month after lockdown",

             font=dict(

            family="Courier New, monospace",

            size=16,

            color="rgb(92,79,111)"

            ),

    )



    fig.add_shape(

        # Line Vertical

        dict(

            type="line",

            x0=month_lockdown,

            y0=data['Active_case'].max(),

            x1=month_lockdown,

    

            line=dict(

                color="rgb(92,79,111)",

                width=3

            )

    ))

    fig

    fig.update_layout(

        title='Evolution of Active cases over time in '+ country,

        template='plotly_white'



    )



    fig.show()

    

def white_plot_recovered(data,lockdown,month_lockdown,country):

    fig = go.Figure()

    fig.add_trace(go.Scatter(x=data["ObservationDate"], y=data['Recovered'],

                    mode="lines+text",

                    name='Recovered cases',

                    marker_color='rgb(206,102,232)',

                        ))



    fig.add_annotation(

            x=lockdown,

            y=data['Recovered'].max(),

            text="COVID-19 pandemic lockdown in "+ country,

             font=dict(

            family="Courier New, monospace",

            size=16,

            color="red"

            ),

    )





    fig.add_shape(

        # Line Vertical

        dict(

            type="line",

            x0=lockdown,

            y0=data['Recovered'].max(),

            x1=lockdown,

    

            line=dict(

                color="red",

                width=3

            )

    ))

    fig.add_annotation(

            x=month_lockdown,

            y=data['Recovered'].min(),

            text="Month after lockdown",

             font=dict(

            family="Courier New, monospace",

            size=16,

            color="rgb(103,219,165)"

            ),

    )



    fig.add_shape(

        # Line Vertical

        dict(

            type="line",

            x0=month_lockdown,

            y0=data['Recovered'].max(),

            x1=month_lockdown,

    

            line=dict(

                color="rgb(103,219,165)",

                width=3

            )

    ))

    fig

    fig.update_layout(

        title='Evolution of Recovered cases over time in '+ country,

        template='plotly_white'



    )



    fig.show()

    

Data_tunisia = data [(data['Country/Region'] == 'Tunisia') ].reset_index(drop=True)

dark_plot_confirmed(Data_tunisia,"03/22/2020","04/22/2020","Tunisia")
dark_plot_active(Data_tunisia,"03/22/2020","04/22/2020","Tunisia")
dark_plot_recovered(Data_tunisia,"03/22/2020","04/22/2020","Tunisia")
Data_Italy = data [(data['Country/Region'] == 'Italy') ].reset_index(drop=True)

Data_italy_op= Data_Italy.groupby(["ObservationDate","Country/Region"])[["Confirmed","Deaths","Recovered","Active_case"]].sum().reset_index().reset_index(drop=True)
white_plot_confirmed(Data_italy_op,"03/09/2020","04/09/2020","Italy")
white_plot_active(Data_italy_op,"03/09/2020","04/09/2020","Italy")
white_plot_recovered(Data_italy_op,"03/09/2020","04/09/2020","Italy")
Data_France = data [(data['Country/Region'] == 'France') ].reset_index(drop=True)

Data_France_op= Data_France.groupby(["ObservationDate","Country/Region"])[["Confirmed","Deaths","Recovered","Active_case"]].sum().reset_index().reset_index(drop=True)
dark_plot_confirmed(Data_France_op,"03/17/2020","04/17/2020","France")
dark_plot_active(Data_France_op,"03/17/2020","04/17/2020","France")
dark_plot_recovered(Data_France_op,"03/17/2020","04/17/2020","France")
Data_UK = data [(data['Country/Region'] == 'UK') ].reset_index(drop=True)

Data_UK_op= Data_UK.groupby(["ObservationDate","Country/Region"])[["Confirmed","Deaths","Recovered","Active_case"]].sum().reset_index().reset_index(drop=True)
white_plot_confirmed(Data_UK_op,"03/23/2020","04/23/2020","United Kingdom")
white_plot_active(Data_UK_op,"03/23/2020","04/23/2020","United Kingdom")
white_plot_recovered(Data_UK_op,"03/23/2020","04/23/2020","United Kingdom")
Data_India = data [(data['Country/Region'] == 'India') ].reset_index(drop=True)

Data_India_op= Data_India.groupby(["ObservationDate","Country/Region"])[["Confirmed","Deaths","Recovered","Active_case"]].sum().reset_index().reset_index(drop=True)
dark_plot_confirmed(Data_India_op,"03/24/2020","04/24/2020","India")
dark_plot_active(Data_India_op,"03/24/2020","04/24/2020","India")
dark_plot_recovered(Data_India_op,"03/24/2020","04/24/2020","India")
Data_Germany = data [(data['Country/Region'] == 'Germany') ].reset_index(drop=True)

Data_Germany_op= Data_Germany.groupby(["ObservationDate","Country/Region"])[["Confirmed","Deaths","Recovered","Active_case"]].sum().reset_index().reset_index(drop=True)

white_plot_confirmed(Data_Germany_op,"03/23/2020","04/23/2020","Germany")
white_plot_active(Data_Germany_op,"03/23/2020","04/23/2020","Germany")
white_plot_recovered(Data_Germany_op,"03/23/2020","04/23/2020","Germany")
Data_Australia = data [(data['Country/Region'] == 'Australia') ].reset_index(drop=True)

Data_Australia_op= Data_Australia.groupby(["ObservationDate","Country/Region"])[["Confirmed","Deaths","Recovered","Active_case"]].sum().reset_index().reset_index(drop=True)

dark_plot_confirmed(Data_Australia_op,"03/23/2020","04/23/2020","Australia")
dark_plot_active(Data_Australia_op,"03/23/2020","04/23/2020","Australia")
dark_plot_recovered(Data_Australia_op,"03/23/2020","04/23/2020","Australia")
Data_Calif = data [(data['Province/State'] == 'California') ].reset_index(drop=True)

Data_Calif_op= Data_Calif.groupby(["ObservationDate"])[["Confirmed","Deaths","Recovered","Active_case"]].sum().reset_index().reset_index(drop=True)

white_plot_confirmed(Data_Calif_op,"03/19/2020","04/19/2020","California")
white_plot_active(Data_Calif_op,"03/19/2020","04/19/2020","California")
Data_NewYork = data [(data['Province/State'] == 'New York') ].reset_index(drop=True)

Data_NewYork_op= Data_NewYork.groupby(["ObservationDate"])[["Confirmed","Deaths","Recovered","Active_case"]].sum().reset_index().reset_index(drop=True)

dark_plot_confirmed(Data_NewYork_op,"03/22/2020","04/22/2020","New York")
dark_plot_active(Data_NewYork_op,"03/22/2020","04/22/2020","New York")
Data_Michigan = data [(data['Province/State'] == 'Michigan') ].reset_index(drop=True)

Data_Michigan_op= Data_Michigan.groupby(["ObservationDate"])[["Confirmed","Deaths","Recovered","Active_case"]].sum().reset_index().reset_index(drop=True)

white_plot_confirmed(Data_Michigan_op,"03/24/2020","04/24/2020","Michigan")
white_plot_active(Data_Michigan_op,"03/24/2020","04/24/2020","Michigan")
Data_Oregon = data [(data['Province/State'] == 'Oregon') ].reset_index(drop=True)

Data_Oregon_op= Data_Oregon.groupby(["ObservationDate"])[["Confirmed","Deaths","Recovered","Active_case"]].sum().reset_index().reset_index(drop=True)

dark_plot_confirmed(Data_Oregon_op,"03/24/2020","04/24/2020","Oregon")
dark_plot_active(Data_Oregon_op,"03/24/2020","04/24/2020","Oregon")
Data_Belgium = data [(data['Country/Region'] == 'Belgium') ].reset_index(drop=True)

Data_Belgium_op= Data_Belgium.groupby(["ObservationDate","Country/Region"])[["Confirmed","Deaths","Recovered","Active_case"]].sum().reset_index().reset_index(drop=True)

white_plot_confirmed(Data_Belgium_op,"03/18/2020","04/18/2020","Belgium")
white_plot_active(Data_Belgium_op,"03/18/2020","04/18/2020","Belgium")
white_plot_recovered(Data_Belgium_op,"03/18/2020","04/18/2020","Belgium")
Data_Czech = data [(data['Country/Region'] == 'Czech Republic') ].reset_index(drop=True)

Data_Czech_op= Data_Czech.groupby(["ObservationDate","Country/Region"])[["Confirmed","Deaths","Recovered","Active_case"]].sum().reset_index().reset_index(drop=True)

dark_plot_confirmed(Data_Czech_op,"03/16/2020","04/16/2020","Czech Republic")
dark_plot_active(Data_Czech_op,"03/16/2020","04/16/2020","Czech Republic")
dark_plot_recovered(Data_Czech_op,"03/16/2020","04/16/2020","Czech Republic")
Data_Portugal = data [(data['Country/Region'] == 'Portugal') ].reset_index(drop=True)

Data_Portugal_op= Data_Portugal.groupby(["ObservationDate","Country/Region"])[["Confirmed","Deaths","Recovered","Active_case"]].sum().reset_index().reset_index(drop=True)

white_plot_confirmed(Data_Portugal_op,"03/19/2020","04/19/2020","Portugal")
white_plot_active(Data_Portugal_op,"03/19/2020","04/19/2020","Portugal")
white_plot_recovered(Data_Portugal_op,"03/19/2020","04/19/2020","Portugal")
Data_Austria = data [(data['Country/Region'] == 'Austria') ].reset_index(drop=True)

Data_Austria_op= Data_Austria.groupby(["ObservationDate","Country/Region"])[["Confirmed","Deaths","Recovered","Active_case"]].sum().reset_index().reset_index(drop=True)

dark_plot_confirmed(Data_Austria_op,"03/16/2020","04/16/2020","Austria")
dark_plot_active(Data_Austria_op,"03/16/2020","04/16/2020","Austria")
dark_plot_recovered(Data_Austria_op,"03/16/2020","04/16/2020","Austria")
Data_Turkey = data [(data['Country/Region'] == 'Turkey') ].reset_index(drop=True)

Data_Turkey_op= Data_Turkey.groupby(["ObservationDate","Country/Region"])[["Confirmed","Deaths","Recovered","Active_case"]].sum().reset_index().reset_index(drop=True)

white_plot_confirmed(Data_Turkey_op,"04/23/2020","05/23/2020","Turkey")
white_plot_active(Data_Turkey_op,"04/23/2020","05/23/2020","Turkey")
white_plot_recovered(Data_Turkey_op,"04/23/2020","05/23/2020","Turkey")
Data_Serbia = data [(data['Country/Region'] == 'Serbia') ].reset_index(drop=True)

Data_Serbia_op= Data_Serbia.groupby(["ObservationDate","Country/Region"])[["Confirmed","Deaths","Recovered","Active_case"]].sum().reset_index().reset_index(drop=True)

dark_plot_confirmed(Data_Serbia_op,"03/15/2020","04/15/2020","Serbia")
dark_plot_active(Data_Serbia_op,"03/15/2020","04/15/2020","Serbia")
dark_plot_recovered(Data_Serbia_op,"03/15/2020","04/15/2020","Serbia")
Data_Poland = data [(data['Country/Region'] == 'Poland') ].reset_index(drop=True)
white_plot_confirmed(Data_Poland,"03/13/2020","04/13/2020","Poland")
white_plot_active(Data_Poland,"03/13/2020","04/13/2020","Poland")
white_plot_recovered(Data_Poland,"03/13/2020","04/13/2020","Poland")
Data_Bolivia = data [(data['Country/Region'] == 'Bolivia') ].reset_index(drop=True)
dark_plot_confirmed(Data_Bolivia,"03/22/2020","04/22/2020","Bolivia")
dark_plot_active(Data_Bolivia,"03/22/2020","04/22/2020","Bolivia")
dark_plot_recovered(Data_Bolivia,"03/22/2020","04/22/2020","Bolivia")
Data_NZ = data [(data['Country/Region'] == 'New Zealand') ].reset_index(drop=True)
white_plot_confirmed(Data_NZ,"03/26/2020","04/26/2020","New Zealand")
white_plot_active(Data_NZ,"03/26/2020","04/26/2020","New Zealand")
white_plot_recovered(Data_NZ,"03/26/2020","04/26/2020","New Zealand")
Data_Morocco = data [(data['Country/Region'] == 'Morocco') ].reset_index(drop=True)
dark_plot_confirmed(Data_Morocco,"03/19/2020","04/19/2020","Morocco")
dark_plot_active(Data_Morocco,"03/19/2020","04/19/2020","Morocco")
dark_plot_recovered(Data_Morocco,"03/19/2020","04/19/2020","Morocco")
Data_Honduras = data [(data['Country/Region'] == 'Honduras') ].reset_index(drop=True)

white_plot_confirmed(Data_Honduras,"03/20/2020","04/20/2020","Honduras")
white_plot_active(Data_Honduras,"03/20/2020","04/20/2020","Honduras")
white_plot_recovered(Data_Honduras,"03/20/2020","04/20/2020","Honduras")
Data_UAE = data [(data['Country/Region'] == 'United Arab Emirates') ].reset_index(drop=True)
dark_plot_confirmed(Data_UAE,"03/26/2020","04/26/2020","United Arab Emirates")
dark_plot_active(Data_UAE,"03/26/2020","04/26/2020","United Arab Emirates")
dark_plot_recovered(Data_UAE,"03/26/2020","04/26/2020","United Arab Emirates")
Data_Iran = data [(data['Country/Region'] == 'Iran') ].reset_index(drop=True)
white_plot_confirmed(Data_Iran,"03/14/2020","04/14/2020","Iran")
white_plot_active(Data_Iran,"03/14/2020","04/14/2020","Iran")
white_plot_recovered(Data_Iran,"03/14/2020","04/14/2020","Iran")
Data_Bangladesh = data [(data['Country/Region'] == 'Bangladesh') ].reset_index(drop=True)
dark_plot_confirmed(Data_Bangladesh,"03/26/2020","04/26/2020","Bangladesh")
dark_plot_active(Data_Bangladesh,"03/26/2020","04/26/2020","Bangladesh")
dark_plot_recovered(Data_Bangladesh,"03/26/2020","04/26/2020","Bangladesh")