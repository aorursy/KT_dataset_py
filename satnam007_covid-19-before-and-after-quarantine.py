import numpy as np 

import pandas as pd 

import matplotlib.pyplot as plt

import plotly.graph_objects as go

import plotly.express as px

pd.set_option('display.max_rows', None)

import datetime

from plotly.subplots import make_subplots

data = pd.read_csv('../input/novel-corona-virus-2019-dataset/covid_19_data.csv')

data.head()
from pylab import figure, axes, pie, title, show



# Make a square figure and axes

figure(1, figsize=(6, 6))

ax = axes([0.1, 0.1, 0.8, 0.8])



labels = 'Frogs', 'Hogs', 'Dogs', 'Logs'

fracs = [15, 30, 45, 10]



explode = (0, 0.05, 0, 0)

pie(fracs, explode=explode, labels=labels, autopct='%1.1f%%', shadow=True)

title('Raining Hogs and Dogs', bbox={'facecolor': '0.8', 'pad': 5})

from matplotlib import pyplot as plt



plt.savefig("foo2.png")

plt.savefig("foo2.pdf")



show() 
from matplotlib import pyplot as plt



plt.savefig("foo1.png")

plt.savefig("foo1.pdf")
data[["Confirmed","Deaths","Recovered"]] =data[["Confirmed","Deaths","Recovered"]].astype(int)

data.head()
data['Country/Region'] = data['Country/Region'].replace('Mainland China', 'China')

data.head(2)
data['Active_case'] = data['Confirmed'] - data['Deaths'] - data['Recovered']

data.head(2)
data.columns
selected_country = np.array(['Tunisia','Italy','France','United Kingdom','India','Germany','Australia','US','Belgium','Czench Republic'])

selected_country
# for x in selected_country:

#     print(x)

# #     data_con = data [(data['Country/Region'] ==x) ].reset_index(drop=True)

# #     print(data_con)

# #     print(i=1)

# #     i=i+1
# COVID-19 pandemic lockdown in Tunisia
for country in selected_country:

        data_con = data [(data['Country/Region'] == country) ].reset_index(drop=True)

        fig = go.Figure()

        fig.add_trace(go.Scatter(x=data_con["ObservationDate"], y=data_con['Confirmed'],

                        mode="lines+text",

                        name='Confirmed cases',

                        marker_color='orange',

                            ))

        fig.add_annotation(

                x="03/22/2020",

                y=data_con['Confirmed'].max(),

                text="COVID-19 pandemic lockdown in Tunisia",

                 font=dict(

                family="Courier New, monospace",

                size=16,

                color="red"

                ),) #,

        fig.add_shape(

            # Line Vertical

            dict(

                type="line",

                x0="03/22/2020",

                y0=data_con['Confirmed'].max(),

                x1="03/22/2020",

                line=dict(

                    color="red",

                    width=3

                ) ))

        fig.add_annotation(

                x="04/22/2020",

                y=data_con['Confirmed'].max()-70,

                text="Month after lockdown",

                 font=dict(

                family="Courier New, monospace",

                size=16,

                color="#00FE58"),)

        fig.add_shape(

            # Line Vertical

            dict(

                type="line",

                x0="04/22/2020",

                y0=data_con['Confirmed'].max(),

                x1="04/22/2020",

                line=dict(

                    color="#00FE58",

                    width=3

                ) 

            ))

        fig

        fig.update_layout(

        title='Evolution of Confirmed cases over time in Tunisia',

            template='plotly_dark'   

        )

        plt.savefig("plot5.png")

        fig.show()
Data_tunisia = data [(data['Country/Region'] == 'Tunisia') ].reset_index(drop=True)
Data_tunisia.head()
Data_tunisia = data [(data['Country/Region'] == 'Tunisia') ].reset_index(drop=True)

fig = go.Figure()

fig.add_trace(go.Scatter(x=Data_tunisia["ObservationDate"], y=Data_tunisia['Confirmed'],

                    mode="lines+text",

                    name='Confirmed cases',

                    marker_color='orange',

                        ))



fig.add_annotation(

            x="03/22/2020",

            y=Data_tunisia['Confirmed'].max(),

            text="COVID-19 pandemic lockdown in Tunisia",

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

            x0="03/22/2020",

            y0=Data_tunisia['Confirmed'].max(),

            x1="03/22/2020",

    

            line=dict(

                color="red",

                width=3

            )

))

fig.add_annotation(

            x="04/22/2020",

            y=Data_tunisia['Confirmed'].max()-70,

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

            x0="04/22/2020",

            y0=Data_tunisia['Confirmed'].max(),

            x1="04/22/2020",

    

            line=dict(

                color="#00FE58",

                width=3

            )

))

fig

fig.update_layout(

    title='Evolution of Confirmed cases over time in Tunisia',

        template='plotly_dark'



)



fig.show()
fig = go.Figure()

fig.add_trace(go.Scatter(x=Data_tunisia["ObservationDate"], y=Data_tunisia['Active_case'],

                    mode="lines+text",

                    name='Active cases',

                    marker_color='#00FE58',

                        ))



fig.add_annotation(

            x="03/22/2020",

            y=Data_tunisia['Active_case'].max(),

            text="COVID-19 pandemic lockdown in Tunisia",

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

            x0="03/22/2020",

            y0=Data_tunisia['Active_case'].max(),

            x1="03/22/2020",

    

            line=dict(

                color="red",

                width=3

            )

))





fig.add_annotation(

            x="04/22/2020",

            y=Data_tunisia['Active_case'].max()-60,

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

            x0="04/22/2020",

            y0=Data_tunisia['Active_case'].max(),

            x1="04/22/2020",

    

            line=dict(

                color="rgb(255,217,47)",

                width=3

            )

))

fig.update_layout(

    title='Evolution of Active cases over time in Tunisia',

        template='plotly_dark'



)



fig.show()
fig = go.Figure()

fig.add_trace(go.Scatter(x=Data_tunisia["ObservationDate"], y=Data_tunisia['Recovered'],

                    mode="lines+text",

                    name='Recovered cases',

                    marker_color='rgb(192,229,232)',

                        ))



fig.add_annotation(

            x="03/22/2020",

            y=Data_tunisia['Recovered'].max(),

            text="COVID-19 pandemic lockdown in Tunisia",

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

            x0="03/22/2020",

            y0=Data_tunisia['Recovered'].max(),

            x1="03/22/2020",

    

            line=dict(

                color="red",

                width=3

            )

))





fig.add_annotation(

            x="04/22/2020",

            y=Data_tunisia['Recovered'].max()-60,

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

            x0="04/22/2020",

            y0=Data_tunisia['Recovered'].max(),

            x1="04/22/2020",

    

            line=dict(

                color="rgb(103,219,165)",

                width=3

            )

))

fig.update_layout(

    title='Evolution of Recovered cases over time in Tunisia',

        template='plotly_dark'



)



fig.show()
Data_Italy = data [(data['Country/Region'] == 'Italy') ].reset_index(drop=True)

Data_italy_op= Data_Italy.groupby(["ObservationDate","Country/Region"])[["Confirmed","Deaths","Recovered","Active_case"]].sum().reset_index().reset_index(drop=True)
fig = go.Figure()

fig.add_trace(go.Scatter(x=Data_italy_op["ObservationDate"], y=Data_italy_op['Confirmed'],

                    mode="lines+text",

                    name='Confirmed cases',

                    marker_color='black',

                        ))



fig.add_annotation(

            x="03/09/2020",

            y=Data_italy_op['Confirmed'].max(),

            text="COVID-19 pandemic lockdown in Italy",

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

            x0="03/09/2020",

            y0=Data_italy_op['Confirmed'].max(),

            x1="03/09/2020",

    

            line=dict(

                color="red",

                width=3

            )

))

fig.add_annotation(

            x="04/09/2020",

            y=Data_italy_op['Confirmed'].max()-20000,

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

            x0="04/09/2020",

            y0=Data_italy_op['Confirmed'].max(),

            x1="04/09/2020",

    

            line=dict(

                color="rgb(209,175,232)",

                width=3

            )

))

fig

fig.update_layout(

    title='Evolution of Confirmed cases over time in Italy',

        template='plotly_white'



)



fig.show()
fig = go.Figure()

fig.add_trace(go.Scatter(x=Data_italy_op["ObservationDate"], y=Data_italy_op['Active_case'],

                    mode="lines+text",

                    name='Active cases',

                    marker_color='rgb(79,144,166)',

                        ))



fig.add_annotation(

            x="03/09/2020",

            y=Data_italy_op['Active_case'].max(),

            text="COVID-19 pandemic lockdown in Italy",

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

            x0="03/09/2020",

            y0=Data_italy_op['Active_case'].max(),

            x1="03/09/2020",

    

            line=dict(

                color="red",

                width=3

            )

))

fig.add_annotation(

            x="04/09/2020",

            y=Data_italy_op['Active_case'].max()-9000,

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

            x0="04/09/2020",

            y0=Data_italy_op['Active_case'].max(),

            x1="04/09/2020",

    

            line=dict(

                color="rgb(92,79,111)",

                width=3

            )

))

fig

fig.update_layout(

    title='Evolution of Active cases over time in Italy',

        template='plotly_white'



)



fig.show()
fig = go.Figure()

fig.add_trace(go.Scatter(x=Data_italy_op["ObservationDate"], y=Data_italy_op['Recovered'],

                    mode="lines+text",

                    name='Recovered cases',

                    marker_color='rgb(206,102,232)',

                        ))



fig.add_annotation(

            x="03/09/2020",

            y=Data_italy_op['Recovered'].max(),

            text="COVID-19 pandemic lockdown in Italy",

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

            x0="03/09/2020",

            y0=Data_italy_op['Recovered'].max(),

            x1="03/09/2020",

    

            line=dict(

                color="red",

                width=3

            )

))





fig.add_annotation(

            x="04/09/2020",

            y=Data_italy_op['Recovered'].max()-20000,

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

            x0="04/09/2020",

            y0=Data_italy_op['Recovered'].max(),

            x1="04/09/2020",

    

            line=dict(

                color="rgb(103,219,165)",

                width=3

            )

))

fig.update_layout(

    title='Evolution of Recovered cases over time in Italy',

        template='plotly_white'



)



fig.show()
Data_France = data [(data['Country/Region'] == 'France') ].reset_index(drop=True)

Data_France_op= Data_France.groupby(["ObservationDate","Country/Region"])[["Confirmed","Deaths","Recovered","Active_case"]].sum().reset_index().reset_index(drop=True)
fig = go.Figure()

fig.add_trace(go.Scatter(x=Data_France_op["ObservationDate"], y=Data_France_op['Confirmed'],

                    mode="lines+text",

                    name='Confirmed cases',

                    marker_color='orange',

                        ))



fig.add_annotation(

            x="03/17/2020",

            y=Data_France_op['Confirmed'].max(),

            text="COVID-19 pandemic lockdown in France",

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

            x0="03/17/2020",

            y0=Data_France_op['Confirmed'].max(),

            x1="03/17/2020",

    

            line=dict(

                color="red",

                width=3

            )

))

fig.add_annotation(

            x="04/17/2020",

            y=Data_France_op['Confirmed'].max()-20000,

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

            x0="04/17/2020",

            y0=Data_France_op['Confirmed'].max(),

            x1="04/17/2020",

    

            line=dict(

                color="#00FE58",

                width=3

            )

))

fig

fig.update_layout(

    title='Evolution of Confirmed cases over time in France',

        template='plotly_dark'



)



fig.show()
fig = go.Figure()

fig.add_trace(go.Scatter(x=Data_France_op["ObservationDate"], y=Data_France_op['Active_case'],

                    mode="lines+text",

                    name='Active cases',

                    marker_color='#00FE58',

                        ))



fig.add_annotation(

            x="03/17/2020",

            y=Data_France_op['Active_case'].max(),

            text="COVID-19 pandemic lockdown in France",

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

            x0="03/17/2020",

            y0=Data_France_op['Active_case'].max(),

            x1="03/17/2020",

    

            line=dict(

                color="red",

                width=3

            )

))





fig.add_annotation(

            x="04/17/2020",

            y=Data_France_op['Active_case'].min(),

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

            x0="04/17/2020",

            y0=Data_France_op['Active_case'].max(),

            x1="04/17/2020",

    

            line=dict(

                color="rgb(255,217,47)",

                width=3

            )

))

fig.update_layout(

    title='Evolution of Active cases over time in France',

        template='plotly_dark'



)



fig.show()
fig = go.Figure()

fig.add_trace(go.Scatter(x=Data_France_op["ObservationDate"], y=Data_France_op['Recovered'],

                    mode="lines+text",

                    name='Recovered cases',

                    marker_color='rgb(229,151,232)',

                        ))



fig.add_annotation(

            x="03/17/2020",

            y=Data_France_op['Recovered'].max(),

            text="COVID-19 pandemic lockdown in France",

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

            x0="03/17/2020",

            y0=Data_France_op['Recovered'].max(),

            x1="03/17/2020",

    

            line=dict(

                color="red",

                width=3

            )

))





fig.add_annotation(

            x="04/17/2020",

            y=Data_France_op['Recovered'].max()-9000,

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

            x0="04/17/2020",

            y0=Data_France_op['Recovered'].max(),

            x1="04/17/2020",

    

            line=dict(

                color="rgb(103,219,165)",

                width=3

            )

))

fig.update_layout(

    title='Evolution of Recovered cases over time in France',

        template='plotly_dark'



)



fig.show()
Data_UK = data [(data['Country/Region'] == 'UK') ].reset_index(drop=True)

Data_UK_op= Data_UK.groupby(["ObservationDate","Country/Region"])[["Confirmed","Deaths","Recovered","Active_case"]].sum().reset_index().reset_index(drop=True)
fig = go.Figure()

fig.add_trace(go.Scatter(x=Data_UK_op["ObservationDate"], y=Data_UK_op['Confirmed'],

                    mode="lines+text",

                    name='Confirmed cases',

                    marker_color='black',

                        ))



fig.add_annotation(

            x="03/23/2020",

            y=Data_UK_op['Confirmed'].max(),

            text="COVID-19 pandemic lockdown in UK",

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

            x0="03/23/2020",

            y0=Data_UK_op['Confirmed'].max(),

            x1="03/23/2020",

    

            line=dict(

                color="red",

                width=3

            )

))

fig.add_annotation(

            x="04/23/2020",

            y=Data_UK_op['Confirmed'].max()-20000,

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

            x0="04/23/2020",

            y0=Data_UK_op['Confirmed'].max(),

            x1="04/23/2020",

    

            line=dict(

                color="rgb(209,175,232)",

                width=3

            )

))

fig

fig.update_layout(

    title='Evolution of Confirmed cases over time in UK',

        template='plotly_white'



)



fig.show()
fig = go.Figure()

fig.add_trace(go.Scatter(x=Data_UK_op["ObservationDate"], y=Data_UK_op['Active_case'],

                    mode="lines+text",

                    name='Active cases',

                    marker_color='rgb(79,144,166)',

                        ))



fig.add_annotation(

            x="03/23/2020",

            y=Data_UK_op['Active_case'].max(),

            text="COVID-19 pandemic lockdown in UK",

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

            x0="03/23/2020",

            y0=Data_UK_op['Active_case'].max(),

            x1="03/23/2020",

    

            line=dict(

                color="red",

                width=3

            )

))

fig.add_annotation(

            x="04/23/2020",

            y=Data_UK_op['Active_case'].min(),

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

            x0="04/23/2020",

            y0=Data_UK_op['Active_case'].max(),

            x1="04/23/2020",

    

            line=dict(

                color="rgb(92,79,111)",

                width=3

            )

))

fig

fig.update_layout(

    title='Evolution of Active cases over time in UK',

        template='plotly_white'



)



fig.show()
Data_India = data [(data['Country/Region'] == 'India') ].reset_index(drop=True)

Data_India_op= Data_India.groupby(["ObservationDate","Country/Region"])[["Confirmed","Deaths","Recovered","Active_case"]].sum().reset_index().reset_index(drop=True)
fig = go.Figure()

fig.add_trace(go.Scatter(x=Data_India_op["ObservationDate"], y=Data_India_op['Confirmed'],

                    mode="lines+text",

                    name='Confirmed cases',

                    marker_color='orange',

                        ))



fig.add_annotation(

            x="03/24/2020",

            y=Data_India_op['Confirmed'].max(),

            text="COVID-19 pandemic lockdown in India",

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

            x0="03/24/2020",

            y0=Data_India_op['Confirmed'].max(),

            x1="03/24/2020",

    

            line=dict(

                color="red",

                width=3

            )

))

fig.add_annotation(

            x="04/24/2020",

            y=Data_India_op['Confirmed'].max()-30000,

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

            x0="04/24/2020",

            y0=Data_India_op['Confirmed'].max(),

            x1="04/24/2020",

    

            line=dict(

                color="#00FE58",

                width=3

            )

))

fig

fig.update_layout(

    title='Evolution of Confirmed cases over time in India',

        template='plotly_dark'



)



fig.show()
fig = go.Figure()

fig.add_trace(go.Scatter(x=Data_India_op["ObservationDate"], y=Data_India_op['Active_case'],

                    mode="lines+text",

                    name='Active cases',

                    marker_color='#00FE58',

                        ))



fig.add_annotation(

            x="03/24/2020",

            y=Data_India_op['Active_case'].max(),

            text="COVID-19 pandemic lockdown in India",

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

            x0="03/24/2020",

            y0=Data_India_op['Active_case'].max(),

            x1="03/24/2020",

    

            line=dict(

                color="red",

                width=3

            )

))





fig.add_annotation(

            x="04/24/2020",

            y=Data_India_op['Active_case'].max()-20000,

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

            x0="04/24/2020",

            y0=Data_India_op['Active_case'].max(),

            x1="04/24/2020",

    

            line=dict(

                color="rgb(255,217,47)",

                width=3

            )

))

fig.update_layout(

    title='Evolution of Active cases over time in India',

        template='plotly_dark'



)



fig.show()
fig = go.Figure()

fig.add_trace(go.Scatter(x=Data_India_op["ObservationDate"], y=Data_India_op['Recovered'],

                    mode="lines+text",

                    name='Recovered cases',

                    marker_color='rgb(229,151,232)',

                        ))



fig.add_annotation(

            x="03/24/2020",

            y=Data_India_op['Recovered'].max(),

            text="COVID-19 pandemic lockdown in India",

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

            x0="03/24/2020",

            y0=Data_India_op['Recovered'].max(),

            x1="03/24/2020",

    

            line=dict(

                color="red",

                width=3

            )

))





fig.add_annotation(

            x="04/24/2020",

            y=Data_India_op['Recovered'].max()-20000,

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

            x0="04/24/2020",

            y0=Data_India_op['Recovered'].max(),

            x1="04/24/2020",

    

            line=dict(

                color="rgb(103,219,165)",

                width=3

            )

))

fig.update_layout(

    title='Evolution of Recovered cases over time in India',

        template='plotly_dark'



)



fig.show()
Data_Germany = data [(data['Country/Region'] == 'Germany') ].reset_index(drop=True)

Data_Germany_op= Data_Germany.groupby(["ObservationDate","Country/Region"])[["Confirmed","Deaths","Recovered","Active_case"]].sum().reset_index().reset_index(drop=True)

fig = go.Figure()

fig.add_trace(go.Scatter(x=Data_Germany_op["ObservationDate"], y=Data_Germany_op['Confirmed'],

                    mode="lines+text",

                    name='Confirmed cases',

                    marker_color='black',

                        ))



fig.add_annotation(

            x="03/23/2020",

            y=Data_Germany_op['Confirmed'].max(),

            text="COVID-19 pandemic lockdown in Germany",

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

            x0="03/23/2020",

            y0=Data_Germany_op['Confirmed'].max(),

            x1="03/23/2020",

    

            line=dict(

                color="red",

                width=3

            )

))

fig.add_annotation(

            x="04/23/2020",

            y=Data_Germany_op['Confirmed'].max()-20000,

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

            x0="04/23/2020",

            y0=Data_Germany_op['Confirmed'].max(),

            x1="04/23/2020",

    

            line=dict(

                color="rgb(209,175,232)",

                width=3

            )

))

fig

fig.update_layout(

    title='Evolution of Confirmed cases over time in Germany',

        template='plotly_white'



)



fig.show()




fig = go.Figure()

fig.add_trace(go.Scatter(x=Data_Germany_op["ObservationDate"], y=Data_Germany_op['Active_case'],

                    mode="lines+text",

                    name='Active cases',

                    marker_color='rgb(79,144,166)',

                        ))



fig.add_annotation(

            x="03/23/2020",

            y=Data_Germany_op['Active_case'].max(),

            text="COVID-19 pandemic lockdown in Germany",

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

            x0="03/23/2020",

            y0=Data_Germany_op['Active_case'].max(),

            x1="03/23/2020",

    

            line=dict(

                color="red",

                width=3

            )

))

fig.add_annotation(

            x="04/23/2020",

            y=Data_Germany_op['Active_case'].max()-9000,

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

            x0="04/23/2020",

            y0=Data_Germany_op['Active_case'].max(),

            x1="04/23/2020",

    

            line=dict(

                color="rgb(92,79,111)",

                width=3

            )

))

fig

fig.update_layout(

    title='Evolution of Active cases over time in Germany',

        template='plotly_white'



)



fig.show()
fig = go.Figure()

fig.add_trace(go.Scatter(x=Data_Germany_op["ObservationDate"], y=Data_Germany_op['Recovered'],

                    mode="lines+text",

                    name='Recovered cases',

                    marker_color='rgb(206,102,232)',

                        ))



fig.add_annotation(

            x="03/23/2020",

            y=Data_Germany_op['Recovered'].max(),

            text="COVID-19 pandemic lockdown in Germany",

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

            x0="03/23/2020",

            y0=Data_Germany_op['Recovered'].max(),

            x1="03/23/2020",

    

            line=dict(

                color="red",

                width=3

            )

))





fig.add_annotation(

            x="04/23/2020",

            y=Data_Germany_op['Recovered'].max()-20000,

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

            x0="04/23/2020",

            y0=Data_Germany_op['Recovered'].max(),

            x1="04/23/2020",

    

            line=dict(

                color="rgb(103,219,165)",

                width=3

            )

))

fig.update_layout(

    title='Evolution of Recovered cases over time in Germany',

        template='plotly_white'



)



fig.show()
Data_Australia = data [(data['Country/Region'] == 'Australia') ].reset_index(drop=True)

Data_Australia_op= Data_Australia.groupby(["ObservationDate","Country/Region"])[["Confirmed","Deaths","Recovered","Active_case"]].sum().reset_index().reset_index(drop=True)

fig = go.Figure()

fig.add_trace(go.Scatter(x=Data_Australia_op["ObservationDate"], y=Data_Australia_op['Confirmed'],

                    mode="lines+text",

                    name='Confirmed cases',

                    marker_color='orange',

                        ))



fig.add_annotation(

            x="03/23/2020",

            y=Data_Australia_op['Confirmed'].max(),

            text="COVID-19 pandemic lockdown in Australia",

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

            x0="03/23/2020",

            y0=Data_Australia_op['Confirmed'].max(),

            x1="03/23/2020",

    

            line=dict(

                color="red",

                width=3

            )

))

fig.add_annotation(

            x="04/23/2020",

            y=Data_Australia_op['Confirmed'].max()-600,

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

            x0="04/23/2020",

            y0=Data_Australia_op['Confirmed'].max(),

            x1="04/23/2020",

    

            line=dict(

                color="#00FE58",

                width=3

            )

))

fig

fig.update_layout(

    title='Evolution of Confirmed cases over time in Australia',

        template='plotly_dark'



)



fig.show()
fig = go.Figure()

fig.add_trace(go.Scatter(x=Data_Australia_op["ObservationDate"], y=Data_Australia_op['Active_case'],

                    mode="lines+text",

                    name='Active cases',

                    marker_color='#00FE58',

                        ))



fig.add_annotation(

            x="03/23/2020",

            y=Data_Australia_op['Active_case'].max(),

            text="COVID-19 pandemic lockdown in Australia",

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

            x0="03/23/2020",

            y0=Data_Australia_op['Active_case'].max(),

            x1="03/23/2020",

    

            line=dict(

                color="red",

                width=3

            )

))





fig.add_annotation(

            x="04/23/2020",

            y=Data_Australia_op['Active_case'].max()-400,

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

            x0="04/23/2020",

            y0=Data_Australia_op['Active_case'].max(),

            x1="04/23/2020",

    

            line=dict(

                color="rgb(255,217,47)",

                width=3

            )

))

fig.update_layout(

    title='Evolution of Active cases over time in Australia',

        template='plotly_dark'



)



fig.show()



fig = go.Figure()

fig.add_trace(go.Scatter(x=Data_Australia_op["ObservationDate"], y=Data_Australia_op['Recovered'],

                    mode="lines+text",

                    name='Recovered cases',

                    marker_color='rgb(229,151,232)',

                        ))



fig.add_annotation(

            x="03/23/2020",

            y=Data_Australia_op['Recovered'].max(),

            text="COVID-19 pandemic lockdown in Australia",

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

            x0="03/23/2020",

            y0=Data_Australia_op['Recovered'].max(),

            x1="03/23/2020",

    

            line=dict(

                color="red",

                width=3

            )

))





fig.add_annotation(

            x="04/23/2020",

            y=Data_Australia_op['Recovered'].max()-600,

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

            x0="04/23/2020",

            y0=Data_Australia_op['Recovered'].max(),

            x1="04/23/2020",

    

            line=dict(

                color="rgb(103,219,165)",

                width=3

            )

))

fig.update_layout(

    title='Evolution of Recovered cases over time in Australia',

        template='plotly_dark'



)



fig.show()
Data_Calif = data [(data['Province/State'] == 'California') ].reset_index(drop=True)

Data_Calif_op= Data_Calif.groupby(["ObservationDate"])[["Confirmed","Deaths","Recovered","Active_case"]].sum().reset_index().reset_index(drop=True)

Data_Calif = data [(data['Province/State'] == 'California') ].reset_index(drop=True)

Data_Calif_op= Data_Calif.groupby(["ObservationDate"])[["Confirmed","Deaths","Recovered","Active_case"]].sum().reset_index().reset_index(drop=True)



fig = go.Figure()

fig.add_trace(go.Scatter(x=Data_Calif_op["ObservationDate"], y=Data_Calif_op['Confirmed'],

                    mode="lines+text",

                    name='Confirmed cases',

                    marker_color='black',

                        ))



fig.add_annotation(

            x="03/19/2020",

            y=Data_Calif_op['Confirmed'].max(),

            text="COVID-19 pandemic lockdown in California",

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

            x0="03/19/2020",

            y0=Data_Calif_op['Confirmed'].max(),

            x1="03/19/2020",

    

            line=dict(

                color="red",

                width=3

            )

))

fig.add_annotation(

            x="04/19/2020",

            y=Data_Calif_op['Confirmed'].max()-20000,

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

            x0="04/19/2020",

            y0=Data_Calif_op['Confirmed'].max(),

            x1="04/19/2020",

    

            line=dict(

                color="rgb(209,175,232)",

                width=3

            )

))

fig

fig.update_layout(

    title='Evolution of Confirmed cases over time in California',

        template='plotly_white'



)



fig.show()



fig = go.Figure()

fig.add_trace(go.Scatter(x=Data_Calif_op["ObservationDate"], y=Data_Calif_op['Active_case'],

                    mode="lines+text",

                    name='Active cases',

                    marker_color='rgb(79,144,166)',

                        ))



fig.add_annotation(

            x="03/19/2020",

            y=Data_Calif_op['Active_case'].max(),

            text="COVID-19 pandemic lockdown in California",

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

            x0="03/19/2020",

            y0=Data_Calif_op['Active_case'].max(),

            x1="03/19/2020",

    

            line=dict(

                color="red",

                width=3

            )

))

fig.add_annotation(

            x="04/19/2020",

            y=Data_Calif_op['Active_case'].max()-9000,

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

            x0="04/19/2020",

            y0=Data_Calif_op['Active_case'].max(),

            x1="04/19/2020",

    

            line=dict(

                color="rgb(92,79,111)",

                width=3

            )

))

fig

fig.update_layout(

    title='Evolution of Active cases over time in California',

        template='plotly_white'



)



fig.show()



Data_NewYork = data [(data['Province/State'] == 'New York') ].reset_index(drop=True)

Data_NewYork_op= Data_NewYork.groupby(["ObservationDate"])[["Confirmed","Deaths","Recovered","Active_case"]].sum().reset_index().reset_index(drop=True)



fig = go.Figure()

fig.add_trace(go.Scatter(x=Data_NewYork_op["ObservationDate"], y=Data_NewYork_op['Confirmed'],

                    mode="lines+text",

                    name='Confirmed cases',

                    marker_color='orange',

                        ))



fig.add_annotation(

            x="03/22/2020",

            y=Data_NewYork_op['Confirmed'].max(),

            text="COVID-19 pandemic lockdown in New York",

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

            x0="03/22/2020",

            y0=Data_NewYork_op['Confirmed'].max(),

            x1="03/22/2020",

    

            line=dict(

                color="red",

                width=3

            )

))

fig.add_annotation(

            x="04/23/2020",

            y=Data_NewYork_op['Confirmed'].max()-20000,

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

            x0="04/22/2020",

            y0=Data_NewYork_op['Confirmed'].max(),

            x1="04/22/2020",

    

            line=dict(

                color="#00FE58",

                width=3

            )

))

fig

fig.update_layout(

    title='Evolution of Confirmed cases over time in New York',

        template='plotly_dark'



)



fig.show()











fig = go.Figure()

fig.add_trace(go.Scatter(x=Data_NewYork_op["ObservationDate"], y=Data_NewYork_op['Active_case'],

                    mode="lines+text",

                    name='Active cases',

                    marker_color='#00FE58',

                        ))



fig.add_annotation(

            x="03/22/2020",

            y=Data_NewYork_op['Active_case'].max(),

            text="COVID-19 pandemic lockdown in New York",

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

            x0="03/22/2020",

            y0=Data_NewYork_op['Active_case'].max(),

            x1="03/22/2020",

    

            line=dict(

                color="red",

                width=3

            )

))





fig.add_annotation(

            x="04/22/2020",

            y=Data_NewYork_op['Active_case'].max()-20000,

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

            x0="04/22/2020",

            y0=Data_NewYork_op['Active_case'].max(),

            x1="04/22/2020",

    

            line=dict(

                color="rgb(255,217,47)",

                width=3

            )

))

fig.update_layout(

    title='Evolution of Active cases over time in New York',

        template='plotly_dark'



)



fig.show()

Data_Michigan = data [(data['Province/State'] == 'Michigan') ].reset_index(drop=True)

Data_Michigan_op= Data_Michigan.groupby(["ObservationDate"])[["Confirmed","Deaths","Recovered","Active_case"]].sum().reset_index().reset_index(drop=True)



fig = go.Figure()

fig.add_trace(go.Scatter(x=Data_Michigan_op["ObservationDate"], y=Data_Michigan_op['Confirmed'],

                    mode="lines+text",

                    name='Confirmed cases',

                    marker_color='black',

                        ))



fig.add_annotation(

            x="03/24/2020",

            y=Data_Michigan_op['Confirmed'].max(),

            text="COVID-19 pandemic lockdown in Michigan",

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

            x0="03/24/2020",

            y0=Data_Michigan_op['Confirmed'].max(),

            x1="03/24/2020",

    

            line=dict(

                color="red",

                width=3

            )

))

fig.add_annotation(

            x="04/24/2020",

            y=Data_Michigan_op['Confirmed'].max()-10000,

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

            x0="04/24/2020",

            y0=Data_Michigan_op['Confirmed'].max(),

            x1="04/24/2020",

    

            line=dict(

                color="rgb(209,175,232)",

                width=3

            )

))

fig

fig.update_layout(

    title='Evolution of Confirmed cases over time in Michigan',

        template='plotly_white'



)



fig.show()



fig = go.Figure()

fig.add_trace(go.Scatter(x=Data_Michigan_op["ObservationDate"], y=Data_Michigan_op['Active_case'],

                    mode="lines+text",

                    name='Active cases',

                    marker_color='rgb(79,144,166)',

                        ))



fig.add_annotation(

            x="03/24/2020",

            y=Data_Michigan_op['Active_case'].max(),

            text="COVID-19 pandemic lockdown in Michigan",

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

            x0="03/24/2020",

            y0=Data_Michigan_op['Active_case'].max(),

            x1="03/24/2020",

    

            line=dict(

                color="red",

                width=3

            )

))

fig.add_annotation(

            x="04/24/2020",

            y=Data_Michigan_op['Active_case'].max()-9000,

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

            x0="04/24/2020",

            y0=Data_Michigan_op['Active_case'].max(),

            x1="04/24/2020",

    

            line=dict(

                color="rgb(92,79,111)",

                width=3

            )

))

fig

fig.update_layout(

    title='Evolution of Active cases over time in Michigan',

        template='plotly_white'



)



fig.show()

Data_Oregon = data [(data['Province/State'] == 'Oregon') ].reset_index(drop=True)

Data_Oregon_op= Data_Oregon.groupby(["ObservationDate"])[["Confirmed","Deaths","Recovered","Active_case"]].sum().reset_index().reset_index(drop=True)



fig = go.Figure()

fig.add_trace(go.Scatter(x=Data_Oregon_op["ObservationDate"], y=Data_Oregon_op['Confirmed'],

                    mode="lines+text",

                    name='Confirmed cases',

                    marker_color='#00FE58',

                        ))



fig.add_annotation(

            x="03/24/2020",

            y=Data_Oregon_op['Confirmed'].max(),

            text="COVID-19 pandemic lockdown in Oregon",

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

            x0="03/24/2020",

            y0=Data_Oregon_op['Confirmed'].max(),

            x1="03/24/2020",

    

            line=dict(

                color="red",

                width=3

            )

))

fig.add_annotation(

            x="04/24/2020",

            y=Data_Oregon_op['Confirmed'].max()-4000,

            text="Month after lockdown",

             font=dict(

            family="Courier New, monospace",

            size=16,

            color="yellow"

            ),

)



fig.add_shape(

        # Line Vertical

        dict(

            type="line",

            x0="04/24/2020",

            y0=Data_Oregon_op['Confirmed'].max(),

            x1="04/24/2020",

    

            line=dict(

                color="yellow",

                width=3

            )

))

fig

fig.update_layout(

    title='Evolution of Confirmed cases over time in Oregon',

        template='plotly_dark'



)



fig.show()



fig = go.Figure()

fig.add_trace(go.Scatter(x=Data_Oregon_op["ObservationDate"], y=Data_Oregon_op['Active_case'],

                    mode="lines+text",

                    name='Active cases',

                    marker_color='orange',

                        ))



fig.add_annotation(

            x="03/24/2020",

            y=Data_Oregon_op['Active_case'].max(),

            text="COVID-19 pandemic lockdown in Oregon",

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

            x0="03/24/2020",

            y0=Data_Oregon_op['Active_case'].max(),

            x1="03/24/2020",

    

            line=dict(

                color="red",

                width=3

            )

))

fig.add_annotation(

            x="04/24/2020",

            y=Data_Oregon_op['Active_case'].max()-4000,

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

            x0="04/24/2020",

            y0=Data_Oregon_op['Active_case'].max(),

            x1="04/24/2020",

    

            line=dict(

                color="#00FE58",

                width=3

            )

))

fig

fig.update_layout(

    title='Evolution of Active cases over time in Oregon',

        template='plotly_dark'



)



fig.show()
Data_Belgium = data [(data['Country/Region'] == 'Belgium') ].reset_index(drop=True)

Data_Belgium_op= Data_Belgium.groupby(["ObservationDate","Country/Region"])[["Confirmed","Deaths","Recovered","Active_case"]].sum().reset_index().reset_index(drop=True)





fig = go.Figure()

fig.add_trace(go.Scatter(x=Data_Belgium_op["ObservationDate"], y=Data_Belgium_op['Confirmed'],

                    mode="lines+text",

                    name='Confirmed cases',

                    marker_color='black',

                        ))



fig.add_annotation(

            x="03/18/2020",

            y=Data_Belgium_op['Confirmed'].max(),

            text="COVID-19 pandemic lockdown in Belgium",

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

            x0="03/18/2020",

            y0=Data_Belgium_op['Confirmed'].max(),

            x1="03/18/2020",

    

            line=dict(

                color="red",

                width=3

            )

))

fig.add_annotation(

            x="04/18/2020",

            y=Data_Belgium_op['Confirmed'].max()-8000,

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

            x0="04/18/2020",

            y0=Data_Belgium_op['Confirmed'].max(),

            x1="04/18/2020",

    

            line=dict(

                color="rgb(209,175,232)",

                width=3

            )

))

fig

fig.update_layout(

    title='Evolution of Confirmed cases over time in Belgium',

        template='plotly_white'



)



fig.show()



fig = go.Figure()

fig.add_trace(go.Scatter(x=Data_Belgium_op["ObservationDate"], y=Data_Belgium_op['Active_case'],

                    mode="lines+text",

                    name='Active cases',

                    marker_color='rgb(79,144,166)',

                        ))



fig.add_annotation(

            x="03/18/2020",

            y=Data_Belgium_op['Active_case'].max(),

            text="COVID-19 pandemic lockdown in Belgium",

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

            x0="03/18/2020",

            y0=Data_Belgium_op['Active_case'].max(),

            x1="03/18/2020",

    

            line=dict(

                color="red",

                width=3

            )

))

fig.add_annotation(

            x="04/18/2020",

            y=Data_Belgium_op['Active_case'].max()-9000,

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

            x0="04/18/2020",

            y0=Data_Belgium_op['Active_case'].max(),

            x1="04/18/2020",

    

            line=dict(

                color="rgb(92,79,111)",

                width=3

            )

))

fig

fig.update_layout(

    title='Evolution of Active cases over time in Belgium',

        template='plotly_white'



)



fig.show()



fig = go.Figure()

fig.add_trace(go.Scatter(x=Data_Belgium_op["ObservationDate"], y=Data_Belgium_op['Recovered'],

                    mode="lines+text",

                    name='Recovered cases',

                    marker_color='rgb(206,102,232)',

                        ))



fig.add_annotation(

            x="03/18/2020",

            y=Data_Belgium_op['Recovered'].max(),

            text="COVID-19 pandemic lockdown in Belgium",

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

            x0="03/18/2020",

            y0=Data_Belgium_op['Recovered'].max(),

            x1="03/18/2020",

    

            line=dict(

                color="red",

                width=3

            )

))





fig.add_annotation(

            x="04/18/2020",

            y=Data_Belgium_op['Recovered'].max()-5000,

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

            x0="04/18/2020",

            y0=Data_Belgium_op['Recovered'].max(),

            x1="04/18/2020",

    

            line=dict(

                color="rgb(103,219,165)",

                width=3

            )

))

fig.update_layout(

    title='Evolution of Recovered cases over time in Belgium',

        template='plotly_white'



)



fig.show()
Data_Czech = data [(data['Country/Region'] == 'Czech Republic') ].reset_index(drop=True)

Data_Czech_op= Data_Czech.groupby(["ObservationDate","Country/Region"])[["Confirmed","Deaths","Recovered","Active_case"]].sum().reset_index().reset_index(drop=True)

fig = go.Figure()

fig.add_trace(go.Scatter(x=Data_Czech_op["ObservationDate"], y=Data_Czech_op['Confirmed'],

                    mode="lines+text",

                    name='Confirmed cases',

                    marker_color='orange',

                        ))



fig.add_annotation(

            x="03/16/2020",

            y=Data_Czech_op['Confirmed'].max(),

            text="COVID-19 pandemic lockdown in Czech Republic",

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

            x0="03/16/2020",

            y0=Data_Czech_op['Confirmed'].max(),

            x1="03/16/2020",

    

            line=dict(

                color="red",

                width=3

            )

))

fig.add_annotation(

            x="04/16/2020",

            y=Data_Czech_op['Confirmed'].max()-2000,

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

            x0="04/16/2020",

            y0=Data_Czech_op['Confirmed'].max(),

            x1="04/16/2020",

    

            line=dict(

                color="#00FE58",

                width=3

            )

))

fig

fig.update_layout(

    title='Evolution of Confirmed cases over time in Czech Republic',

        template='plotly_dark'



)



fig.show()



fig = go.Figure()

fig.add_trace(go.Scatter(x=Data_Czech_op["ObservationDate"], y=Data_Czech_op['Active_case'],

                    mode="lines+text",

                    name='Active cases',

                    marker_color='#00FE58',

                        ))



fig.add_annotation(

            x="03/16/2020",

            y=Data_Czech_op['Active_case'].max(),

            text="COVID-19 pandemic lockdown in Czech Republic",

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

            x0="03/16/2020",

            y0=Data_Czech_op['Active_case'].max(),

            x1="03/16/2020",

    

            line=dict(

                color="red",

                width=3

            )

))





fig.add_annotation(

            x="04/16/2020",

            y=Data_Czech_op['Active_case'].min(),

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

            x0="04/16/2020",

            y0=Data_Czech_op['Active_case'].max(),

            x1="04/16/2020",

    

            line=dict(

                color="rgb(255,217,47)",

                width=3

            )

))

fig.update_layout(

    title='Evolution of Active cases over time in Czech Republic',

        template='plotly_dark'



)



fig.show()



fig = go.Figure()

fig.add_trace(go.Scatter(x=Data_Czech_op["ObservationDate"], y=Data_Czech_op['Recovered'],

                    mode="lines+text",

                    name='Recovered cases',

                    marker_color='rgb(229,151,232)',

                        ))



fig.add_annotation(

            x="03/16/2020",

            y=Data_Czech_op['Recovered'].max(),

            text="COVID-19 pandemic lockdown in Czech Republic",

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

            x0="03/16/2020",

            y0=Data_Czech_op['Recovered'].max(),

            x1="03/16/2020",

    

            line=dict(

                color="red",

                width=3

            )

))





fig.add_annotation(

            x="04/16/2020",

            y=Data_Czech_op['Recovered'].min(),

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

            x0="04/16/2020",

            y0=Data_Czech_op['Recovered'].max(),

            x1="04/16/2020",

    

            line=dict(

                color="rgb(103,219,165)",

                width=3

            )

))

fig.update_layout(

    title='Evolution of Recovered cases over time in Czech Republic',

        template='plotly_dark'



)



fig.show()
Data_Portugal = data [(data['Country/Region'] == 'Portugal') ].reset_index(drop=True)

Data_Portugal_op= Data_Portugal.groupby(["ObservationDate","Country/Region"])[["Confirmed","Deaths","Recovered","Active_case"]].sum().reset_index().reset_index(drop=True)

fig = go.Figure()

fig.add_trace(go.Scatter(x=Data_Portugal_op["ObservationDate"], y=Data_Portugal_op['Confirmed'],

                    mode="lines+text",

                    name='Confirmed cases',

                    marker_color='black',

                        ))



fig.add_annotation(

            x="03/19/2020",

            y=Data_Portugal_op['Confirmed'].max(),

            text="COVID-19 pandemic lockdown in Portugal",

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

            x0="03/19/2020",

            y0=Data_Portugal_op['Confirmed'].max(),

            x1="03/19/2020",

    

            line=dict(

                color="red",

                width=3

            )

))

fig.add_annotation(

            x="04/19/2020",

            y=Data_Portugal_op['Confirmed'].max()-8000,

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

            x0="04/19/2020",

            y0=Data_Portugal_op['Confirmed'].max(),

            x1="04/19/2020",

    

            line=dict(

                color="rgb(209,175,232)",

                width=3

            )

))

fig

fig.update_layout(

    title='Evolution of Confirmed cases over time in Portugal',

        template='plotly_white'



)



fig.show()

fig = go.Figure()

fig.add_trace(go.Scatter(x=Data_Portugal_op["ObservationDate"], y=Data_Portugal_op['Active_case'],

                    mode="lines+text",

                    name='Active cases',

                    marker_color='rgb(79,144,166)',

                        ))



fig.add_annotation(

            x="03/19/2020",

            y=Data_Portugal_op['Active_case'].max(),

            text="COVID-19 pandemic lockdown in Portugal",

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

            x0="03/19/2020",

            y0=Data_Portugal_op['Active_case'].max(),

            x1="03/19/2020",

    

            line=dict(

                color="red",

                width=3

            )

))

fig.add_annotation(

            x="04/19/2020",

            y=Data_Portugal_op['Active_case'].max()-9000,

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

            x0="04/19/2020",

            y0=Data_Portugal_op['Active_case'].max(),

            x1="04/19/2020",

    

            line=dict(

                color="rgb(92,79,111)",

                width=3

            )

))

fig

fig.update_layout(

    title='Evolution of Active cases over time in Portugal',

        template='plotly_white'



)



fig.show()

fig = go.Figure()

fig.add_trace(go.Scatter(x=Data_Portugal_op["ObservationDate"], y=Data_Portugal_op['Recovered'],

                    mode="lines+text",

                    name='Recovered cases',

                    marker_color='rgb(206,102,232)',

                        ))



fig.add_annotation(

            x="03/19/2020",

            y=Data_Portugal_op['Recovered'].max(),

            text="COVID-19 pandemic lockdown in Portugal",

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

            x0="03/19/2020",

            y0=Data_Portugal_op['Recovered'].max(),

            x1="03/19/2020",

    

            line=dict(

                color="red",

                width=3

            )

))





fig.add_annotation(

            x="04/19/2020",

            y=Data_Portugal_op['Recovered'].max()-5000,

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

            x0="04/19/2020",

            y0=Data_Portugal_op['Recovered'].max(),

            x1="04/19/2020",

    

            line=dict(

                color="rgb(103,219,165)",

                width=3

            )

))

fig.update_layout(

    title='Evolution of Recovered cases over time in Portugal',

        template='plotly_white'



)



fig.show()