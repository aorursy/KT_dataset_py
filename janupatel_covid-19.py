

#storing and analysis

import pandas as pd



#visualization

import plotly.graph_objs as go

from plotly.subplots import make_subplots

import plotly



worldwide = pd.read_csv("../input/covid-2019-india/corona_worldwide.csv")

print(worldwide.head())



print("\ntable columns\n", worldwide.columns)





#donut chart(world/india)

labels = ['Recovered', 'Deaths', 'Active cases']

values_world = [

            worldwide[worldwide['Location'] == 'Worldwide']['Recovered'][0],

            worldwide[worldwide['Location'] == 'Worldwide']['Deaths'][0],

            worldwide[worldwide['Location'] == 'Worldwide']['Active cases'][0]

        ]

values_india = [

            worldwide[worldwide['Location'] == 'India']['Recovered'][1],

            worldwide[worldwide['Location'] == 'India']['Deaths'][1],

            worldwide[worldwide['Location'] == 'India']['Active cases'][1]

        ]



# Create subplots: use 'domain' type for Pie subplot

fig = make_subplots(rows=1, cols=2, specs=[[{'type':'domain'}, {'type':'domain'}]])

fig.add_trace(go.Pie(labels=labels, values=values_world, name="worldwide"),

              1, 1)

fig.add_trace(go.Pie(labels=labels, values=values_india, name="India"),

              1, 2)



# Use `hole` to create a donut-like pie chart

fig.update_traces(hole=0.6, hoverinfo="label+percent+name")



fig.update_layout(

    title_text="coronavirus cases",

    # Add annotations in the center of the donut pies.

    annotations=[dict(text='worldwide', x=0.18, y=0.5, font_size=20, showarrow=False),

                 dict(text='India', x=0.82, y=0.5, font_size=20, showarrow=False)])

fig.show()



table = pd.read_csv("../input/covid-2019-india/corona(case_time_series).csv")

# print("\ntable\n", table.head())

# print("\ntable columns\n", table.columns)



#bar graph(daily cases)

fig = go.Figure()

fig.add_trace(go.Bar(x=table['Date'], y=table['Daily Confirmed'],name='Daily Confirmed'))

fig.add_trace(go.Bar(x=table['Date'], y=table['Daily Deceased'],name='Daily Deceased'))

fig.add_trace(go.Bar(x=table['Date'], y=table['Daily Recovered'],name='Daily Recovered'))

fig.update_layout(barmode='stack', title_text='Daily cases in india')

fig.show()



#bar graph(total cases)

fig = go.Figure()

fig.add_trace(go.Bar(x=table['Date'], y=table['Total Confirmed'],name='Total Confirmed'))

fig.add_trace(go.Bar(x=table['Date'], y=table['Total Deceased'],name='Total Deceased'))

fig.add_trace(go.Bar(x=table['Date'], y=table['Total Recovered'],name='Total Recovered'))

fig.update_layout(barmode='stack', title_text='Total cases in india')

fig.show()



table = pd.read_csv("../input/covid-2019-india/corona(statewise).csv")

table.drop('Unnamed: 1',axis = 1,inplace = True)

print("\ntable columns\n", table.columns)

print("\ntable\n", table.head())



#bar graph(state wise cases)

fig = go.Figure()

fig.add_trace(go.Bar(x=table['State'], y=table['Confirmed'],name='Confirmed'))

fig.add_trace(go.Bar(x=table['State'], y=table['Deaths'],name='Deaths'))

fig.add_trace(go.Bar(x=table['State'], y=table['Recovered'],name='Recovered'))

fig.update_layout(barmode='stack', title_text='atate wise cases')

fig.show()