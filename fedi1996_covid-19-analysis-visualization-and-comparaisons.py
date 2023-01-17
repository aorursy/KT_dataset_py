import numpy as np 

import pandas as pd 

import matplotlib.pyplot as plt

import plotly.graph_objects as go

import plotly.express as px

pd.set_option('display.max_rows', None)

from plotly.subplots import make_subplots

import seaborn as sns

import datetime
data = pd.read_csv('../input/novel-corona-virus-2019-dataset/covid_19_data.csv')

#Percentage of NAN Values 

NAN = [(c, data[c].isna().mean()*100) for c in data]

NAN = pd.DataFrame(NAN, columns=["column_name", "percentage"])

NAN
data["Province/State"]= data["Province/State"].fillna('Unknown')

data.head()
data[["Confirmed","Deaths","Recovered"]] =data[["Confirmed","Deaths","Recovered"]].astype(int)
data['Country/Region'] = data['Country/Region'].replace('Mainland China', 'China')
data['Active_case'] = data['Confirmed'] - data['Deaths'] - data['Recovered']

data.head()
Data = data[data['ObservationDate'] == max(data['ObservationDate'])].reset_index()

Data_world = Data.groupby(["ObservationDate"])[["Confirmed","Active_case","Recovered","Deaths"]].sum().reset_index()
labels = ["Last Update","Confirmed","Active cases","Recovered","Deaths"]

fig = go.Figure(data=[go.Table(header=dict(values=labels),

                 cells=dict(values=Data_world.loc[0,["ObservationDate","Confirmed","Active_case","Recovered","Deaths"]]))

                     ])

fig.update_layout(

    title='Coronavirus in the word : ',

)

fig.show()

labels = ["Active cases","Recovered","Deaths"]

values = Data_world.loc[0, ["Active_case","Recovered","Deaths"]]

fig = px.pie(Data_world, values=values, names=labels,color_discrete_sequence=['rgb(77,146,33)','rgb(69,144,185)','rgb(77,77,77)'],hole=0.7)

fig.update_layout(

    title='Total cases : '+str(Data_world["Confirmed"][0]),

)

fig.show()
data_over_time= data.groupby(["ObservationDate"])[["Confirmed","Active_case","Recovered","Deaths"]].sum().reset_index().sort_values("ObservationDate",ascending=True).reset_index(drop=True)



fig = go.Figure()

fig.add_trace(go.Scatter(x=data_over_time.index, y=data_over_time['Confirmed'],

                    mode='lines',

                    name='Confirmed cases'))





fig.update_layout(

    title='Evolution of Confirmed cases over time in the word',

        template='plotly_white',

      yaxis_title="Confirmed cases",

    xaxis_title="Days",



)



fig.show()
fig = go.Figure()





fig.add_trace(go.Scatter(x=data_over_time.index, y=data_over_time['Active_case'],

                    mode='lines',marker_color='yellow',

                    name='Active cases',line=dict( dash='dot')))



fig.update_layout(

    title='Evolution of Active cases over time in the world',

        template='plotly_dark',

      yaxis_title="Active cases",

    xaxis_title="Days",



)



fig.show()



fig = go.Figure()



fig.add_trace(go.Scatter(x=data_over_time.index, y=data_over_time['Recovered'],

                    mode='lines',

                    name='Recovered cases',marker_color='green'))



fig.update_layout(

    title='Evolution of Recovered cases over time in the world',

        template='plotly_white',

      yaxis_title="Recovered cases",

    xaxis_title="Days",



)



fig.show()





fig = go.Figure()



fig.add_trace(go.Scatter(x=data_over_time.index, y=data_over_time['Deaths'],name='Deaths',

                                   marker_color='black',mode='lines',line=dict( dash='dot') ))



fig.update_layout(

    title='Evolution of Deaths over time in the world',

        template='plotly_white',

     yaxis_title="Deaths",

    xaxis_title="Days",



)



fig.show()


fig = go.Figure(go.Bar(

            x=data_over_time['ObservationDate'],

            y=data_over_time['Confirmed'],

           ))

fig.update_layout(

    title='Confirmed Cases In Each Day',

    template='plotly_white',

     xaxis_title="Confirmed Cases",

    yaxis_title="Days",

)

fig.show()
fig = go.Figure(go.Bar(

            x=data_over_time['ObservationDate'],

            y=data_over_time['Active_case'],

    marker_color='rgb(253,187,132)'

           ))

fig.update_layout(

    title='Active Cases In Each Day',

    template='plotly_dark',

     xaxis_title="Active Cases",

    yaxis_title="Days",

)

fig.show()
fig = go.Figure(go.Bar(

            x=data_over_time['ObservationDate'],

            y=data_over_time['Recovered'],

    marker_color='rgb(178,24,43)'

           ))

fig.update_layout(

    title='Recovered Cases In Each Day',

    template='plotly_white',

     xaxis_title="Recovered Cases",

    yaxis_title="Days",

)

fig.show()
fig = go.Figure(go.Bar(

            x=data_over_time['ObservationDate'],

            y=data_over_time['Deaths'],

    marker_color='rgb(13,48,100)'

           ))

fig.update_layout(

    title='Deaths In Each Day',

    template='plotly_white',

     xaxis_title="Deaths",

    yaxis_title="Days",

)

fig.show()
Data_per_country = Data.groupby(["Country/Region"])["Confirmed","Active_case","Recovered","Deaths"].sum().reset_index().sort_values("Confirmed",ascending=False).reset_index(drop=True)

headerColor = 'grey'

rowEvenColor = 'lightgrey'

rowOddColor = 'white'



fig = go.Figure(data=[go.Table(

  header=dict(

    values=['<b>Country</b>','<b>Confirmed Cases</b>'],

    line_color='darkslategray',

    fill_color=headerColor,

    align=['left','center'],

      

    font=dict(color='white', size=12)

  ),

  cells=dict(

    values=[

      Data_per_country['Country/Region'],

      Data_per_country['Confirmed'],

      ],

    line_color='darkslategray',

    # 2-D list of colors for alternating rows

    fill_color = [[rowOddColor,rowEvenColor,rowOddColor, rowEvenColor,rowOddColor]*len(Data_per_country)],

    align = ['left', 'center'],

    font = dict(color = 'darkslategray', size = 11)

    ))

])

fig.update_layout(

    title='Confirmed Cases In Each Country',

)

fig.show()


fig = go.Figure(go.Bar(

            x=Data_per_country['Confirmed'],

            y=Data_per_country['Country/Region'],

            orientation='h'))

fig.update_layout(

    title='Confirmed Cases In Each Country',

    template='plotly_white',

     xaxis_title="Confirmed Cases",

    yaxis_title="Countries",

)

fig.show()
fig = go.Figure(go.Bar(

            x=Data_per_country['Active_case'],

            y=Data_per_country['Country/Region'],

            orientation='h',

            marker_color='#DC3912',))

fig.update_layout(

    title='Active Cases In Each Country',

    template='plotly_white',

    xaxis_title="Active Cases",

    yaxis_title="Countries",

)

fig.show()
fig = go.Figure(go.Bar(

            x=Data_per_country['Recovered'],

            y=Data_per_country['Country/Region'],

            orientation='h',

            marker_color='#2CA02C',))

fig.update_layout(

    title='Recovered Cases In Each Country',

    template='plotly_white',

     xaxis_title="Recovered Cases",

    yaxis_title="Countries",

)

fig.show()
fig = go.Figure(go.Bar(

            x=Data_per_country['Deaths'],

            y=Data_per_country['Country/Region'],

            orientation='h',

            marker_color='black',))

fig.update_layout(

    title='Deaths In Each Country',

    template='plotly_white',

    xaxis_title="Deaths",

    yaxis_title="Countries",

)



fig.show()
fig = px.choropleth(Data_per_country, locations=Data_per_country['Country/Region'],

                    color=Data_per_country['Confirmed'],locationmode='country names', 

                    hover_name=Data_per_country['Country/Region'], 

                    color_continuous_scale=px.colors.sequential.Tealgrn,template='plotly_dark', )

fig.update_layout(

    title='Confirmed Cases In Each Country',

)

fig.show()
fig = px.choropleth(Data_per_country, locations=Data_per_country['Country/Region'],

                    color=Data_per_country['Active_case'],locationmode='country names', 

                    hover_name=Data_per_country['Country/Region'], 

                    color_continuous_scale=px.colors.sequential.Tealgrn,template='plotly_white', )

fig.update_layout(

    title='Active Cases In Each Country',

)

fig.show()
fig = px.choropleth(Data_per_country, locations=Data_per_country['Country/Region'],

                    color=Data_per_country['Recovered'],locationmode='country names', 

                    hover_name=Data_per_country['Country/Region'], 

                    color_continuous_scale=px.colors.sequential.Tealgrn,template='plotly_white', )

fig.update_layout(

    title='Recovered Cases In Each Country',

)

fig.show()
fig = px.choropleth(Data_per_country, locations=Data_per_country['Country/Region'],

                    color=Data_per_country['Deaths'],locationmode='country names', 

                    hover_name=Data_per_country['Country/Region'], 

                    color_continuous_scale=px.colors.sequential.Tealgrn,template='plotly_dark', )

fig.update_layout(

    title='Deaths In Each Country',

)

fig.show()
data_per_country = data.groupby(["Country/Region","ObservationDate"])[["Confirmed","Active_case","Recovered","Deaths"]].sum().reset_index().sort_values("ObservationDate",ascending=True).reset_index(drop=True)

fig = px.choropleth(data_per_country, locations=data_per_country['Country/Region'],

                    color=data_per_country['Confirmed'],locationmode='country names', 

                    hover_name=data_per_country['Country/Region'], 

                    color_continuous_scale=px.colors.sequential.deep,

                    animation_frame="ObservationDate")

fig.update_layout(



    title='Evolution of confirmed cases In Each Country',

)

fig.show()
fig = px.choropleth(data_per_country, locations=data_per_country['Country/Region'],

                    color=data_per_country['Active_case'],locationmode='country names', 

                    hover_name=data_per_country['Country/Region'], 

                    color_continuous_scale=px.colors.sequential.Tealgrn,

                    animation_frame="ObservationDate")

fig.update_layout(



    title='Evolution of active cases In Each Country',

    template='plotly_dark'

)

fig.show()
fig = px.choropleth(data_per_country, locations=data_per_country['Country/Region'],

                    color=data_per_country['Recovered'],locationmode='country names', 

                    hover_name=data_per_country['Country/Region'], 

                    color_continuous_scale=px.colors.sequential.deep,

                    animation_frame="ObservationDate")

fig.update_layout(

    title='Evolution of recovered cases In Each Country',

)

fig.show()
fig = px.choropleth(data_per_country, locations=data_per_country['Country/Region'],

                    color=data_per_country['Deaths'],locationmode='country names', 

                    hover_name=data_per_country['Country/Region'], 

                    color_continuous_scale=px.colors.sequential.Tealgrn,

                    animation_frame="ObservationDate")

fig.update_layout(

    title='Evolution of deaths In Each Country',

    template='plotly_dark'

)

fig.show()
fig = go.Figure(data=[go.Bar(

            x=Data_per_country['Country/Region'][0:10], y=Data_per_country['Confirmed'][0:10],

            text=Data_per_country['Confirmed'][0:10],

            textposition='auto',

            marker_color='black',

            



        )])

fig.update_layout(

    title='Most 10 infected Countries',

    xaxis_title="Countries",

    yaxis_title="Confirmed Cases",

        template='plotly_white'



)

fig.show()
fig = go.Figure(data=[go.Scatter(

    x=Data_per_country['Country/Region'][0:10],

    y=Data_per_country['Confirmed'][0:10],

    mode='markers',

    

    marker=dict(

        color=100+np.random.randn(500),

        size=(Data_per_country['Confirmed'][0:10]/25000),

        showscale=True

        )

)])



fig.update_layout(

    title='Most 10 infected Countries',

    xaxis_title="Countries",

    yaxis_title="Confirmed Cases",

    template='plotly_dark'

)

fig.show()
Recovered_per_country = Data.groupby(["Country/Region"])["Recovered"].sum().reset_index().sort_values("Recovered",ascending=False).reset_index(drop=True)

headerColor = 'grey'

rowEvenColor = 'lightgrey'

rowOddColor = 'white'



fig = go.Figure(data=[go.Table(

  header=dict(

    values=['<b>Country</b>','<b>Recovered Cases</b>'],

    line_color='darkslategray',

    fill_color=headerColor,

    align=['left','center'],

    font=dict(color='white', size=12)

  ),

  cells=dict(

    values=[

      Recovered_per_country['Country/Region'],

      Recovered_per_country['Recovered'],

      ],

    line_color='darkslategray',

    # 2-D list of colors for alternating rows

    fill_color = [[rowOddColor,rowEvenColor,rowOddColor, rowEvenColor,rowOddColor]*len(Data_per_country)],

    align = ['left', 'center'],

    font = dict(color = 'darkslategray', size = 11)

    ))

])

fig.update_layout(

    title='Recovered Cases In Each Country',

)

fig.show()
fig = px.pie(Recovered_per_country, values=Recovered_per_country['Recovered'], names=Recovered_per_country['Country/Region'],

             title='Recovered cases',

            )

fig.update_traces(textposition='inside', textinfo='percent+label')

fig.update_layout(

    template='plotly_white'

)

fig.show()
fig = go.Figure(data=[go.Bar(

            x=Recovered_per_country['Country/Region'][0:10], y=Recovered_per_country['Recovered'][0:10],

            text=Recovered_per_country['Recovered'][0:10],

            textposition='auto',

            marker_color='green',



        )])

fig.update_layout(

    title='Most 10 infected Countries',

    xaxis_title="Countries",

    yaxis_title="Recovered Cases",

    template='plotly_white'

)

fig.show()

fig = go.Figure(data=[go.Scatter(

    x=Recovered_per_country['Country/Region'][0:10],

    y=Recovered_per_country['Recovered'][0:10],

    mode='markers',

    marker=dict(

        color=100+np.random.randn(500),

        size=(Data_per_country['Recovered'][0:10]/20000),

        showscale=True

        )

)])

fig.update_layout(

    title='Most 10 infected Countries',

    xaxis_title="Countries",

    yaxis_title="Recovered Cases",

    template='plotly_white'



)

fig.show()
Active_per_country = Data.groupby(["Country/Region"])["Active_case"].sum().reset_index().sort_values("Active_case",ascending=False).reset_index(drop=True)

headerColor = 'grey'

rowEvenColor = 'lightgrey'

rowOddColor = 'white'



fig = go.Figure(data=[go.Table(

  header=dict(

    values=['<b>Country</b>','<b>Active Cases</b>'],

    line_color='darkslategray',

    fill_color=headerColor,

    align=['left','center'],

    font=dict(color='white', size=12)

  ),

  cells=dict(

    values=[

      Active_per_country['Country/Region'],

      Active_per_country['Active_case'],

      ],

    line_color='darkslategray',

    # 2-D list of colors for alternating rows

    fill_color = [[rowOddColor,rowEvenColor,rowOddColor, rowEvenColor,rowOddColor]*len(Data_per_country)],

    align = ['left', 'center'],

    font = dict(color = 'darkslategray', size = 11)

    ))

])

fig.update_layout(

    title='Active Cases In Each Country',

)

fig.show()
fig = go.Figure(data=[go.Bar(

            x=Active_per_country['Country/Region'][0:10], y=Active_per_country['Active_case'][0:10],

            text=Active_per_country['Active_case'][0:10],

           

        )])

fig.update_layout(

    title='Most 10 infected Countries',

    xaxis_title="Countries",

    yaxis_title="Active Cases",

    template='plotly_white'

)

fig.show()

fig = go.Figure(data=[go.Scatter(

    x=Active_per_country['Country/Region'][0:10],

    y=Active_per_country['Active_case'][0:10],

    mode='markers',

    marker=dict(

        color=10+np.random.randn(200),



        size=Active_per_country['Active_case'][0:10]/15000,

        showscale=True

        )

)])

fig.update_layout(

    title='Most 10 infected Countries',

    xaxis_title="Countries",

    yaxis_title="Active Cases",

        template='plotly_white'



)

fig.show()
Deaths_per_country = Data.groupby(["Country/Region"])["Deaths"].sum().reset_index().sort_values("Deaths",ascending=False).reset_index(drop=True)

headerColor = 'grey'

rowEvenColor = 'lightgrey'

rowOddColor = 'white'



fig = go.Figure(data=[go.Table(

  header=dict(

    values=['<b>Country</b>','<b>Deaths</b>'],

    line_color='darkslategray',

    fill_color=headerColor,

    align=['left','center'],

    font=dict(color='white', size=12)

  ),

  cells=dict(

    values=[

      Deaths_per_country['Country/Region'],

      Deaths_per_country['Deaths'],

      ],

    line_color='darkslategray',

    # 2-D list of colors for alternating rows

    fill_color = [[rowOddColor,rowEvenColor,rowOddColor, rowEvenColor,rowOddColor]*len(Data_per_country)],

    align = ['left', 'center'],

    font = dict(color = 'darkslategray', size = 11)

    ))

])

fig.update_layout(

    title='Deaths In Each Country',

)

fig.show()
fig = go.Figure(data=[go.Bar(

            x=Deaths_per_country['Country/Region'][0:10], y=Deaths_per_country['Deaths'][0:10],

            text=Deaths_per_country['Deaths'][0:10],

            textposition='auto',

            marker_color='black'



        )])

fig.update_layout(

    title='Most 10 infected Countries',

    xaxis_title="Countries",

    yaxis_title="Deaths",

        template='plotly_white'



)

fig.show()



fig = go.Figure(data=[go.Scatter(

    x=Deaths_per_country['Country/Region'][0:10],

    y=Deaths_per_country['Deaths'][0:10],

    mode='markers',

    marker=dict(

        color=[145, 140, 135, 130, 125, 120,115,110,105,100],

        size=Deaths_per_country['Deaths'][0:10]/1000,

        showscale=True

        )

)])

fig.update_layout(

    title='Most 10 infected Countries',

    xaxis_title="Countries",

    yaxis_title="Deaths",

        template='plotly_white'



)

fig.show()
Data_tunisia = data [(data['Country/Region'] == 'Tunisia') ].reset_index(drop=True)

Data_tunisia.head()


fig = go.Figure()

fig.add_trace(go.Scatter(x=Data_tunisia['ObservationDate'], y=Data_tunisia['Confirmed'],

                    mode='lines',

                    name='Confirmed cases'))



fig.add_trace(go.Scatter(x=Data_tunisia['ObservationDate'], y=Data_tunisia['Active_case'],

                    mode='lines',

                    name='Active cases',line=dict( dash='dot')))

fig.add_trace(go.Scatter(x=Data_tunisia['ObservationDate'], y=Data_tunisia['Deaths'],name='Deaths',

                                   marker_color='black',mode='lines',line=dict( dash='dot') ))

fig.add_trace(go.Scatter(x=Data_tunisia['ObservationDate'], y=Data_tunisia['Recovered'],

                    mode='lines',

                    name='Recovered cases',marker_color='green'))

fig.update_layout(

    title='Evolution of cases over time in Tunisia',

        template='plotly_white'



)



fig.show()



fig = go.Figure(go.Bar(

            x=Data_tunisia['ObservationDate'],

            y=Data_tunisia['Confirmed'],

    marker_color='rgb(13,48,100)'

           ))

fig.update_layout(

    title='Confirmed cases In Each Day',

    template='plotly_white',

     xaxis_title="Confirmed cases",

    yaxis_title="Days",

)

fig.show()
fig = go.Figure(go.Bar(

            x=Data_tunisia['ObservationDate'],

            y=Data_tunisia['Active_case'],

    marker_color='rgb(13,48,100)'

           ))

fig.update_layout(

    title='Active cases In Each Day',

    template='plotly_white',

     xaxis_title="Active cases",

    yaxis_title="Days",

)

fig.show()
fig = go.Figure(go.Bar(

            x=Data_tunisia['ObservationDate'],

            y=Data_tunisia['Recovered'],

    marker_color='rgb(13,48,100)'

           ))

fig.update_layout(

    title='Recovered cases In Each Day',

    template='plotly_white',

     xaxis_title="Recovered cases",

    yaxis_title="Days",

)

fig.show()
fig = go.Figure(go.Bar(

            x=Data_tunisia['ObservationDate'],

            y=Data_tunisia['Deaths'],

    marker_color='rgb(13,48,100)'

           ))

fig.update_layout(

    title='Deaths In Each Day',

    template='plotly_white',

     xaxis_title="Deaths",

    yaxis_title="Days",

)

fig.show()
Data_tunisia_last = Data_tunisia[Data_tunisia['ObservationDate'] == max(Data_tunisia['ObservationDate'])].reset_index()

Data_tunisia_last


colors = ['rgb(2,58,88)','rgb(65,171,93)', 'rgb(127,0,0)']

labels = ["Active cases","Recovered","Deaths"]

values = Data_tunisia_last.loc[0, ["Active_case","Recovered","Deaths"]]



fig = go.Figure(data=[go.Pie(labels=labels,

                             values=values)])

fig.update_traces(hoverinfo='label+percent', textinfo='value', textfont_size=20,

                  marker=dict(colors=colors, line=dict(color='#000000', width=2)))

fig.show()
Data_China = data [(data['Country/Region'] == 'China') ].reset_index(drop=True)

Data_China.head()
Data_china_last = Data_China[Data_China['ObservationDate'] == max(Data_China['ObservationDate'])].reset_index()

Data_china_last.head()
Data_china_per_state= Data_china_last.groupby(["Province/State"])["Confirmed","Active_case","Recovered","Deaths"].sum().reset_index().sort_values("Confirmed",ascending=False).reset_index(drop=True)

fig = px.pie(Data_china_per_state, values=Data_china_per_state['Confirmed'], names=Data_china_per_state['Province/State'],

             title='Confirmed cases in China',

            hole=.2)

fig.update_traces(textposition='inside', textinfo='percent+label')

fig.show()
fig = go.Figure(go.Bar(

            x=Data_china_per_state['Active_case'],

            y=Data_china_per_state['Province/State'],

            orientation='h',

            marker_color='#DC3912',))

fig.update_layout(

    title='Active Cases In Each Province/State',

    template='plotly_white',

    xaxis_title="Active Cases",

    yaxis_title="Province/State",

)

fig.show()
fig = go.Figure(go.Bar(

            x=Data_china_per_state['Recovered'],

            y=Data_china_per_state['Province/State'],

            orientation='h',

            marker_color='green',))

fig.update_layout(

    title='Active Cases In Each Province/State',

    template='plotly_white',

    xaxis_title="Recovered Cases",

    yaxis_title="Province/State",

)

fig.show()
fig = go.Figure(go.Bar(

            x=Data_china_per_state['Deaths'],

            y=Data_china_per_state['Province/State'],

            orientation='h',

            marker_color='black',))

fig.update_layout(

    title='Deaths In Each Province/State',

    template='plotly_white',

    xaxis_title="Deaths",

    yaxis_title="Province/State",

)

fig.show()
Data_china_total= Data_china_last.groupby(["Country/Region"])["Confirmed","Deaths","Recovered","Active_case"].sum().reset_index().reset_index(drop=True)

labels = ["Active cases","Recovered","Deaths"]

values = Data_china_total.loc[0, ["Active_case","Recovered","Deaths"]]

df = px.data.tips()

fig = px.pie(Data_china_total, values=values, names=labels, color_discrete_sequence=['green','royalblue','darkblue'], hole=0.5)

fig.update_layout(

    title='Total cases in China : '+str(Data_china_total["Confirmed"][0]),

)

fig.show()
Data_china_op= Data_China.groupby(["ObservationDate","Country/Region"])["Confirmed","Deaths","Recovered","Active_case"].sum().reset_index().reset_index(drop=True)

fig = go.Figure()

fig.add_trace(go.Scatter(x=Data_china_op['ObservationDate'], y=Data_china_op['Confirmed'],

                    mode='lines',

                    name='Confirmed cases'))





fig.add_trace(go.Scatter(x=Data_china_op['ObservationDate'], y=Data_china_op['Active_case'],

                    mode='lines',

                    name='Active cases',line=dict( dash='dot')))

fig.add_trace(go.Scatter(x=Data_china_op['ObservationDate'], y=Data_china_op['Deaths'],name='Deaths',

                                   marker_color='black',mode='lines',line=dict( dash='dot') ))

fig.add_trace(go.Scatter(x=Data_china_op['ObservationDate'], y=Data_china_op['Recovered'],

                    mode='lines',

                    name='Recovered cases',marker_color='green'))



fig.update_layout(

    title='Evolution of cases over time in China',

        template='plotly_white'



)



fig.show()

Data_Italy = data [(data['Country/Region'] == 'Italy') ].reset_index(drop=True)

Data_italy_last = Data_Italy[Data_Italy['ObservationDate'] == max(Data_Italy['ObservationDate'])].reset_index()

Data_italy= Data_italy_last.groupby(["Country/Region"])["Confirmed","Deaths","Recovered","Active_case"].sum().reset_index().reset_index(drop=True)

labels = ["Active cases","Recovered","Deaths"]

values = Data_italy.loc[0, ["Active_case","Recovered","Deaths"]]

df = px.data.tips()

fig = px.pie(Data_italy, values=values, names=labels, color_discrete_sequence=['royalblue','green','darkblue'], hole=0.5)

fig.update_layout(

    title='Total cases in Italy : '+str(Data_italy["Confirmed"][0]),

)

fig.show()
Data_italy_per_state= Data_italy_last.groupby(["Province/State"])["Confirmed","Deaths","Recovered","Active_case"].sum().reset_index().sort_values("Confirmed",ascending=False).reset_index(drop=True)

fig = px.pie(Data_italy_per_state, values=Data_italy_per_state['Confirmed'], names=Data_italy_per_state['Province/State'],

             title='Confirmed cases in Italy',

            hole=.5)

fig.update_traces(textposition='inside', textinfo='percent+label')

fig.show()





fig = px.pie(Data_italy_per_state, values=Data_italy_per_state['Recovered'], names=Data_italy_per_state['Province/State'],

             title='Recovered cases in Italy',

            )

fig.update_traces(textposition='inside', textinfo='percent+label')

fig.show()

fig = px.pie(Data_italy_per_state, values=Data_italy_per_state['Active_case'], names=Data_italy_per_state['Province/State'],

             

             title='Active cases in Italy',

            )

fig.update_traces(textposition='inside', textinfo='percent+label')

fig.show()

fig = px.pie(Data_italy_per_state, values=Data_italy_per_state['Deaths'], names=Data_italy_per_state['Province/State'],

             

             title='Deaths in Italy',

            )

fig.update_traces(textposition='inside', textinfo='percent+label')

fig.show()

Data_italy_op= Data_Italy.groupby(["ObservationDate","Country/Region"])["Confirmed","Deaths","Recovered","Active_case"].sum().reset_index().reset_index(drop=True)

fig = go.Figure()

fig.add_trace(go.Scatter(x=Data_italy_op.index, y=Data_italy_op['Confirmed'],

                    mode='lines',

                    name='Confirmed cases'))





fig.update_layout(

    title='Evolution of Confirmed cases over time in Italy',

        template='plotly_white'



)



fig.show()

fig = go.Figure()





fig.add_trace(go.Scatter(x=Data_italy_op.index, y=Data_italy_op['Active_case'],

                    mode='lines',marker_color='yellow',

                    name='Active cases',line=dict( dash='dot')))



fig.update_layout(

    title='Evolution of Acitive cases over time in Italy',

        template='plotly_dark'



)



fig.show()
fig = go.Figure()



fig.add_trace(go.Scatter(x=Data_italy_op.index, y=Data_italy_op['Recovered'],

                    mode='lines',

                    name='Recovered cases',marker_color='green'))



fig.update_layout(

    title='Evolution of Recovered cases over time in Italy',

        template='plotly_white'



)



fig.show()
fig = go.Figure()



fig.add_trace(go.Scatter(x=Data_italy_op.index, y=Data_italy_op['Deaths'],name='Deaths',

                                   marker_color='black',mode='lines',line=dict( dash='dot') ))



fig.update_layout(

    title='Evolution of Deaths over time in Italy',

        template='plotly_white'



)



fig.show()
Data_US = data [(data['Country/Region'] == 'US') ].reset_index(drop=True)

Data_us_last = Data_US[Data_US['ObservationDate'] == max(Data_US['ObservationDate'])].reset_index()
Data_us_total= Data_us_last.groupby(["Country/Region"])["Confirmed","Deaths","Recovered","Active_case"].sum().reset_index().reset_index(drop=True)

labels = ["Active cases","Recovered","Deaths"]

values = Data_us_total.loc[0, ["Active_case","Recovered","Deaths"]]

df = px.data.tips()

fig = px.pie(Data_us_total, values=values, names=labels, color_discrete_sequence=['royalblue','darkblue','green'], hole=0.5)

fig.update_layout(

    title='Total cases in United States : '+str(Data_us_total["Confirmed"][0]),

)

fig.show()
Data_us_per_state= Data_us_last.groupby(["Province/State"])["Confirmed","Active_case","Deaths"].sum().reset_index().sort_values("Confirmed",ascending=False).reset_index(drop=True)

fig = px.pie(Data_us_per_state, values=Data_us_per_state['Confirmed'], names=Data_us_per_state['Province/State'],

             title='Confirmed cases in United States',

            hole=.2)

fig.update_traces(textposition='inside', textinfo='percent+label')

fig.show()
fig = px.pie(Data_us_per_state, values=Data_us_per_state['Active_case'], names=Data_us_per_state['Province/State'],

             title='Active cases in United States',

            hole=.2)

fig.update_traces(textposition='inside', textinfo='percent+label')

fig.show()
fig = px.pie(Data_us_per_state, values=Data_us_per_state['Deaths'], names=Data_us_per_state['Province/State'],

             title='Deaths in United States',

            hole=.2)

fig.update_traces(textposition='inside', textinfo='percent+label')

fig.show()
Data_US_op= Data_US.groupby(["ObservationDate","Country/Region"])["Confirmed","Deaths","Recovered","Active_case"].sum().reset_index().reset_index(drop=True)



fig = go.Figure()

fig.add_trace(go.Scatter(x=Data_US_op.index, y=Data_US_op['Confirmed'],

                    mode='lines',

                    name='Confirmed cases'))





fig.update_layout(

    title='Evolution of Confirmed cases over time in US',

        template='plotly_white'



)



fig.show()

fig = go.Figure()



fig.add_trace(go.Scatter(x=Data_US_op.index, y=Data_US_op['Active_case'],

                    mode='lines',

                    name='Active cases',line=dict( dash='dot')))



fig.update_layout(

    title='Evolution of Acitive cases over time in US',

        template='plotly_white'



)



fig.show()
fig = go.Figure()



fig.add_trace(go.Scatter(x=Data_US_op.index, y=Data_US_op['Recovered'],

                    mode='lines',

                    name='Recovered cases',marker_color='green'))



fig.update_layout(

    title='Evolution of Recovered cases over time in US',

        template='plotly_white'



)



fig.show()

fig = go.Figure()



fig.add_trace(go.Scatter(x=Data_US_op.index, y=Data_US_op['Deaths'],name='Deaths',

                                   marker_color='white',mode='lines',line=dict( dash='dot') ))



fig.update_layout(

    title='Evolution of Deaths over time in US',

        template='plotly_dark'



)



fig.show()
Data_Spain = data [(data['Country/Region'] == 'Spain') ].reset_index(drop=True)

Data_spain = Data_Spain[Data_Spain['ObservationDate'] == max(Data_Spain['ObservationDate'])].reset_index()

Data_spain_last= Data_spain.groupby(["Country/Region"])["Confirmed","Deaths","Recovered","Active_case"].sum().reset_index().reset_index(drop=True)

labels = ["Active cases","Recovered","Deaths"]

values = Data_spain_last.loc[0, ["Active_case","Recovered","Deaths"]]

df = px.data.tips()

fig = px.pie(Data_spain_last, values=values, names=labels, color_discrete_sequence=['royalblue','green','darkblue'], hole=0.5)

fig.update_layout(

    title='Total cases in Spain : '+str(Data_spain_last["Confirmed"][0]),

)

fig.show()
Data_spain_per_state= Data_spain.groupby(["Province/State"])["Confirmed","Deaths","Recovered","Active_case"].sum().reset_index().sort_values("Confirmed",ascending=False).reset_index(drop=True)



fig = px.treemap(Data_spain_per_state, path=['Province/State'], values=Data_spain_per_state['Confirmed'], height=700,

                 title='Confirmed cases in Spain', color_discrete_sequence = px.colors.qualitative.Dark2)

fig.data[0].textinfo = 'label+text+value'

fig.show()





fig = px.treemap(Data_spain_per_state, path=['Province/State'], values=Data_spain_per_state['Recovered'], height=700,

                 title='Recovered cases in Spain', color_discrete_sequence = px.colors.qualitative.Dark2)

fig.data[0].textinfo = 'label+text+value'

fig.show()





fig = px.treemap(Data_spain_per_state, path=['Province/State'], values=Data_spain_per_state['Active_case'], height=700,

                 title='Active cases in Spain', color_discrete_sequence = px.colors.sequential.deep)

fig.data[0].textinfo = 'label+text+value'

fig.show()







fig = px.treemap(Data_spain_per_state, path=['Province/State'], values=Data_spain_per_state['Deaths'], height=700,

                 title='Deaths in Spain', color_discrete_sequence = px.colors.sequential.deep)

fig.data[0].textinfo = 'label+text+value'

fig.show()



Data_spain_op= Data_Spain.groupby(["ObservationDate","Country/Region"])["Confirmed","Deaths","Recovered","Active_case"].sum().reset_index().reset_index(drop=True)

fig = go.Figure(go.Bar(

            x=Data_spain_op['ObservationDate'],

            y=Data_spain_op['Confirmed'],

           ))

fig.update_layout(

    title='Confirmed Cases In Each Day',

    template='plotly_white',

     xaxis_title="Confirmed Cases",

    yaxis_title="Days",

)

fig.show()
fig = go.Figure(go.Bar(

            x=Data_spain_op['ObservationDate'],

            y=Data_spain_op['Active_case'],

    marker_color='rgb(253,187,132)'

           ))

fig.update_layout(

    title='Active Cases In Each Day',

    template='plotly_dark',

     xaxis_title="Active Cases",

    yaxis_title="Days",

)

fig.show()
fig = go.Figure(go.Bar(

            x=Data_spain_op['ObservationDate'],

            y=Data_spain_op['Recovered'],

    marker_color='rgb(178,24,43)'

           ))

fig.update_layout(

    title='Recovered Cases In Each Day',

    template='plotly_white',

     xaxis_title="Recovered Cases",

    yaxis_title="Days",

)

fig.show()
fig = go.Figure(go.Bar(

            x=Data_spain_op['ObservationDate'],

            y=Data_spain_op['Deaths'],

    marker_color='rgb(13,48,100)'

           ))

fig.update_layout(

    title='Deaths In Each Day',

    template='plotly_white',

     xaxis_title="Deaths",

    yaxis_title="Days",

)

fig.show()
Data_Brazil = data [(data['Country/Region'] == 'Brazil') ].reset_index(drop=True)

Data_brazil = Data_Brazil[Data_Brazil['ObservationDate'] == max(Data_Brazil['ObservationDate'])].reset_index()

Data_brazil_last= Data_brazil.groupby(["Country/Region"])["Confirmed","Deaths","Recovered","Active_case"].sum().reset_index().reset_index(drop=True)
labels = ["Active cases","Recovered","Deaths"]

values = Data_brazil_last.loc[0, ["Active_case","Recovered","Deaths"]]

fig = px.pie(Data_brazil_last, values=values, names=labels, color_discrete_sequence=px.colors.sequential.RdBu)

fig.update_layout(

    title='Total cases in Brazil : '+str(Data_brazil_last["Confirmed"][0])

    

)

fig.show()

Data_brazil_per_state= Data_Brazil.groupby(["Province/State"])["Confirmed","Deaths","Recovered","Active_case"].sum().reset_index().sort_values("Confirmed",ascending=False).reset_index(drop=True)

fig = px.treemap(Data_brazil_per_state, path=['Province/State'], values=Data_brazil_per_state['Confirmed'], height=700,

                 title='Confirmed cases in Brazil', color_discrete_sequence = px.colors.sequential.RdBu)

fig.data[0].textinfo = 'label+text+value'

fig.show()



fig = px.treemap(Data_brazil_per_state, path=['Province/State'], values=Data_brazil_per_state['Active_case'], height=700,

                 title='Active cases in Brazil', color_discrete_sequence = px.colors.sequential.Agsunset

)

fig.data[0].textinfo = 'label+text+value'

fig.show()
fig = px.treemap(Data_brazil_per_state, path=['Province/State'], values=Data_brazil_per_state['Recovered'], height=700,

                 title='Recovered cases in Brazil', color_discrete_sequence = px.colors.sequential.algae

)

fig.data[0].textinfo = 'label+text+value'

fig.show()
fig = px.treemap(Data_brazil_per_state, path=['Province/State'], values=Data_brazil_per_state['Deaths'], height=700,

                 title='Deaths in Brazil', color_discrete_sequence = px.colors.sequential.Cividis



)

fig.data[0].textinfo = 'label+text+value'

fig.show()
Data_Brazil_op= Data_Brazil.groupby(["ObservationDate","Country/Region"])["Confirmed","Deaths","Recovered","Active_case"].sum().reset_index().reset_index(drop=True)

fig = go.Figure(go.Bar(

            x=Data_Brazil_op['ObservationDate'],

            y=Data_Brazil_op['Confirmed'],

    marker_color='rgb(139,196,168)'

           ))

fig.update_layout(

    title='Confirmed cases In Each Day',

    template='plotly_white',

     xaxis_title="Confirmed cases",

    yaxis_title="Days",

)

fig.show()

fig = go.Figure(go.Bar(

            x=Data_Brazil_op['ObservationDate'],

            y=Data_Brazil_op['Deaths'],

    marker_color='rgb(139,196,168)'

           ))

fig.update_layout(

    title='Deaths In Each Day',

    template='plotly_white',

     xaxis_title="Deaths",

    yaxis_title="Days",

)

fig.show()


Data_Word = data [(data['Country/Region'] != 'US') ].reset_index(drop=True)

Data_WORD_last = Data_Word[Data_Word['ObservationDate'] == max(Data_Word['ObservationDate'])].reset_index()

Data_us_total= Data_us_last.groupby(["Country/Region"])["Confirmed","Deaths","Recovered","Active_case"].sum().reset_index().reset_index(drop=True)

Data_word_total= Data_WORD_last.groupby(["ObservationDate"])["Confirmed","Deaths","Recovered","Active_case"].sum().reset_index().reset_index(drop=True)



Data_Word_op= Data_Word.groupby(["ObservationDate"])["Confirmed","Deaths","Recovered","Active_case"].sum().reset_index().reset_index(drop=True)

labels = ["US","The rest of the world"]

values = [Data_us_total.loc[0, ["Confirmed"]], Data_word_total.loc[0, ["Confirmed"]]]

df = px.data.tips()

fig = px.pie(data, values=values, names=labels, color_discrete_sequence=['rgb(251,121,114)','rgb(128,177,211)'])

fig.update_layout(

    title='Confirmed cases in US and The Rest Of The World : ',

)

fig.show()
labels = ["US","The rest of the world"]

values = [Data_us_total.loc[0, ["Active_case"]], Data_word_total.loc[0, ["Active_case"]]]

df = px.data.tips()

fig = px.pie(data, values=values, names=labels, color_discrete_sequence=['rgb(204,235,197)','rgb(222,203,228)'],hole=0.6)

fig.update_layout(

    title='Active cases in US and The Rest Of The World : ',

)

fig.show()
labels = ["US","The rest of the world"]

values = [Data_us_total.loc[0, ["Recovered"]], Data_word_total.loc[0, ["Recovered"]]]

df = px.data.tips()

fig = px.pie(data, values=values, names=labels, color_discrete_sequence=['rgb(253,205,172)','rgb(179,226,205)'],hole=0.4)

fig.update_layout(

    title='Recovered cases in US and The Rest Of The World : ',

)

fig.show()
labels = ["US","The rest of the world"]

values = [Data_us_total.loc[0, ["Deaths"]], Data_word_total.loc[0, ["Deaths"]]]

df = px.data.tips()

fig = px.pie(data, values=values, names=labels, color_discrete_sequence=['rgb(229,196,148)','rgb(179,226,205)'])

fig.update_layout(

    title='Deaths in US and The Rest Of The World : ',

)

fig.show()
fig = go.Figure()

fig.add_trace(go.Scatter(x=Data_US_op.index, y=Data_US_op['Confirmed'],

                    mode='lines',

                    name='Confirmed cases in US'))





fig.add_trace(go.Scatter(x=Data_Word_op.index, y=Data_Word_op['Confirmed'],

                    mode='lines',

                    name='Confirmed cases in The Rest Of The Word'))



fig.update_layout(

    title='Evolution of Confirmed cases over time in US and The Rest Of The Word',

        template='plotly_white'



)



fig.show()

fig = go.Figure()

fig.add_trace(go.Scatter(x=Data_US_op.index, y=Data_US_op['Active_case'],

                    mode='lines',

                    name='Active cases in US'))





fig.add_trace(go.Scatter(x=Data_Word_op.index, y=Data_Word_op['Active_case'],

                    mode='lines',

                    name='Active cases in The Rest Of The Word'))



fig.update_layout(

    title='Evolution of Active cases over time in US and The Rest Of The Word',

        template='plotly_dark'



)



fig.show()

fig = go.Figure()

fig.add_trace(go.Scatter(x=Data_US_op.index, y=Data_US_op['Recovered'],

                    mode='lines',

                    name='Recovered cases in US'))





fig.add_trace(go.Scatter(x=Data_Word_op.index, y=Data_Word_op['Recovered'],

                    mode='lines',

                    name='Recovered cases in The Rest Of The Word'))



fig.update_layout(

    title='Evolution of Recovered cases over time in US and The Rest Of The Word',

        template='plotly_dark'



)



fig.show()

fig = go.Figure()

fig.add_trace(go.Scatter(x=Data_US_op.index, y=Data_US_op['Deaths'],

                    mode='lines',

                    name='Deaths in US'))





fig.add_trace(go.Scatter(x=Data_Word_op.index, y=Data_Word_op['Deaths'],

                    mode='lines',

                    name='Deathsin The Rest Of The Word'))

fig.update_layout(

    title='Evolution of Deaths over time in US and The Rest Of The Word',

        template='plotly_white'



)



fig.show()

Data_Ger = data [(data['Country/Region'] == 'Germany') ].reset_index(drop=True)

Data_Iran = data [(data['Country/Region'] == 'Iran') ].reset_index(drop=True)

def getDate(date,weeks):



    datein = datetime.datetime.strptime(date, "%m/%d/%Y")

    threeWeeks = datetime.timedelta(weeks =weeks)

    datefinal = (datein + threeWeeks)

    

    return(datefinal.strftime('%m/%d/%Y'))
Data_tunisia_after_twoweeks = Data_tunisia[ Data_tunisia['ObservationDate'] < getDate(min(Data_tunisia['ObservationDate']),2) ].reset_index()

Data_italy_after_twoweeks = Data_Italy[Data_Italy['ObservationDate'] < getDate(min(Data_Italy['ObservationDate']),2) ].reset_index()

Data_spain_after_twoweeks = Data_Spain[Data_Spain['ObservationDate'] < getDate(min(Data_Spain['ObservationDate']),2) ].reset_index()

Data_germany_after_twoweeks = Data_Ger[Data_Ger['ObservationDate'] < getDate(min(Data_Ger['ObservationDate']),2) ].reset_index()







fig = make_subplots(rows=2, cols=2,

                    specs=[[{"secondary_y": True}, {"secondary_y": True}],

                           [{"secondary_y": True}, {"secondary_y": True}]],

                   subplot_titles=("Tunisia : "+str(Data_tunisia_after_twoweeks['Confirmed'].max()),

                                   " Italy : "+ str(Data_italy_after_twoweeks['Confirmed'].max()),

                                   "  Spain : "+ str(Data_spain_after_twoweeks['Confirmed'].max()),

                                   " Germany : "+ str(Data_germany_after_twoweeks['Confirmed'].max())

                                  ))



# Top left

fig.add_trace(

    go.Scatter(x=list(range(1,len(Data_tunisia_after_twoweeks)+1)), y=Data_tunisia_after_twoweeks['Confirmed'], 

                name="Confirmed Cases In Tunisia"),row=1, col=1, secondary_y=False)





# Top right

fig.add_trace(

    go.Scatter(x=list(range(1,len(Data_italy_after_twoweeks)+1)), y=Data_italy_after_twoweeks['Confirmed'], 

               name="Confirmed Cases In Italy"), row=1, col=2, secondary_y=False,

)



# Bottom left

fig.add_trace(

    go.Scatter(x=list(range(1,len(Data_spain_after_twoweeks)+1)), y=Data_spain_after_twoweeks['Confirmed'], 

               name="Confirmed Cases In Spain"),row=2, col=1, secondary_y=False,

)





# Bottom right

fig.add_trace(

    go.Scatter(x=list(range(1,len(Data_germany_after_twoweeks)+1)), y=Data_germany_after_twoweeks['Confirmed'], 

               name="Confirmed Cases In Germany"),row=2, col=2, secondary_y=False,

)

# Update xaxis properties

fig.update_xaxes(title_text="Days")

fig.update_xaxes(title_text="Days")

fig.update_xaxes(title_text="Days")

fig.update_xaxes(title_text="Days")



# Update yaxis properties

fig.update_yaxes(title_text="Confirmed Cases")

fig.update_yaxes(title_text="Confirmed Cases")

fig.update_yaxes(title_text="Confirmed Cases")

fig.update_yaxes(title_text="Confirmed Cases")



fig.update_layout(

    title_text="Corona virus confirmed cases per country for the first two weeks",

    width=800,

)



fig.show()


fig = make_subplots(rows=2, cols=2,

                    specs=[[{"secondary_y": True}, {"secondary_y": True}],

                           [{"secondary_y": True}, {"secondary_y": True}]],

                   subplot_titles=("Tunisia : "+str(Data_tunisia_after_twoweeks['Recovered'].max()),

                                   " Italy : "+ str(Data_italy_after_twoweeks['Recovered'].max()),

                                   "  Spain : "+ str(Data_spain_after_twoweeks['Recovered'].max()),

                                   " Germany : "+ str(Data_germany_after_twoweeks['Recovered'].max())

                                  ))



# Top left

fig.add_trace(

    go.Scatter(x=list(range(1,len(Data_tunisia_after_twoweeks)+1)), y=Data_tunisia_after_twoweeks['Recovered'], 

                name="Recovered Cases In Tunisia"),row=1, col=1, secondary_y=False)





# Top right

fig.add_trace(

    go.Scatter(x=list(range(1,len(Data_italy_after_twoweeks)+1)), y=Data_italy_after_twoweeks['Recovered'], 

               name="Recovered Cases In Italy"), row=1, col=2, secondary_y=False,

)



# Bottom left

fig.add_trace(

    go.Scatter(x=list(range(1,len(Data_spain_after_twoweeks)+1)), y=Data_spain_after_twoweeks['Recovered'], 

               name="Recovered Cases In Spain"),row=2, col=1, secondary_y=False,

)





# Bottom right

fig.add_trace(

    go.Scatter(x=list(range(1,len(Data_germany_after_twoweeks)+1)), y=Data_germany_after_twoweeks['Recovered'], 

               name="Recovered Cases In Germany"),row=2, col=2, secondary_y=False,

)

# Update xaxis properties

fig.update_xaxes(title_text="Days")

fig.update_xaxes(title_text="Days")

fig.update_xaxes(title_text="Days")

fig.update_xaxes(title_text="Days")



# Update yaxis properties

fig.update_yaxes(title_text="Recovered Cases")

fig.update_yaxes(title_text="Recovered Cases")

fig.update_yaxes(title_text="Recovered Cases")

fig.update_yaxes(title_text="Recovered Cases")



fig.update_layout(

    title_text="Corona virus Recovered cases per country for the first two weeks",

    width=800,

)



fig.show()


fig = make_subplots(rows=2, cols=2,

                    specs=[[{"secondary_y": True}, {"secondary_y": True}],

                           [{"secondary_y": True}, {"secondary_y": True}]],

                   subplot_titles=("Tunisia : "+str(Data_tunisia_after_twoweeks['Deaths'].max()),

                                   " Italy : "+ str(Data_italy_after_twoweeks['Deaths'].max()),

                                   "  Spain : "+ str(Data_spain_after_twoweeks['Deaths'].max()),

                                   " Germany : "+ str(Data_germany_after_twoweeks['Deaths'].max())

                                  ))



# Top left

fig.add_trace(

    go.Scatter(x=list(range(1,len(Data_tunisia_after_twoweeks)+1)), y=Data_tunisia_after_twoweeks['Deaths'], 

                name="Deaths In Tunisia"),row=1, col=1, secondary_y=False)





# Top right

fig.add_trace(

    go.Scatter(x=list(range(1,len(Data_italy_after_twoweeks)+1)), y=Data_italy_after_twoweeks['Deaths'], 

               name="Deaths In Italy"), row=1, col=2, secondary_y=False,

)



# Bottom left

fig.add_trace(

    go.Scatter(x=list(range(1,len(Data_spain_after_twoweeks)+1)), y=Data_spain_after_twoweeks['Deaths'], 

               name="Deaths In Spain"),row=2, col=1, secondary_y=False,

)





# Bottom right

fig.add_trace(

    go.Scatter(x=list(range(1,len(Data_germany_after_twoweeks)+1)), y=Data_germany_after_twoweeks['Deaths'], 

               name="Deaths In Germany"),row=2, col=2, secondary_y=False,

)

# Update xaxis properties

fig.update_xaxes(title_text="Days")

fig.update_xaxes(title_text="Days")

fig.update_xaxes(title_text="Days")

fig.update_xaxes(title_text="Days")



# Update yaxis properties

fig.update_yaxes(title_text="Deaths")

fig.update_yaxes(title_text="Deaths")

fig.update_yaxes(title_text="Deaths")

fig.update_yaxes(title_text="Deaths")



fig.update_layout(

    title_text="Corona virus Deaths per country for the first two weeks",

    width=800,

)



fig.show()
Data_iran_after_fourweeks = Data_Iran[ Data_Iran['ObservationDate'] < getDate(min(Data_Iran['ObservationDate']),4) ].reset_index()

Data_italy_after_fourweeks = Data_Italy[Data_Italy['ObservationDate'] < getDate(min(Data_Italy['ObservationDate']),4) ].reset_index()

Data_spain_after_fourweeks = Data_Spain[Data_Spain['ObservationDate'] < getDate(min(Data_Spain['ObservationDate']),4) ].reset_index()

Data_germany_after_fourweeks = Data_Ger[Data_Ger['ObservationDate'] < getDate(min(Data_Ger['ObservationDate']),4) ].reset_index()





fig = make_subplots(rows=2, cols=2,

                    specs=[[{"secondary_y": True}, {"secondary_y": True}],

                           [{"secondary_y": True}, {"secondary_y": True}]],

                   subplot_titles=("Iran : "+str(Data_iran_after_fourweeks['Confirmed'].max()),

                                   " Italy : "+ str(Data_italy_after_fourweeks['Confirmed'].max()),

                                   "  Spain : "+ str(Data_spain_after_fourweeks['Confirmed'].max()),

                                   " Germany : "+ str(Data_germany_after_fourweeks['Confirmed'].max())

                                  ))



# Top left

fig.add_trace(

    go.Scatter(x=list(range(1,len(Data_iran_after_fourweeks)+1)), y=Data_iran_after_fourweeks['Confirmed'], 

                name="Confirmed Cases In Iran"),row=1, col=1, secondary_y=False)





# Top right

fig.add_trace(

    go.Scatter(x=list(range(1,len(Data_italy_after_fourweeks)+1)), y=Data_italy_after_fourweeks['Confirmed'], 

               name="Confirmed Cases In Italy"), row=1, col=2, secondary_y=False,

)



# Bottom left

fig.add_trace(

    go.Scatter(x=list(range(1,len(Data_spain_after_fourweeks)+1)), y=Data_spain_after_fourweeks['Confirmed'], 

               name="Confirmed Cases In Spain"),row=2, col=1, secondary_y=False,

)





# Bottom right

fig.add_trace(

    go.Scatter(x=list(range(1,len(Data_germany_after_fourweeks)+1)), y=Data_germany_after_fourweeks['Confirmed'], 

               name="Confirmed Cases In Germany"),row=2, col=2, secondary_y=False,

)

# Update xaxis properties

fig.update_xaxes(title_text="Days",range=[0,30])

fig.update_xaxes(title_text="Days")

fig.update_xaxes(title_text="Days")

fig.update_xaxes(title_text="Days")



# Update yaxis properties

fig.update_yaxes(title_text="Confirmed Cases")

fig.update_yaxes(title_text="Confirmed Cases")

fig.update_yaxes(title_text="Confirmed Cases")

fig.update_yaxes(title_text="Confirmed Cases")



fig.update_layout(

    title_text="Corona virus confirmed cases per country for the first month",

    width=800,

)



fig.show()


fig = make_subplots(rows=2, cols=2,

                    specs=[[{"secondary_y": True}, {"secondary_y": True}],

                           [{"secondary_y": True}, {"secondary_y": True}]],

                   subplot_titles=("Iran : "+str(Data_iran_after_fourweeks['Recovered'].max()),

                                   " Italy : "+ str(Data_italy_after_fourweeks['Recovered'].max()),

                                   "  Spain : "+ str(Data_spain_after_fourweeks['Recovered'].max()),

                                   " Germany : "+ str(Data_germany_after_fourweeks['Recovered'].max())

                                  ))



# Top left

fig.add_trace(

    go.Scatter(x=list(range(1,len(Data_iran_after_fourweeks)+1)), y=Data_iran_after_fourweeks['Recovered'], 

                name="Recovered Cases In Iran"),row=1, col=1, secondary_y=False)





# Top right

fig.add_trace(

    go.Scatter(x=list(range(1,len(Data_italy_after_fourweeks)+1)), y=Data_italy_after_fourweeks['Recovered'], 

               name="Recovered Cases In Italy"), row=1, col=2, secondary_y=False,

)



# Bottom left

fig.add_trace(

    go.Scatter(x=list(range(1,len(Data_spain_after_fourweeks)+1)), y=Data_spain_after_fourweeks['Recovered'], 

               name="Recovered Cases In Spain"),row=2, col=1, secondary_y=False,

)





# Bottom right

fig.add_trace(

    go.Scatter(x=list(range(1,len(Data_germany_after_fourweeks)+1)), y=Data_germany_after_fourweeks['Recovered'], 

               name="Recovered Cases In Germany"),row=2, col=2, secondary_y=False,

)

# Update xaxis properties

fig.update_xaxes(title_text="Days",range=[0,30])

fig.update_xaxes(title_text="Days")

fig.update_xaxes(title_text="Days")

fig.update_xaxes(title_text="Days")



# Update yaxis properties

fig.update_yaxes(title_text="Recovered Cases")

fig.update_yaxes(title_text="Recovered Cases")

fig.update_yaxes(title_text="Recovered Cases")

fig.update_yaxes(title_text="Recovered Cases")



fig.update_layout(

    title_text="Corona virus Recovered cases per country for the first month",

    width=800,

)



fig.show()


fig = make_subplots(rows=2, cols=2,

                    specs=[[{"secondary_y": True}, {"secondary_y": True}],

                           [{"secondary_y": True}, {"secondary_y": True}]],

                   subplot_titles=("Iran : "+str(Data_iran_after_fourweeks['Deaths'].max()),

                                   " Italy : "+ str(Data_italy_after_fourweeks['Deaths'].max()),

                                   "  Spain : "+ str(Data_spain_after_fourweeks['Deaths'].max()),

                                   " Germany : "+ str(Data_germany_after_fourweeks['Deaths'].max())

                                  ))



# Top left

fig.add_trace(

    go.Scatter(x=list(range(1,len(Data_iran_after_fourweeks)+1)), y=Data_iran_after_fourweeks['Deaths'], 

                name="Deaths In Iran"),row=1, col=1, secondary_y=False)





# Top right

fig.add_trace(

    go.Scatter(x=list(range(1,len(Data_italy_after_fourweeks)+1)), y=Data_italy_after_fourweeks['Deaths'], 

               name="Deaths In Italy"), row=1, col=2, secondary_y=False,

)



# Bottom left

fig.add_trace(

    go.Scatter(x=list(range(1,len(Data_spain_after_fourweeks)+1)), y=Data_spain_after_fourweeks['Deaths'], 

               name="Deaths In Spain"),row=2, col=1, secondary_y=False,

)





# Bottom right

fig.add_trace(

    go.Scatter(x=list(range(1,len(Data_germany_after_fourweeks)+1)), y=Data_germany_after_fourweeks['Deaths'], 

               name="Deaths In Germany"),row=2, col=2, secondary_y=False,

)

# Update xaxis properties

fig.update_xaxes(title_text="Days",range=[0,30])

fig.update_xaxes(title_text="Days")

fig.update_xaxes(title_text="Days")

fig.update_xaxes(title_text="Days")



# Update yaxis properties

fig.update_yaxes(title_text="Deaths")

fig.update_yaxes(title_text="Deaths")

fig.update_yaxes(title_text="Deaths")

fig.update_yaxes(title_text="Deaths")



fig.update_layout(

    title_text="Corona virus Deaths per country for the first month",

    width=800,

)



fig.show()
Data_iran_after_fiveweeks = Data_Iran[ Data_Iran['ObservationDate'] < getDate(min(Data_Iran['ObservationDate']),5) ].reset_index()

Data_italy_after_fiveweeks  = Data_Italy[Data_Italy['ObservationDate'] < getDate(min(Data_Italy['ObservationDate']),5) ].reset_index()

Data_spain_after_fiveweeks  = Data_Spain[Data_Spain['ObservationDate'] < getDate(min(Data_Spain['ObservationDate']),5) ].reset_index()

Data_germany_after_fiveweeks  = Data_Ger[Data_Ger['ObservationDate'] < getDate(min(Data_Ger['ObservationDate']),5) ].reset_index()



fig = make_subplots(rows=2, cols=2,

                    specs=[[{"secondary_y": True}, {"secondary_y": True}],

                           [{"secondary_y": True}, {"secondary_y": True}]],

                   subplot_titles=("Iran : "+str(Data_iran_after_fiveweeks['Confirmed'].max()),

                                   " Italy : "+ str(Data_italy_after_fiveweeks['Confirmed'].max()),

                                   "  Spain : "+ str(Data_spain_after_fiveweeks['Confirmed'].max()),

                                   " Germany : "+ str(Data_germany_after_fiveweeks['Confirmed'].max())

                                  ))



# Top left

fig.add_trace(

    go.Scatter(x=list(range(1,len(Data_iran_after_fiveweeks)+1)), y=Data_iran_after_fiveweeks['Confirmed'], 

                name="Confirmed Cases In Iran"),row=1, col=1, secondary_y=False)





# Top right

fig.add_trace(

    go.Scatter(x=list(range(1,len(Data_italy_after_fiveweeks)+1)), y=Data_italy_after_fiveweeks['Confirmed'], 

               name="Confirmed Cases In Italy"), row=1, col=2, secondary_y=False,

)



# Bottom left

fig.add_trace(

    go.Scatter(x=list(range(1,len(Data_spain_after_fiveweeks)+1)), y=Data_spain_after_fiveweeks['Confirmed'], 

               name="Confirmed Cases In Spain"),row=2, col=1, secondary_y=False,

)





# Bottom right

fig.add_trace(

    go.Scatter(x=list(range(1,len(Data_germany_after_fiveweeks)+1)), y=Data_germany_after_fiveweeks['Confirmed'], 

               name="Confirmed Cases In Germany"),row=2, col=2, secondary_y=False,

)

# Update xaxis properties

fig.update_xaxes(title_text="Days",range=[0,35])

fig.update_xaxes(title_text="Days")

fig.update_xaxes(title_text="Days")

fig.update_xaxes(title_text="Days")



# Update yaxis properties

fig.update_yaxes(title_text="Confirmed Cases")

fig.update_yaxes(title_text="Confirmed Cases")

fig.update_yaxes(title_text="Confirmed Cases")

fig.update_yaxes(title_text="Confirmed Cases")



fig.update_layout(

    title_text="Corona virus confirmed cases per country for the fist five weeks",

    width=800,

)



fig.show()


fig = make_subplots(rows=2, cols=2,

                    specs=[[{"secondary_y": True}, {"secondary_y": True}],

                           [{"secondary_y": True}, {"secondary_y": True}]],

                   subplot_titles=("Iran : "+str(Data_iran_after_fiveweeks['Recovered'].max()),

                                   " Italy : "+ str(Data_italy_after_fiveweeks['Recovered'].max()),

                                   "  Spain : "+ str(Data_spain_after_fiveweeks['Recovered'].max()),

                                   " Germany : "+ str(Data_germany_after_fiveweeks['Recovered'].max())

                                  ))



# Top left

fig.add_trace(

    go.Scatter(x=list(range(1,len(Data_iran_after_fiveweeks)+1)), y=Data_iran_after_fiveweeks['Recovered'], 

                name="Recovered Cases In Iran"),row=1, col=1, secondary_y=False)





# Top right

fig.add_trace(

    go.Scatter(x=list(range(1,len(Data_italy_after_fiveweeks)+1)), y=Data_italy_after_fiveweeks['Recovered'], 

               name="Recovered Cases In Italy"), row=1, col=2, secondary_y=False,

)



# Bottom left

fig.add_trace(

    go.Scatter(x=list(range(1,len(Data_spain_after_fiveweeks)+1)), y=Data_spain_after_fiveweeks['Recovered'], 

               name="Recovered Cases In Spain"),row=2, col=1, secondary_y=False,

)





# Bottom right

fig.add_trace(

    go.Scatter(x=list(range(1,len(Data_germany_after_fiveweeks)+1)), y=Data_germany_after_fiveweeks['Recovered'], 

               name="Recovered Cases In Germany"),row=2, col=2, secondary_y=False,

)

# Update xaxis properties

fig.update_xaxes(title_text="Days",range=[0,35])

fig.update_xaxes(title_text="Days")

fig.update_xaxes(title_text="Days")

fig.update_xaxes(title_text="Days")



# Update yaxis properties

fig.update_yaxes(title_text="Recovered Cases")

fig.update_yaxes(title_text="Recovered Cases")

fig.update_yaxes(title_text="Recovered Cases")

fig.update_yaxes(title_text="Recovered Cases")



fig.update_layout(

    title_text="Corona virus Recovered cases per country for the fist five weeks",

    width=800,

)



fig.show()


fig = make_subplots(rows=2, cols=2,

                    specs=[[{"secondary_y": True}, {"secondary_y": True}],

                           [{"secondary_y": True}, {"secondary_y": True}]],

                   subplot_titles=("Iran : "+str(Data_iran_after_fiveweeks['Deaths'].max()),

                                   " Italy : "+ str(Data_italy_after_fiveweeks['Deaths'].max()),

                                   "  Spain : "+ str(Data_spain_after_fiveweeks['Deaths'].max()),

                                   " Germany : "+ str(Data_germany_after_fiveweeks['Deaths'].max())

                                  ))



# Top left

fig.add_trace(

    go.Scatter(x=list(range(1,len(Data_iran_after_fiveweeks)+1)), y=Data_iran_after_fiveweeks['Deaths'], 

                name="Deaths In Iran"),row=1, col=1, secondary_y=False)





# Top right

fig.add_trace(

    go.Scatter(x=list(range(1,len(Data_italy_after_fiveweeks)+1)), y=Data_italy_after_fiveweeks['Deaths'], 

               name="Deaths In Italy"), row=1, col=2, secondary_y=False,

)



# Bottom left

fig.add_trace(

    go.Scatter(x=list(range(1,len(Data_spain_after_fiveweeks)+1)), y=Data_spain_after_fiveweeks['Deaths'], 

               name="Deaths In Spain"),row=2, col=1, secondary_y=False,

)





# Bottom right

fig.add_trace(

    go.Scatter(x=list(range(1,len(Data_germany_after_fiveweeks)+1)), y=Data_germany_after_fiveweeks['Deaths'], 

               name="Deaths In Germany"),row=2, col=2, secondary_y=False,

)

# Update xaxis properties

fig.update_xaxes(title_text="Days",range=[0,35])

fig.update_xaxes(title_text="Days")

fig.update_xaxes(title_text="Days")

fig.update_xaxes(title_text="Days")



# Update yaxis properties

fig.update_yaxes(title_text="Deaths")

fig.update_yaxes(title_text="Deaths")

fig.update_yaxes(title_text="Deaths")

fig.update_yaxes(title_text="Deaths")



fig.update_layout(

    title_text="Corona virus Deaths per country for the fist five weeks",

    width=800,

)



fig.show()
Data_italy_after_sixweeks = Data_Italy[Data_Italy['ObservationDate'] < getDate(min(Data_Italy['ObservationDate']),6) ].reset_index()

Data_spain_after_sixweeks = Data_Spain[Data_Spain['ObservationDate'] < getDate(min(Data_Spain['ObservationDate']),6) ].reset_index()

Data_germany_after_sixweeks = Data_Ger[Data_Ger['ObservationDate'] < getDate(min(Data_Ger['ObservationDate']),6) ].reset_index()



fig = make_subplots(rows=1, cols=2,

                    specs=[[{"secondary_y": True}, {"secondary_y": True}]],

                   subplot_titles=(

                                   " Italy : "+ str(Data_italy_after_sixweeks['Confirmed'].max()),

                                   "  Spain : "+ str(Data_spain_after_sixweeks['Confirmed'].max())

                                  ))





fig.add_trace(

    go.Scatter(x=list(range(1,len(Data_italy_after_sixweeks)+1)), y=Data_italy_after_sixweeks['Confirmed'], 

               name="Confirmed Cases In Italy"), row=1, col=1, secondary_y=False,

)



fig.add_trace(

    go.Scatter(x=list(range(1,len(Data_spain_after_sixweeks)+1)), y=Data_spain_after_sixweeks['Confirmed'], 

               name="Confirmed Cases In Spain"),row=1, col=2, secondary_y=False,

)







# Update xaxis properties

fig.update_xaxes(title_text="Days",range=[0,45])

fig.update_xaxes(title_text="Days")



# Update yaxis properties

fig.update_yaxes(title_text="Confirmed Cases")

fig.update_yaxes(title_text="Confirmed Cases")



fig.update_layout(

    title_text="Corona virus confirmed cases per country for the first six weeks",

    width=800,

)



fig.show()


fig = make_subplots(rows=1, cols=2,

                    specs=[[{"secondary_y": True}, {"secondary_y": True}]],

                   subplot_titles=(

                                   " Italy : "+ str(Data_italy_after_sixweeks['Confirmed'].max()),

                                   "  Germany : "+ str(Data_germany_after_sixweeks['Confirmed'].max())

                                  ))





fig.add_trace(

    go.Scatter(x=list(range(1,len(Data_italy_after_sixweeks)+1)), y=Data_italy_after_sixweeks['Confirmed'], 

               name="Confirmed Cases In Italy"), row=1, col=1, secondary_y=False,

)



fig.add_trace(

    go.Scatter(x=list(range(1,len(Data_germany_after_sixweeks)+1)), y=Data_germany_after_sixweeks['Confirmed'], 

               name="Confirmed Cases In Germany"),row=1, col=2, secondary_y=False,

)







# Update xaxis properties

fig.update_xaxes(title_text="Days",range=[0,45])

fig.update_xaxes(title_text="Days")



# Update yaxis properties

fig.update_yaxes(title_text="Confirmed Cases")

fig.update_yaxes(title_text="Confirmed Cases")



fig.update_layout(

    title_text="Corona virus confirmed cases per country for the first six weeks",

    width=800,

)



fig.show()


fig = make_subplots(rows=1, cols=2,

                    specs=[[{"secondary_y": True}, {"secondary_y": True}]],

                   subplot_titles=(

                                   " Italy : "+ str(Data_italy_after_sixweeks['Recovered'].max()),

                                   "  Spain : "+ str(Data_spain_after_sixweeks['Recovered'].max())

                                  ))





fig.add_trace(

    go.Scatter(x=list(range(1,len(Data_italy_after_sixweeks)+1)), y=Data_italy_after_sixweeks['Recovered'], 

               name="Recovered Cases In Italy"), row=1, col=1, secondary_y=False,

)



fig.add_trace(

    go.Scatter(x=list(range(1,len(Data_spain_after_sixweeks)+1)), y=Data_spain_after_sixweeks['Recovered'], 

               name="Recovered Cases In Spain"),row=1, col=2, secondary_y=False,

)







# Update xaxis properties

fig.update_xaxes(title_text="Days",range=[0,45])

fig.update_xaxes(title_text="Days")



# Update yaxis properties

fig.update_yaxes(title_text="Recovered Cases")

fig.update_yaxes(title_text="Recovered Cases")



fig.update_layout(

    title_text="Corona virus Recovered cases per country for the first six weeks",

    width=800,

)



fig.show()


fig = make_subplots(rows=1, cols=2,

                    specs=[[{"secondary_y": True}, {"secondary_y": True}]],

                   subplot_titles=(

                                   " Italy : "+ str(Data_italy_after_sixweeks['Recovered'].max()),

                                   "  Germany : "+ str(Data_germany_after_sixweeks['Recovered'].max())

                                  ))





fig.add_trace(

    go.Scatter(x=list(range(1,len(Data_italy_after_sixweeks)+1)), y=Data_italy_after_sixweeks['Recovered'], 

               name="Recovered Cases In Italy"), row=1, col=1, secondary_y=False,

)



fig.add_trace(

    go.Scatter(x=list(range(1,len(Data_germany_after_sixweeks)+1)), y=Data_germany_after_sixweeks['Recovered'], 

               name="Recovered Cases In Germany"),row=1, col=2, secondary_y=False,

)







# Update xaxis properties

fig.update_xaxes(title_text="Days",range=[0,45])

fig.update_xaxes(title_text="Days")



# Update yaxis properties

fig.update_yaxes(title_text="Recovered Cases")

fig.update_yaxes(title_text="Recovered Cases")



fig.update_layout(

    title_text="Corona virus Recovered cases per country for the first six weeks",

    width=800,

)



fig.show()


fig = make_subplots(rows=1, cols=2,

                    specs=[[{"secondary_y": True}, {"secondary_y": True}]],

                   subplot_titles=(

                                   " Italy : "+ str(Data_italy_after_sixweeks['Deaths'].max()),

                                   "  Spain : "+ str(Data_spain_after_sixweeks['Deaths'].max())

                                  ))





fig.add_trace(

    go.Scatter(x=list(range(1,len(Data_italy_after_sixweeks)+1)), y=Data_italy_after_sixweeks['Deaths'], 

               name="Deaths In Italy"), row=1, col=1, secondary_y=False,

)



fig.add_trace(

    go.Scatter(x=list(range(1,len(Data_spain_after_sixweeks)+1)), y=Data_spain_after_sixweeks['Deaths'], 

               name="Deaths In Spain"),row=1, col=2, secondary_y=False,

)







# Update xaxis properties

fig.update_xaxes(title_text="Days",range=[0,45])

fig.update_xaxes(title_text="Days")



# Update yaxis properties

fig.update_yaxes(title_text="Deaths")

fig.update_yaxes(title_text="Deaths")



fig.update_layout(

    title_text="Corona virus Deaths per country for the first six weeks",

    width=800,

)



fig.show()


fig = make_subplots(rows=1, cols=2,

                    specs=[[{"secondary_y": True}, {"secondary_y": True}]],

                   subplot_titles=(

                                   " Italy : "+ str(Data_italy_after_sixweeks['Deaths'].max()),

                                   "  Germany : "+ str(Data_germany_after_sixweeks['Deaths'].max())

                                  ))





fig.add_trace(

    go.Scatter(x=list(range(1,len(Data_italy_after_sixweeks)+1)), y=Data_italy_after_sixweeks['Deaths'], 

               name="Deaths In Italy"), row=1, col=1, secondary_y=False,

)



fig.add_trace(

    go.Scatter(x=list(range(1,len(Data_germany_after_sixweeks)+1)), y=Data_germany_after_sixweeks['Deaths'], 

               name="Deaths In Germany"),row=1, col=2, secondary_y=False,

)







# Update xaxis properties

fig.update_xaxes(title_text="Days",range=[0,45])

fig.update_xaxes(title_text="Days")



# Update yaxis properties

fig.update_yaxes(title_text="Deaths")

fig.update_yaxes(title_text="Deaths")



fig.update_layout(

    title_text="Corona virus Deaths per country for the first six weeks",

    width=800,

)



fig.show()
Data_italy_after_2months = Data_Italy[Data_Italy['ObservationDate'] < getDate(min(Data_Italy['ObservationDate']),8) ].reset_index()

Data_spain_after_2months = Data_Spain[Data_Spain['ObservationDate'] < getDate(min(Data_Spain['ObservationDate']),8) ].reset_index()

Data_germany_after_2months = Data_Ger[Data_Ger['ObservationDate'] < getDate(min(Data_Ger['ObservationDate']),8) ].reset_index()
fig = make_subplots(rows=1, cols=3,

                    specs=[[{"secondary_y": True}, {"secondary_y": True}, {"secondary_y": True}]],

                   subplot_titles=(

                                   " Italy : "+ str(Data_italy_after_2months['Confirmed'].max()),

                               " Spain : "+ str(Data_spain_after_2months['Confirmed'].max()),

                                   "  Germany : "+ str(Data_germany_after_2months['Confirmed'].max())

                                  ))





fig.add_trace(

    go.Scatter(x=list(range(1,len(Data_italy_after_2months)+1)), y=Data_italy_after_2months['Confirmed'], 

               name="Confirmed Cases In Italy"), row=1, col=1, secondary_y=False,

)

fig.add_trace(

    go.Scatter(x=list(range(1,len(Data_spain_after_2months)+1)), y=Data_spain_after_2months['Confirmed'], 

               name="Confirmed Cases In Spain"),row=1, col=2, secondary_y=False,

)

fig.add_trace(

    go.Scatter(x=list(range(1,len(Data_germany_after_2months)+1)), y=Data_germany_after_2months['Confirmed'], 

               name="Confirmed Cases In Germany"),row=1, col=3, secondary_y=False,

)







# Update xaxis properties

fig.update_xaxes(title_text="Days",range=[0,60])

fig.update_xaxes(title_text="Days")

fig.update_xaxes(title_text="Days")



# Update yaxis properties

fig.update_yaxes(title_text="Confirmed Cases")

fig.update_yaxes(title_text="Confirmed Cases")

fig.update_yaxes(title_text="Confirmed Cases")



fig.update_layout(

    title_text="Corona virus confirmed cases per country for the first 2 months",

    width=800,

)



fig.show()
fig = make_subplots(rows=1, cols=3,

                    specs=[[{"secondary_y": True}, {"secondary_y": True}, {"secondary_y": True}]],

                   subplot_titles=(

                                   " Italy : "+ str(Data_italy_after_2months['Recovered'].max()),

                               " Spain : "+ str(Data_spain_after_2months['Recovered'].max()),

                                   "  Germany : "+ str(Data_germany_after_2months['Recovered'].max())

                                  ))





fig.add_trace(

    go.Scatter(x=list(range(1,len(Data_italy_after_2months)+1)), y=Data_italy_after_2months['Recovered'], 

               name="Recovered Cases In Italy"), row=1, col=1, secondary_y=False,

)

fig.add_trace(

    go.Scatter(x=list(range(1,len(Data_spain_after_2months)+1)), y=Data_spain_after_2months['Recovered'], 

               name="Recovered Cases In Spain"),row=1, col=2, secondary_y=False,

)

fig.add_trace(

    go.Scatter(x=list(range(1,len(Data_germany_after_2months)+1)), y=Data_germany_after_2months['Recovered'], 

               name="Recovered Cases In Germany"),row=1, col=3, secondary_y=False,

)







# Update xaxis properties

fig.update_xaxes(title_text="Days",range=[0,60])

fig.update_xaxes(title_text="Days")

fig.update_xaxes(title_text="Days")



# Update yaxis properties

fig.update_yaxes(title_text="Recovered Cases")

fig.update_yaxes(title_text="Recovered Cases")

fig.update_yaxes(title_text="Recovered Cases")



fig.update_layout(

    title_text="Corona virus Recovered cases per country for the first 2 months",

    width=800,

)



fig.show()

fig = make_subplots(rows=1, cols=3,

                    specs=[[{"secondary_y": True}, {"secondary_y": True}, {"secondary_y": True}]],

                   subplot_titles=(

                                   " Italy : "+ str(Data_italy_after_2months['Deaths'].max()),

                               " Spain : "+ str(Data_spain_after_2months['Deaths'].max()),

                                   "  Germany : "+ str(Data_germany_after_2months['Deaths'].max())

                                  ))





fig.add_trace(

    go.Scatter(x=list(range(1,len(Data_italy_after_2months)+1)), y=Data_italy_after_2months['Deaths'], 

               name="Deaths In Italy"), row=1, col=1, secondary_y=False,

)

fig.add_trace(

    go.Scatter(x=list(range(1,len(Data_spain_after_2months)+1)), y=Data_spain_after_2months['Deaths'], 

               name="Deaths In Spain"),row=1, col=2, secondary_y=False,

)

fig.add_trace(

    go.Scatter(x=list(range(1,len(Data_germany_after_2months)+1)), y=Data_germany_after_2months['Deaths'], 

               name="Deaths In Germany"),row=1, col=3, secondary_y=False,

)







# Update xaxis properties

fig.update_xaxes(title_text="Days",range=[0,60])

fig.update_xaxes(title_text="Days")

fig.update_xaxes(title_text="Days")



# Update yaxis properties

fig.update_yaxes(title_text="Deaths")

fig.update_yaxes(title_text="Deaths")

fig.update_yaxes(title_text="Deaths")



fig.update_layout(

    title_text="Corona virus Deaths per country for the first 2 months",

    width=800,

)



fig.show()