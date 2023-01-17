# import modules

import math

import pandas as pd



import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline



import numpy as np

import plotly.express as px

import plotly.figure_factory as ff

import plotly.graph_objects as go



from pandas.plotting import register_matplotlib_converters

register_matplotlib_converters()
#load dataframe

df_it = pd.read_csv('../input/covid19-in-italy/covid19_italy_region.csv')

df_it['Date'] =  pd.to_datetime(df_it['Date'], infer_datetime_format=True)

df_it['NewPositiveCases'] = df_it['NewPositiveCases'].abs()
# new dataframe

df_regioni = df_it.loc[df_it['RegionName'].isin(['Lombardia','Emilia-Romagna','Piemonte','Veneto', 'Toscana'])]



# plot figure

fig = px.line(df_regioni, x="Date", y="TotalHospitalizedPatients", color='RegionName')



fig.update_layout(

    xaxis=dict(

        showline=True,

        showgrid=False,

        showticklabels=True,

        linecolor='rgb(204, 204, 204)',

        linewidth=2,

        ticks='outside',

        tickfont=dict(

            family='Arial',

            size=12,

            color='rgb(82, 82, 82)',

        )))



fig.update_layout(

    yaxis=dict(

        showline=True,

        linecolor='rgb(204, 204, 204)',

        linewidth=2,

        ticks='outside',

        tickfont=dict(

            family='Arial',

            size=12,

            color='rgb(82, 82, 82)',

        )))



annotations = []



# source

annotations.append(dict(xref='paper', yref='paper', x=0.9, y=-0.10,

                              xanchor='center', yanchor='top',

                              text='source: Sito del Dipartimento della Protezione Civile<br>Emergenza Coronavirus: la risposta nazionale<br>original dataset: https://github.com/pcm-dpc/COVID-19',

                              font=dict(family='arial narrow',

                                        size=8,

                                        color='rgb(96,96,96)'),

                              showarrow=False))



fig.update_layout(annotations=annotations)

fig.update_layout(title_text='<b>COVID-19</b>:<br>hospitalized patients in Italy by region',              

    font=dict(family='calibri',

        size=12,

        color='rgb(64,64,64)'),

     legend=dict(

        x=0.75,

        y=0.97,

        bgcolor='rgba(255, 255, 255, 0)',

        bordercolor='rgba(255, 255, 255, 0)'

    ),

    barmode='group',

    bargap=0.15,

    bargroupgap=0.1

)



fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='LightGrey')

fig.update_yaxes(title_text='Patients')

fig.update_xaxes(title_text='Date')

fig.update_yaxes(title_font=dict(size=14))

fig.update_xaxes(title_font=dict(size=14))

fig.update_layout(xaxis_showgrid=False)

   

fig.show()
# new dataframe

df_regions = df_it.loc[df_it['RegionName'].isin(['Lombardia','Emilia-Romagna','Piemonte','Veneto', 'Toscana'])]



# plot figure

fig = px.line(df_regions, x="Date", y="IntensiveCarePatients", color='RegionName')



fig.update_layout(

    xaxis=dict(

        showline=True,

        showgrid=False,

        showticklabels=True,

        linecolor='rgb(204, 204, 204)',

        linewidth=2,

        ticks='outside',

        tickfont=dict(

            family='Arial',

            size=12,

            color='rgb(82, 82, 82)',

        )))



fig.update_layout(

    yaxis=dict(

        showline=True,

        linecolor='rgb(204, 204, 204)',

        linewidth=2,

        ticks='outside',

        tickfont=dict(

            family='Arial',

            size=12,

            color='rgb(82, 82, 82)',

        )))



annotations = []



# source

annotations.append(dict(xref='paper', yref='paper', x=0.85, y=-0.12,

                              xanchor='center', yanchor='top',

                              text='source: Sito del Dipartimento della Protezione Civile:<br>Emergenza Coronavirus: la risposta nazionale<br>original dataset: https://github.com/pcm-dpc/COVID-19',

                              font=dict(family='arial narrow',

                                        size=8,

                                        color='rgb(96,96,96)'),

                              showarrow=False))

fig.update_layout(annotations=annotations)





fig.update_layout(title_text='<b>COVID-19</b>:<br>intensive care patients in Italy by region',

                  

      font=dict(family='calibri',

                                size=12,

                                color='rgb(64,64,64)'),

     legend=dict(

        x=0.75,

        y=0.97,

        bgcolor='rgba(255, 255, 255, 0)',

        bordercolor='rgba(255, 255, 255, 0)'

    ),

    barmode='group',

    bargap=0.15,

    bargroupgap=0.1

)





fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='LightGrey')

fig.update_yaxes(title_text='Patients')

fig.update_xaxes(title_text='Date')

fig.update_yaxes(title_font=dict(size=14))

fig.update_xaxes(title_font=dict(size=14))

fig.update_layout(xaxis_showgrid=False)

   

   

fig.show()
# new dataframe

df_regions = df_it.loc[df_it['RegionName'].isin(['Lombardia','Emilia-Romagna','Piemonte','Veneto', 'Toscana'])]



# plot figure

fig = px.line(df_regions, x="Date", y="HospitalizedPatients", color='RegionName')



fig.update_layout(

    xaxis=dict(

        showline=True,

        showgrid=False,

        showticklabels=True,

        linecolor='rgb(204, 204, 204)',

        linewidth=2,

        ticks='outside',

        tickfont=dict(

            family='Arial',

            size=12,

            color='rgb(82, 82, 82)',

        )))



fig.update_layout(

    yaxis=dict(

        showline=True,

        linecolor='rgb(204, 204, 204)',

        linewidth=2,

        ticks='outside',

        tickfont=dict(

            family='Arial',

            size=12,

            color='rgb(82, 82, 82)',

        )))



annotations = []



# source

annotations.append(dict(xref='paper', yref='paper', x=0.9, y=-0.10,

                              xanchor='center', yanchor='top',

                              text='source: Sito del Dipartimento della Protezione Civile<br>Emergenza Coronavirus: la risposta nazionale<br>original dataset: https://github.com/pcm-dpc/COVID-19',

                              font=dict(family='arial narrow',

                                        size=8,

                                        color='rgb(96,96,96)'),

                              showarrow=False))



fig.update_layout(annotations=annotations)







fig.update_layout(title_text='<b>COVID-19</b>:<br>regular ward care patients in Italy by region',

                  

      font=dict(family='calibri',

                                size=12,

                                color='rgb(64,64,64)'),

     legend=dict(

        x=0.75,

        y=0.97,

        bgcolor='rgba(255, 255, 255, 0)',

        bordercolor='rgba(255, 255, 255, 0)'

    ),

    barmode='group',

    bargap=0.15,

    bargroupgap=0.1

)





fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='LightGrey')

fig.update_yaxes(title_text='Patients')

fig.update_xaxes(title_text='Date')

fig.update_yaxes(title_font=dict(size=14))

fig.update_xaxes(title_font=dict(size=14))

fig.update_layout(xaxis_showgrid=False)

   

fig.show()
# plot figure

fig = px.line(df_regions, x="Date", y="TotalPositiveCases", color='RegionName')



fig.update_layout(

    xaxis=dict(

        showline=True,

        showgrid=False,

        showticklabels=True,

        linecolor='rgb(204, 204, 204)',

        linewidth=2,

        ticks='outside',

        tickfont=dict(

            family='Arial',

            size=12,

            color='rgb(82, 82, 82)',

        )))



fig.update_layout(

    yaxis=dict(

        showline=True,

        linecolor='rgb(204, 204, 204)',

        linewidth=2,

        ticks='outside',

        tickfont=dict(

            family='Arial',

            size=12,

            color='rgb(82, 82, 82)',

        )))



fig.update_layout(title_text='<b>COVID-19</b>:<br>number of confirmed positive cases in Italy',

                                font=dict(family='calibri',

                                size=12,

                                color='rgb(64,64,64)'))



annotations = []



# Source

annotations.append(dict(xref='paper', yref='paper', x=0.88, y=-0.12,

                              xanchor='center', yanchor='top',

                              text='source: Sito del Dipartimento della Protezione Civile<br>Emergenza Coronavirus: la risposta nazionale<br>original dataset: https://github.com/pcm-dpc/COVID-19',

                              font=dict(family='arial narrow',

                                        size=8,

                                        color='rgb(96,96,96)'),

                              showarrow=False))



fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='LightGrey')

fig.update_yaxes(title_text='Confirmed cases')

fig.update_layout(legend_title='<b> Region </b>')

fig.update_layout(annotations=annotations)

fig.update_layout(xaxis_showgrid=False)



fig.show()
# mask dataframe

start_date = '2020-04-01 17:00:00'

mask = (df_it['Date'] > start_date)

df_cases = df_it.loc[mask]



df_masked_lo  = df_cases.loc[df_cases['RegionName'] == 'Lombardia']

df_masked_er  = df_cases.loc[df_cases['RegionName'] == 'Emilia-Romagna']

df_masked_pi  = df_cases.loc[df_cases['RegionName'] == 'Piemonte']

df_masked_ve  = df_cases.loc[df_cases['RegionName'] == 'Veneto']





#plot figure

fig = go.Figure()



fig.add_trace(go.Bar(

    x=df_masked_lo['Date'],

    y=df_masked_lo['NewPositiveCases'],

    name='Lombardy',

    marker_color='red'

))

fig.add_trace(go.Bar(

    x=df_masked_er['Date'],

    y=df_masked_er['NewPositiveCases'],

    name='Emilia-Romagna',

    marker_color='orange'

))



fig.add_trace(go.Bar(

    x=df_masked_pi['Date'],

    y=df_masked_pi['NewPositiveCases'],

    name='Piemonte',

    marker_color='mediumblue'

))

fig.add_trace(go.Bar(

    x=df_masked_ve['Date'],

    y=df_masked_ve['NewPositiveCases'],

    name='Veneto',

    marker_color='green'

))





fig.update_layout(

    xaxis=dict(

        showline=True,

        showgrid=False,

        showticklabels=True,

        linecolor='rgb(204, 204, 204)',

        linewidth=2,

        ticks='outside',

        tickfont=dict(

            family='Arial',

            size=12,

            color='rgb(82, 82, 82)',

        )))



fig.update_layout(

    yaxis=dict(

        showline=True,

        linecolor='rgb(204, 204, 204)',

        linewidth=2,

        ticks='outside',

        tickfont=dict(

            family='Arial',

            size=12,

            color='rgb(82, 82, 82)',

        )))



annotations = []



# source

annotations.append(dict(xref='paper', yref='paper', x=0.9, y=-0.14,

                              xanchor='center', yanchor='top',

                              text='source: Sito del Dipartimento della Protezione Civile<br>Emergenza Coronavirus: la risposta nazionale<br>original dataset: https://github.com/pcm-dpc/COVID-19',

                              font=dict(family='arial narrow',

                                        size=8,

                                        color='rgb(96,96,96)'),

                              showarrow=False))



fig.update_layout(annotations=annotations)



fig.update_layout(barmode='group', xaxis_tickangle=-25)



fig.update_layout(title_text='<b>COVID-19</b>:<br>new daily positive cases in Italy after April 1st',

                  

     font=dict(family='calibri',

        size=12,

        color='rgb(64,64,64)'),

     legend=dict(

        x=0.7,

        y=1.0,

        bgcolor='rgba(255, 255, 255, 0)',

        bordercolor='rgba(255, 255, 255, 0)'

    ),

    barmode='group',

    bargap=0.2,

    bargroupgap=0.2

)





fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='LightGrey')

fig.update_yaxes(title_text='Positive cases')

fig.update_xaxes(title_text='Date')

fig.update_yaxes(title_font=dict(size=14))

fig.update_xaxes(title_font=dict(size=14))

fig.update_layout(xaxis_showgrid=False)

   

fig.show()
# new dataframe

df_lo = df_it[df_it.RegionName == 'Lombardia']



# plot figure

fig = px.bar(df_lo, y='TotalHospitalizedPatients', x='Date', text='TotalHospitalizedPatients', color ='TotalHospitalizedPatients')

fig.data[0].marker.line.width = 0.5

fig.data[0].marker.line.color = "black"



fig.update_traces(texttemplate='%{text:.2s}', textposition='outside')

fig.update_layout(uniformtext_mode='hide')



fig.update_layout(

    xaxis=dict(

        showline=True,

        showgrid=False,

        showticklabels=True,

        linecolor='rgb(204, 204, 204)',

        linewidth=2,

        ticks='outside',

        tickfont=dict(

            family='Arial',

            size=12,

            color='rgb(82, 82, 82)',

        )))



fig.update_layout(

    yaxis=dict(

        showline=True,

        linecolor='rgb(204, 204, 204)',

        linewidth=2,

        ticks='outside',

        tickfont=dict(

            family='Arial',

            size=12,

            color='rgb(82, 82, 82)',

        )))



annotations = []



# source

annotations.append(dict(xref='paper', yref='paper', x=0.9, y=-0.12,

                              xanchor='center', yanchor='top',

                              text='source: Sito del Dipartimento della Protezione Civile<br>Emergenza Coronavirus: la risposta nazionale<br>original dataset: https://github.com/pcm-dpc/COVID-19',

                              font=dict(family='arial narrow',

                                        size=8,

                                        color='rgb(96,96,96)'),

                              showarrow=False))



fig.update_layout(annotations=annotations)

fig.update_layout(title_text='<b>COVID-19</b>:<br>total number of hospitalized patients in Lombardy region',

                  

 font=dict(family='calibri',

            size=12,

            color='rgb(64,64,64)'))



fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='LightGrey')

fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='LightGrey')

fig.update_yaxes(title_text='Hospitalized patients')

fig.update_yaxes(title_font=dict(size=14))

fig.update_xaxes(title_font=dict(size=14))

fig.update_layout(xaxis_showgrid=False)



fig.show()
# new dataframe

data_lombardia = df_it[df_it.RegionName == 'Lombardia']



# plot figure

fig = px.bar(data_lombardia, x='Date', y='IntensiveCarePatients', color='IntensiveCarePatients', text ='IntensiveCarePatients')

fig.data[0].marker.line.width = 0.5

fig.data[0].marker.line.color = "black"





fig.update_traces(texttemplate='%{text:.2s}', textposition='outside')

fig.update_layout(uniformtext_mode='hide')



fig.update_layout(

    xaxis=dict(

        showline=True,

        showgrid=False,

        showticklabels=True,

        linecolor='rgb(204, 204, 204)',

        linewidth=2,

        ticks='outside',

        tickfont=dict(

            family='Arial',

            size=12,

            color='rgb(82, 82, 82)',

        )))



fig.update_layout(

    yaxis=dict(

        showline=True,

        linecolor='rgb(204, 204, 204)',

        linewidth=2,

        ticks='outside',

        tickfont=dict(

            family='Arial',

            size=12,

            color='rgb(82, 82, 82)',

        )))



annotations = []



# source

annotations.append(dict(xref='paper', yref='paper', x=0.9, y=-0.10,

                              xanchor='center', yanchor='top',

                              text='source: Sito del Dipartimento della Protezione Civile<br>Emergenza Coronavirus: la risposta nazionale<br>original dataset: https://github.com/pcm-dpc/COVID-19',

                              font=dict(family='arial narrow',

                                        size=8,

                                        color='rgb(96,96,96)'),

                              showarrow=False))



fig.update_layout(annotations=annotations)



fig.update_layout(title_text='<b>COVID-19</b>:<br>number of intensive care patients in Lombardy region',



 font=dict(family='calibri',

                                size=12,

                                color='rgb(64,64,64)'))



fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='LightGrey')

fig.update_yaxes(title_text='Intensive care patients')

fig.update_yaxes(title_font=dict(size=14))

fig.update_xaxes(title_font=dict(size=14))

fig.update_layout(xaxis_showgrid=False)



fig.show()
# define start date

start_date = '2020-03-24 17:00:00'

mask = (df_lo['Date'] > start_date)

df_lo = df_lo.loc[mask]



# plot figure

fig = go.Figure()

fig.add_trace(go.Bar(

    x=df_lo['Date'],

    y=df_lo['TotalHospitalizedPatients'],

    name='Hospitalized',

    marker_color='indianred'

))

fig.add_trace(go.Bar(

    x=df_lo['Date'],

    y=df_lo['Recovered'],

    name='Recovered',

    marker_color='lightsalmon'

))



fig.update_layout(

    xaxis=dict(

        showline=True,

        showgrid=False,

        showticklabels=True,

        linecolor='rgb(204, 204, 204)',

        linewidth=2,

        ticks='outside',

        tickfont=dict(

            family='Arial',

            size=12,

            color='rgb(82, 82, 82)',

        )))



fig.update_layout(

    yaxis=dict(

        showline=True,

        linecolor='rgb(204, 204, 204)',

        linewidth=2,

        ticks='outside',

        tickfont=dict(

            family='Arial',

            size=12,

            color='rgb(82, 82, 82)',

        )))



annotations = []



# source

annotations.append(dict(xref='paper', yref='paper', x=0.9, y=-0.14,

                              xanchor='center', yanchor='top',

                              text='source: Sito del Dipartimento della Protezione Civile<br>Emergenza Coronavirus: la risposta nazionale<br>original dataset: https://github.com/pcm-dpc/COVID-19',

                              font=dict(family='arial narrow',

                                        size=8,

                                        color='rgb(96,96,96)'),

                              showarrow=False))



fig.update_layout(annotations=annotations)

fig.update_layout(barmode='group', xaxis_tickangle=-25)

fig.update_layout(title_text='<b>COVID-19</b>:<br>level of hospitalization and recovery in Lombardy region',

                  

      font=dict(family='calibri',

        size=12,

        color='rgb(64,64,64)'),

      legend=dict(

        x=0,

        y=1.0,

        bgcolor='rgba(255, 255, 255, 0)',

        bordercolor='rgba(255, 255, 255, 0)'

    ),

    barmode='group',

    bargap=0.15,

    bargroupgap=0.1 

)



fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='LightGrey')

fig.update_yaxes(title_text='Patients')

fig.update_xaxes(title_text='Date')

fig.update_yaxes(title_font=dict(size=14))

fig.update_xaxes(title_font=dict(size=14))

fig.update_layout(xaxis_showgrid=False)

   

fig.show()
# new dataframe

df_er = df_it.loc[df_it['RegionName'] == "Emilia-Romagna"]

start_date = '2020-04-04 17:00:00'

mask = (df_er['Date'] > start_date)

df_er = df_er.loc[mask]





# plot figure

fig = go.Figure()

fig.add_trace(go.Bar(

    x=df_er['Date'],

    y=df_er['TotalHospitalizedPatients'],

    name='Hospitalized',

    marker_color='indianred'

))

fig.add_trace(go.Bar(

    x=df_er['Date'],

    y=df_er['Recovered'],

    name='Recovered',

    marker_color='lightsalmon'

))



fig.update_layout(

    xaxis=dict(

        showline=True,

        showgrid=False,

        showticklabels=True,

        linecolor='rgb(204, 204, 204)',

        linewidth=2,

        ticks='outside',

        tickfont=dict(

            family='Arial',

            size=12,

            color='rgb(82, 82, 82)',

        )))



fig.update_layout(

    yaxis=dict(

        showline=True,

        linecolor='rgb(204, 204, 204)',

        linewidth=2,

        ticks='outside',

        tickfont=dict(

            family='Arial',

            size=12,

            color='rgb(82, 82, 82)',

        )))



annotations = []



# source

annotations.append(dict(xref='paper', yref='paper', x=0.9, y=-0.14,

                              xanchor='center', yanchor='top',

                              text='source: Sito del Dipartimento della Protezione Civile<br>Emergenza Coronavirus: la risposta nazionale<br>original dataset: https://github.com/pcm-dpc/COVID-19',

                              font=dict(family='arial narrow',

                                        size=8,

                                        color='rgb(96,96,96)'),

                              showarrow=False))



fig.update_layout(annotations=annotations)

fig.update_layout(barmode='group', xaxis_tickangle=-25)

fig.update_layout(title_text='<b>COVID-19</b>:<br>level of hospitalization and recovery in Emilia-Romagna region',          

      font=dict(family='calibri',

        size=12,

        color='rgb(64,64,64)'),

      legend=dict(

        x=0,

        y=1.0,

        bgcolor='rgba(255, 255, 255, 0)',

        bordercolor='rgba(255, 255, 255, 0)'

    ),

    barmode='group',

    bargap=0.15,

    bargroupgap=0.1 

)



fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='LightGrey')

fig.update_yaxes(title_text='Patients')

fig.update_xaxes(title_text='Date')

fig.update_yaxes(title_font=dict(size=14))

fig.update_xaxes(title_font=dict(size=14))

fig.update_layout(xaxis_showgrid=False)

   

fig.show()
# new dataframe

data_piemonte = df_it[df_it.RegionName == 'Piemonte']





# plot figure

fig = px.bar(data_piemonte, x='Date', y='IntensiveCarePatients', color='IntensiveCarePatients', text ='IntensiveCarePatients')

fig.data[0].marker.line.width = 0.5

fig.data[0].marker.line.color = "black"



fig.update_traces(texttemplate='%{text:.2s}', textposition='outside')

fig.update_layout(uniformtext_mode='hide')



fig.update_layout(

    xaxis=dict(

        showline=True,

        showgrid=False,

        showticklabels=True,

        linecolor='rgb(204, 204, 204)',

        linewidth=2,

        ticks='outside',

        tickfont=dict(

            family='Arial',

            size=12,

            color='rgb(82, 82, 82)',

        )))



fig.update_layout(

    yaxis=dict(

        showline=True,

        linecolor='rgb(204, 204, 204)',

        linewidth=2,

        ticks='outside',

        tickfont=dict(

            family='Arial',

            size=12,

            color='rgb(82, 82, 82)',

        )))



annotations = []



# source

annotations.append(dict(xref='paper', yref='paper', x=0.9, y=-0.10,

                              xanchor='center', yanchor='top',

                              text='source: Sito del Dipartimento della Protezione Civile<br>Emergenza Coronavirus: la risposta nazionale<br>original dataset: https://github.com/pcm-dpc/COVID-19',

                              font=dict(family='arial narrow',

                                        size=8,

                                        color='rgb(96,96,96)'),

                              showarrow=False))



fig.update_layout(annotations=annotations)









fig.update_layout(title_text='<b>COVID-19</b>:<br>number of intensive care patients in Piemonte region',



 font=dict(family='calibri',

                                size=12,

                                color='rgb(64,64,64)'))





fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='LightGrey')

fig.update_yaxes(title_text='Intensive care patients')

fig.update_yaxes(title_font=dict(size=14))

fig.update_xaxes(title_font=dict(size=14))

fig.update_layout(xaxis_showgrid=False)



fig.show()
# new dataframe

df_pi = df_it.loc[df_it['RegionName'] == "Piemonte"]



# plot figure

fig = px.bar(df_pi, y='TotalHospitalizedPatients', x='Date', text='TotalHospitalizedPatients', color ='TotalHospitalizedPatients')

fig.data[0].marker.line.width = 0.5

fig.data[0].marker.line.color = "black"



fig.update_traces(texttemplate='%{text:.2s}', textposition='outside')

fig.update_layout(uniformtext_mode='hide')



fig.update_layout(

    xaxis=dict(

        showline=True,

        showgrid=False,

        showticklabels=True,

        linecolor='rgb(204, 204, 204)',

        linewidth=2,

        ticks='outside',

        tickfont=dict(

            family='Arial',

            size=12,

            color='rgb(82, 82, 82)',

        )))



fig.update_layout(

    yaxis=dict(

        showline=True,

        linecolor='rgb(204, 204, 204)',

        linewidth=2,

        ticks='outside',

        tickfont=dict(

            family='Arial',

            size=12,

            color='rgb(82, 82, 82)',

        )))



annotations = []



# source

annotations.append(dict(xref='paper', yref='paper', x=0.9, y=-0.12,

                              xanchor='center', yanchor='top',

                              text='source: Sito del Dipartimento della Protezione Civile<br>Emergenza Coronavirus: la risposta nazionale<br>original dataset: https://github.com/pcm-dpc/COVID-19',

                              font=dict(family='arial narrow',

                                        size=8,

                                        color='rgb(96,96,96)'),

                              showarrow=False))



fig.update_layout(annotations=annotations)

fig.update_layout(title_text='<b>COVID-19</b>:<br>total number of hospitalized patients in Piemonte region',

                  

 font=dict(family='calibri',

            size=12,

            color='rgb(64,64,64)'))



fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='LightGrey')

fig.update_yaxes(title_text='Hospitalized patients')

fig.update_yaxes(title_font=dict(size=14))

fig.update_xaxes(title_font=dict(size=14))

fig.update_layout(xaxis_showgrid=False)



fig.show()
# new dataframe

df_pi = df_it.loc[df_it['RegionName'] == "Piemonte"]



start_date = '2020-04-02 17:00:00'

mask = (df_pi['Date'] > start_date)

df_pi = df_pi.loc[mask]



# plot figure

fig = go.Figure()

fig.add_trace(go.Bar(

    x=df_pi['Date'],

    y=df_pi['TotalHospitalizedPatients'],

    name='Hospitalized',

    marker_color='indianred'

))

fig.add_trace(go.Bar(

    x=df_pi['Date'],

    y=df_pi['Recovered'],

    name='Recovered',

    marker_color='lightsalmon'

))



fig.update_layout(

    xaxis=dict(

        showline=True,

        showgrid=False,

        showticklabels=True,

        linecolor='rgb(204, 204, 204)',

        linewidth=2,

        ticks='outside',

        tickfont=dict(

            family='Arial',

            size=12,

            color='rgb(82, 82, 82)',

        )))



fig.update_layout(

    yaxis=dict(

        showline=True,

        linecolor='rgb(204, 204, 204)',

        linewidth=2,

        ticks='outside',

        tickfont=dict(

            family='Arial',

            size=12,

            color='rgb(82, 82, 82)',

        )))



annotations = []



# source

annotations.append(dict(xref='paper', yref='paper', x=0.9, y=-0.14,

                              xanchor='center', yanchor='top',

                              text='source: Sito del Dipartimento della Protezione Civile<br>Emergenza Coronavirus: la risposta nazionale<br>original dataset: https://github.com/pcm-dpc/COVID-19',

                              font=dict(family='arial narrow',

                                        size=8,

                                        color='rgb(96,96,96)'),

                              showarrow=False))



fig.update_layout(annotations=annotations)

fig.update_layout(barmode='group', xaxis_tickangle=-25)

fig.update_layout(title_text='<b>COVID-19</b>:<br>level of hospitalization and recovery in Piemonte region',          

      font=dict(family='calibri',

        size=12,

        color='rgb(64,64,64)'),

      legend=dict(

        x=0,

        y=1.0,

        bgcolor='rgba(255, 255, 255, 0)',

        bordercolor='rgba(255, 255, 255, 0)'

    ),

    barmode='group',

    bargap=0.15,

    bargroupgap=0.1 

)



fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='LightGrey')

fig.update_yaxes(title_text='Patients')

fig.update_xaxes(title_text='Date')

fig.update_yaxes(title_font=dict(size=14))

fig.update_xaxes(title_font=dict(size=14))

fig.update_layout(xaxis_showgrid=False)

   

fig.show()
# new dataframe

data_er = df_it[df_it.RegionName == 'Veneto']





# plot figure

fig = px.bar(data_er, x='Date', y='IntensiveCarePatients', color='IntensiveCarePatients', text='IntensiveCarePatients')

fig.data[0].marker.line.width = 0.5

fig.data[0].marker.line.color = "black"



fig.update_traces(texttemplate='%{text:.2s}', textposition='outside')

fig.update_layout(uniformtext_minsize=8)



fig.update_layout(

    xaxis=dict(

        showline=True,

        showgrid=False,

        showticklabels=True,

        linecolor='rgb(204, 204, 204)',

        linewidth=2,

        ticks='outside',

        tickfont=dict(

            family='Arial',

            size=12,

            color='rgb(82, 82, 82)',

        )))



fig.update_layout(

    yaxis=dict(

        showline=True,

        linecolor='rgb(204, 204, 204)',

        linewidth=2,

        ticks='outside',

        tickfont=dict(

            family='Arial',

            size=12,

            color='rgb(82, 82, 82)',

        )))



annotations = []



# source

annotations.append(dict(xref='paper', yref='paper', x=0.9, y=-0.10,

                              xanchor='center', yanchor='top',

                              text='source: Sito del Dipartimento della Protezione Civile<br>Emergenza Coronavirus: la risposta nazionale<br>original dataset: https://github.com/pcm-dpc/COVID-19',

                              font=dict(family='arial narrow',

                                        size=8,

                                        color='rgb(96,96,96)'),

                              showarrow=False))





fig.update_layout(annotations=annotations)



fig.update_layout(title_text='<b>COVID-19</b>:<br>number of intensive care patients in Veneto region',

                  

 font=dict(family='calibri',

            size=12,

            color='rgb(64,64,64)'))



fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='LightGrey')

fig.update_yaxes(title_text='Intensive care patients')

fig.update_yaxes(title_font=dict(size=14))

fig.update_xaxes(title_font=dict(size=14))

fig.update_layout(xaxis_showgrid=False)



fig.show()
# new dataframe

df_vene = df_it[df_it.RegionName == 'Veneto']



# plot figure

fig = px.bar(df_vene, y='TotalHospitalizedPatients', x='Date', text='TotalHospitalizedPatients', color ='TotalHospitalizedPatients')

fig.data[0].marker.line.width = 0.5

fig.data[0].marker.line.color = "black"



fig.update_traces(texttemplate='%{text:.2s}', textposition='outside')

fig.update_layout(uniformtext_mode='hide')



fig.update_layout(

    xaxis=dict(

        showline=True,

        showgrid=False,

        showticklabels=True,

        linecolor='rgb(204, 204, 204)',

        linewidth=2,

        ticks='outside',

        tickfont=dict(

            family='Arial',

            size=12,

            color='rgb(82, 82, 82)',

        )))



fig.update_layout(

    yaxis=dict(

        showline=True,

        linecolor='rgb(204, 204, 204)',

        linewidth=2,

        ticks='outside',

        tickfont=dict(

            family='Arial',

            size=12,

            color='rgb(82, 82, 82)',

        )))



annotations = []



# source

annotations.append(dict(xref='paper', yref='paper', x=0.9, y=-0.10,

                              xanchor='center', yanchor='top',

                              text='source: Sito del Dipartimento della Protezione Civile<br>Emergenza Coronavirus: la risposta nazionale<br>original dataset: https://github.com/pcm-dpc/COVID-19',

                              font=dict(family='arial narrow',

                                        size=8,

                                        color='rgb(96,96,96)'),

                              showarrow=False))



fig.update_layout(annotations=annotations)



fig.update_layout(title_text='<b>COVID-19</b>:<br>total number of hospitalized patients in Veneto region',

                  

 font=dict(family='calibri',

            size=12,

            color='rgb(64,64,64)'))



fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='LightGrey')

fig.update_yaxes(title_text='Hospitalized patients')

fig.update_yaxes(title_font=dict(size=14))

fig.update_xaxes(title_font=dict(size=14))

fig.update_layout(xaxis_showgrid=False)



fig.show()
# new dataframe

df_ve = df_it.loc[df_it['RegionName'] == "Veneto"]



start_date = '2020-04-02 17:00:00'

mask = (df_ve['Date'] > start_date)

df_ve = df_ve.loc[mask]



# plot figure

fig = go.Figure()

fig.add_trace(go.Bar(

    x=df_ve['Date'],

    y=df_ve['TotalHospitalizedPatients'],

    name='Hospitalized',

    marker_color='indianred'

))

fig.add_trace(go.Bar(

    x=df_ve['Date'],

    y=df_ve['Recovered'],

    name='Recovered',

    marker_color='lightsalmon'

))



fig.update_layout(

    xaxis=dict(

        showline=True,

        showgrid=False,

        showticklabels=True,

        linecolor='rgb(204, 204, 204)',

        linewidth=2,

        ticks='outside',

        tickfont=dict(

            family='Arial',

            size=12,

            color='rgb(82, 82, 82)',

        )))



fig.update_layout(

    yaxis=dict(

        showline=True,

        linecolor='rgb(204, 204, 204)',

        linewidth=2,

        ticks='outside',

        tickfont=dict(

            family='Arial',

            size=12,

            color='rgb(82, 82, 82)',

        )))



annotations = []



# source

annotations.append(dict(xref='paper', yref='paper', x=0.9, y=-0.15,

                              xanchor='center', yanchor='top',

                              text='source: Sito del Dipartimento della Protezione Civile<br>Emergenza Coronavirus: la risposta nazionale<br>original dataset: https://github.com/pcm-dpc/COVID-19',

                              font=dict(family='arial narrow',

                                        size=8,

                                        color='rgb(96,96,96)'),

                              showarrow=False))



fig.update_layout(annotations=annotations)

fig.update_layout(barmode='group', xaxis_tickangle=-25)

fig.update_layout(title_text='<b>COVID-19</b>:<br>level of hospitalization and recovery in Veneto region',

                  

      font=dict(family='calibri',

        size=12,

        color='rgb(64,64,64)'),

      legend=dict(

        x=0,

        y=1.0,

        bgcolor='rgba(255, 255, 255, 0)',

        bordercolor='rgba(255, 255, 255, 0)'

    ),

    barmode='group',

    bargap=0.15,

    bargroupgap=0.1

)



fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='LightGrey')

fig.update_yaxes(title_text='Patients')

fig.update_xaxes(title_text='Date')

fig.update_yaxes(title_font=dict(size=14))

fig.update_xaxes(title_font=dict(size=14))

fig.update_layout(xaxis_showgrid=False)

   

fig.show()
# new dataframe

url="https://covidtracking.com/api/v1/states/daily.csv"

df_us=pd.read_csv(url)



df_us = df_us.rename(columns = {'dateChecked':'Date'})

df_us.drop(df_us.index[df_us['Date'] == 'Invalid DateTime'], inplace = True)



df_us['Date'] =  pd.to_datetime(df_us['Date'], infer_datetime_format=True)



df_la = df_us



#df_us.head(20)





#df_us_two = df_us[df_us['dateChecked'].notna()]

#pd.set_option('display.max_columns', None)

#df_us_two.Date.apply(str)

#df_us.head(10)

#df_us['Date'] =  pd.to_datetime(df_us['Date'], format = '%d-%b-%y %H.%M.%S.%f %p')

#df_us['Date']= pd.to_datetime(df_us['Date'],dayfirst=True)

#df_us['NewDate'] = df_us['Date'].dt.date

#df_us['NewDate'] =  pd.to_datetime(df_us['NewDate'], infer_datetime_format=True)





# drop column

dropped = ['hash']

df_us = df_us.drop(dropped, axis=1)



# mask dataframe

df_la = df_us[df_us.state == 'NY']



df_la['NewDate'] = df_la['Date'].dt.date

df_la['NewDate'] =  pd.to_datetime(df_la['NewDate'], infer_datetime_format=True)



start_date = '2020-05-01'

mask = (df_la['NewDate'] > start_date)

df_la = df_la.loc[mask]



df_la = df_la.drop_duplicates(subset='hospitalizedCurrently', keep="first")

df_la = df_la.drop_duplicates(subset='NewDate', keep="first")





# plot figure

fig = px.bar(df_la, y='hospitalizedCurrently', x='NewDate', text='hospitalizedCurrently', color ='hospitalizedCurrently')

fig.data[0].marker.line.width = 0.5

fig.data[0].marker.line.color = "black"



fig.update_traces(texttemplate='%{text:.2s}', textposition='outside')

fig.update_layout(uniformtext_mode='hide')



fig.update_layout(

    xaxis=dict(

        showline=True,

        showgrid=False,

        showticklabels=True,

        linecolor='rgb(204, 204, 204)',

        linewidth=2,

        ticks='outside',

       tickfont=dict(

            family='Arial',

            size=12,

            color='rgb(82, 82, 82)',

        )))



fig.update_layout(

    yaxis=dict(

        showline=True,

        linecolor='rgb(204, 204, 204)',

        linewidth=2,

        ticks='outside',

        tickfont=dict(

            family='Arial',

            size=12,

            color='rgb(82, 82, 82)',

        )))



annotations = []



# source

annotations.append(dict(xref='paper', yref='paper', x=0.88, y=-0.10,

                              xanchor='center', yanchor='top',

                              text='Source: COVID-19 Tracking Project<br>original dataset: https://covidtracking.com',

                              font=dict(family='arial narrow',

                                        size=8,

                                        color='rgb(96,96,96)'),

                              showarrow=False))





fig.update_layout(annotations=annotations)



fig.update_layout(title_text='<b>COVID-19</b>:<br>number of current hospitalizations in New York state',

                  

                font=dict(family='calibri',

                                size=12,

                                color='rgb(64,64,64)'))



fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='LightGrey')

fig.update_yaxes(title_text='Patients')

fig.update_yaxes(title_font=dict(size=14))

fig.update_xaxes(title_font=dict(size=14))

fig.update_layout(xaxis_showgrid=False)



fig.show()
# new dataframe

url="https://covidtracking.com/api/v1/states/daily.csv"

df_us=pd.read_csv(url)



df_us = df_us.rename(columns = {'dateChecked':'Date'})

df_us.drop(df_us.index[df_us['Date'] == 'Invalid DateTime'], inplace = True)



df_us['Date'] =  pd.to_datetime(df_us['Date'], infer_datetime_format=True)



#df_la = df_us

# drop column

dropped = ['hash']

df_us = df_us.drop(dropped, axis=1)





df_us['NewDate'] = df_us['Date'].dt.date

df_us['NewDate'] =  pd.to_datetime(df_us['NewDate'], infer_datetime_format=True)



#df['time'] = df['full_date'].dt.time





# mask dataframe

start_date = '2020-06-01'

mask = (df_us['NewDate'] > start_date)

df_us_cases = df_us.loc[mask]



df_masked_nys  = df_us_cases.loc[df_us_cases['state'] == 'NY']

df_masked_ca  = df_us_cases.loc[df_us_cases['state'] == 'CA']

df_masked_fl  = df_us_cases.loc[df_us_cases['state'] == 'FL']



df_masked_nys = df_masked_nys.drop_duplicates(subset='positiveIncrease', keep="first")

df_masked_nys = df_masked_nys.drop_duplicates(subset='NewDate', keep="first")



df_masked_ca = df_masked_ca.drop_duplicates(subset='positiveIncrease', keep="first")

df_masked_ca = df_masked_ca.drop_duplicates(subset='NewDate', keep="first")



df_masked_fl = df_masked_fl.drop_duplicates(subset='positiveIncrease', keep="first")

df_masked_fl = df_masked_fl.drop_duplicates(subset='NewDate', keep="first")



# plot figure

fig = go.Figure()



fig.add_trace(go.Bar(

    x=df_masked_nys['NewDate'],

    y=df_masked_nys['positiveIncrease'],

    name='New York',

    marker_color='red'

))

fig.add_trace(go.Bar(

    x=df_masked_ca['NewDate'],

    y=df_masked_ca['positiveIncrease'],

    name='California',

    marker_color='orange'

))





fig.add_trace(go.Bar(

    x=df_masked_fl['NewDate'],

    y=df_masked_fl['positiveIncrease'],

    name='Florida',

    marker_color='green'

))





fig.update_layout(

    xaxis=dict(

        showline=True,

        showgrid=False,

        showticklabels=True,

        linecolor='rgb(204, 204, 204)',

        linewidth=2,

        ticks='outside',

        tickfont=dict(

            family='Arial',

            size=12,

            color='rgb(82, 82, 82)',

        )))



fig.update_layout(

    yaxis=dict(

        showline=True,

        linecolor='rgb(204, 204, 204)',

        linewidth=2,

        ticks='outside',

        tickfont=dict(

            family='Arial',

            size=12,

            color='rgb(82, 82, 82)',

        )))



annotations = []



# source

annotations.append(dict(xref='paper', yref='paper', x=0.9, y=-0.14,

                              xanchor='center', yanchor='top',

                              text='Source: COVID-19 Tracking Project<br>original dataset: https://covidtracking.com',

                              font=dict(family='arial narrow',

                                        size=8,

                                        color='rgb(96,96,96)'),

                              showarrow=False))



fig.update_layout(annotations=annotations)



fig.update_layout(barmode='group', xaxis_tickangle=-25)



fig.update_layout(title_text='<b>COVID-19</b>:<br>new daily positive cases in some U.S. states after June 1st',

                  

     font=dict(family='calibri',

        size=12,

        color='rgb(64,64,64)'),

     legend=dict(

        x=0.05,

        y=1.0,

        bgcolor='rgba(255, 255, 255, 0)',

        bordercolor='rgba(255, 255, 255, 0)'

    ),

    barmode='group',

    bargap=0.15,

    bargroupgap=0.1

)



fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='LightGrey')

fig.update_yaxes(title_text='Positive cases')

fig.update_xaxes(title_text='Date')

fig.update_yaxes(title_font=dict(size=14))

fig.update_xaxes(title_font=dict(size=14))

fig.update_layout(xaxis_showgrid=False)

   

fig.show()
# new dataframe

url="https://covidtracking.com/api/v1/states/daily.csv"

df_us=pd.read_csv(url)



df_us = df_us.rename(columns = {'dateChecked':'Date'})

df_us.drop(df_us.index[df_us['Date'] == 'Invalid DateTime'], inplace = True)



df_us['Date'] =  pd.to_datetime(df_us['Date'], infer_datetime_format=True)



#df_la = df_us

# drop column

dropped = ['hash']

df_us = df_us.drop(dropped, axis=1)





df_us['NewDate'] = df_us['Date'].dt.date

df_us['NewDate'] =  pd.to_datetime(df_us['NewDate'], infer_datetime_format=True)



df_states = df_us.loc[df_us['state'].isin(['TX', 'IL', 'NY', 'CA', 'FL'])]



# mask dataframe

start_date = '2020-03-15'

mask = (df_states['NewDate'] > start_date)

df_states = df_states.loc[mask]



# plot figure

fig = px.line(df_states, x="NewDate", y="positive",color='state')



fig.update_layout(

    xaxis=dict(

        showline=True,

        showgrid=False,

        showticklabels=True,

        linecolor='rgb(204, 204, 204)',

        linewidth=2,

        ticks='outside',

        tickfont=dict(

            family='Arial',

            size=12,

            color='rgb(82, 82, 82)',

        )))



fig.update_layout(

    yaxis=dict(

        showline=True,

        linecolor='rgb(204, 204, 204)',

        linewidth=2,

        ticks='outside',

        tickfont=dict(

            family='Arial',

            size=12,

            color='rgb(82, 82, 82)',

        )))



annotations = []



# Source

annotations.append(dict(xref='paper', yref='paper', x=0.88, y=-0.10,

                              xanchor='center', yanchor='top',

                              text='Source: COVID-19 Tracking Project<br>original dataset: https://covidtracking.com',

                              font=dict(family='arial narrow',

                                        size=8,

                                        color='rgb(96,96,96)'),

                              showarrow=False))





fig.update_layout(annotations=annotations)



fig.update_layout(title_text='<b>COVID-19</b>:<br>number of confirmed positive cases in some U.S. states',

                  

                  

                font=dict(family='calibri',

                                size=12,

                                color='rgb(64,64,64)'))



fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='LightGrey')

fig.update_yaxes(title_text='Confirmed cases')

fig.update_yaxes(title_font=dict(size=14))

fig.update_xaxes(title_font=dict(size=14))

fig.update_layout(xaxis_showgrid=False)



fig.show()