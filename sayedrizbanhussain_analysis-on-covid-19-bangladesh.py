# Importing Libraries

import numpy as np

import pandas as pd 

import seaborn as sns

import matplotlib.pyplot as plt

import matplotlib.dates as mdates

import plotly.express as px

from datetime import date, timedelta

from sklearn.cluster import KMeans

import plotly.offline as py

from statsmodels.tsa.arima_model import ARIMA

from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

import statsmodels.api as sm

from keras.models import Sequential

from keras.layers import LSTM,Dense

from keras.layers import Dropout

from sklearn.neural_network import MLPRegressor

from sklearn.preprocessing import MinMaxScaler

from sklearn.linear_model import LinearRegression, BayesianRidge

from sklearn.model_selection import RandomizedSearchCV, train_test_split

from sklearn.preprocessing import PolynomialFeatures

from sklearn.tree import DecisionTreeRegressor

from sklearn.svm import SVR

from sklearn.metrics import mean_squared_error, mean_absolute_error

from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator

import plotly.offline as py

py.init_notebook_mode(connected=True)

from plotly.subplots import make_subplots

import plotly.graph_objects as go

import datetime
import folium

import pandas as pd

import numpy as np

df = pd.read_csv('../input/bangladesh-covid19-data-may-21-2020/covid-19-DistrictWiseCaseMap.csv')

m = folium.Map(

    tiles='OpenStreetMap',

    location=[24.25135,89.91671],

    zoom_start=7,

    zoom_control=True,

    scrollWheelZoom=False

)

for index,rows in df.iterrows():

    if rows['lat'] and rows['lon']:

        folium.Marker([rows['lat'], rows['lon']], tooltip=rows['District']+" - "+str(rows['Cases'])).add_to(m)

folium.LayerControl().add_to(m)

m
dataFromCsv = pd.read_csv('../input/bangladesh-covid19-data-may-21-2020/covid-19-PerDayCaseSummaryBD.csv',parse_dates=["Date"])

totalCaseNumber = dataFromCsv['Case'].sum()

totalDeathNumber = dataFromCsv['Death'].sum()

totalRecoveredNumber = dataFromCsv['Recovered'].sum()

totalPatientNumber = totalCaseNumber - (totalDeathNumber+totalRecoveredNumber)



slices = [totalPatientNumber,totalDeathNumber,totalRecoveredNumber]

activities = ['Patient','Death','Recovered']



dataFrame = {

    "slices" : slices,

    "activities" : activities

}

dataset = pd.DataFrame(dataFrame)

fig = px.pie(dataset,

             values="slices",

             names="activities",

             template="presentation",

             labels = {'slices' : 'No Cases', 'activities' : 'Status'},

             color_discrete_sequence=['#4169E1', '#DC143C', '#006400'],

             width=800,

             height=450,

             hole=0.6)

fig.update_traces(rotation=180, pull=0.05, textinfo="percent+label")

py.offline.iplot(fig)
dataFromCsv = pd.read_csv('../input/bangladesh-covid19-data-may-21-2020/covid-19-PerDayCaseSummaryBD.csv',parse_dates=["Date"])

fig = go.Figure(data=[go.Table(

    header=dict(values=['<b>New Cases</b>','<b>Death</b>','<b>Recovered</b>','<b>Tested</b>'],

                fill_color='blue',

                align='center',

                line_color='darkslategray',

                font = dict(color = 'White', size = 18)),

    cells=dict(values=[dataFromCsv['Case'].iloc[-1],dataFromCsv['Death'].iloc[-1], dataFromCsv['Recovered'].iloc[-1], dataFromCsv['Tested'].iloc[-1]],

               fill_color='White',

               line_color='darkslategray',

               align='center',

               height=40,

               font = dict(color = 'Black', size = 24)))

])



datetimeobject = datetime.datetime.strptime(str(dataFromCsv['Date'].iloc[-1]),'%Y-%m-%d %H:%M:%S')

endDate = datetimeobject.strftime('%d-%b-%Y')

title = "Update On " + str(endDate)

fig.update_layout(

    title={

        'text': title,

        'y':.9,

        'x':0.5,

        'xanchor': 'center',

        'yanchor': 'top'})



fig.show()
dataFromCsv = pd.read_csv('../input/bangladesh-covid19-data-may-21-2020/covid-19-PerDayCaseSummaryBD.csv',parse_dates=["Date"])

fig = go.Figure(data=[

    go.Bar(name='Positove', x=dataFromCsv['Date'], y=dataFromCsv['Case'], marker_color = '#4169E1'),

    go.Bar(name='Death', x=dataFromCsv['Date'], y=dataFromCsv['Death'], marker_color = '#DC143C'),

    go.Bar(name='Recovered', x=dataFromCsv['Date'], y=dataFromCsv['Recovered'], marker_color = '#006400')

])

# Change the bar mode

fig.update_layout(barmode='stack',template="presentation",title_text='Per Day Case Summary')

fig.show()
dataFromCsv = pd.read_csv('../input/bangladesh-covid19-data-may-21-2020/covid-19-PerDayCaseSummaryBD.csv',parse_dates=["Date"], index_col='Date')

dataFromCsv = dataFromCsv.cumsum()

dataFromCsv = dataFromCsv.reset_index()

datetimeobject = datetime.datetime.strptime(str(dataFromCsv['Date'].iloc[1]),'%Y-%m-%d %H:%M:%S')

startDate = datetimeobject.strftime('%d-%b-%Y')

datetimeobject = datetime.datetime.strptime(str(dataFromCsv['Date'].iloc[-1]),'%Y-%m-%d %H:%M:%S')

endDate = datetimeobject.strftime('%d-%b-%Y')

xlabel = str(str(startDate) + ' UPTO ' + str(endDate))

fig = go.Figure()

fig.add_trace(go.Scatter(x=dataFromCsv.Date, y=dataFromCsv.Case,

                    mode='lines',

                    name='Case'))

fig.add_trace(go.Scatter(x=dataFromCsv.Date, y=dataFromCsv.Death,

                    mode='lines',

                    name='Death'))

fig.add_trace(go.Scatter(x=dataFromCsv.Date, y=dataFromCsv.Recovered,

                    mode='lines', 

                    name='Recovered'))

annotations = []

annotations.append(dict(x=dataFromCsv['Date'].iloc[-1], y=dataFromCsv['Case'].iloc[-1],

                      xanchor='left', yanchor='top',

                      text=' {}'.format(dataFromCsv['Case'].iloc[-1]),

                      font=dict(family='Arial',

                                size=18,

                                color='#4169E1'),

                      showarrow=True,

                      borderwidth = 1))

annotations.append(dict(x=dataFromCsv['Date'].iloc[-1], y=dataFromCsv['Death'].iloc[-1],

                      xanchor='left', yanchor='top',

                      text=' {}'.format(dataFromCsv['Death'].iloc[-1]),

                      font=dict(family='Arial',

                                size=18,

                                color='#DC143C'),

                      showarrow=True,

                      borderwidth = 1))

annotations.append(dict(x=dataFromCsv['Date'].iloc[-1], y=dataFromCsv['Recovered'].iloc[-1],

                      xanchor='left', yanchor='top',

                      text=' {}'.format(dataFromCsv['Recovered'].iloc[-1]),

                      font=dict(family='Arial',

                                size=18,

                                color='#FF7F50'),

                      showarrow=True,

                      borderwidth = 1))



fig.update_layout(title='Bangladesh Cases Summary Trend',

                   xaxis_title=xlabel,

                   yaxis_title='Number of Cases',

                   template="presentation",

                   annotations=annotations)



fig.show()
dataFromCsv = pd.read_csv('../input/bangladesh-covid19-data-may-21-2020/covid-19-PerDayCaseSummaryBD.csv',parse_dates=["Date"], index_col='Date')

dataset = dataFromCsv.cumsum()



postitiveCases = int(dataset['Case'].iloc[-1])

negativeCases = int(dataset['Tested'].iloc[-1]) - postitiveCases

slices = [postitiveCases,negativeCases]

activities = ['Covid-19 Positive','Covid-19 Negative']

cols = ['#DC143C','#4169E1']

fig = make_subplots(

    rows=1, cols=2,

    subplot_titles=("Test Results", "Test Results 24 Hour"),

    specs=[[{"type": "domain"}, {"type": "domain"}]]

)

fig.add_trace(

    go.Pie(labels = activities,

           values=slices,

           marker_colors=cols,pull=0.05,textinfo="percent+label"),

           row=1, col=1)



dataset = dataFromCsv.reset_index()

postitiveCases = int(dataset['Case'].iloc[-1])

negativeCases = int(dataset['Tested'].iloc[-1]) - postitiveCases

slices = [postitiveCases,negativeCases]

activities = ['Covid-19 Positive','Covid-19 Negative']

cols = ['#DC143C','#4169E1']



fig.add_trace(

    go.Pie(labels = activities,

           values=slices,

           marker_colors=cols,pull=0.05,textinfo="percent+label"),

           row=1, col=2)

fig.update_traces(rotation = 180,hole=0, hoverinfo="label+percent+value",textfont_size=14)

fig.update_layout(height=500, showlegend=True, template="presentation")



fig.show()
dataFromCsv = pd.read_csv('../input/bangladesh-covid19-data-may-21-2020/covid-19-PerDayCaseSummaryBD.csv')

dataFromCsv = dataFromCsv[1:]

positiveRatio = round((dataFromCsv['Case'] / dataFromCsv['Tested'])*100,2)



positiveRatioDF = pd.Series(positiveRatio,name="Ratio")

datavsratio = pd.concat([dataFromCsv['Date'],positiveRatioDF], axis=1)



fig = px.bar(datavsratio, x='Date', y='Ratio',

             hover_data=['Date', 'Ratio'],

             color='Ratio',

             color_continuous_scale= "Reds",

             labels={'Ratio' : 'Positive Case Ratio(%)'},

             template="ggplot2",

             height = 600,

             title='Tests Vs Positove Cases Ratio Per Day')

fig.show()
CoronaTestCaseNo = pd.read_csv('../input/bangladesh-covid19-data-may-21-2020/covid-19-TestResultsBD.csv')

iedrcCases = int(CoronaTestCaseNo['Lab'].iloc[1])

otherCases = int(CoronaTestCaseNo['OLab'].iloc[1])

slices = [iedrcCases,otherCases]

activities = ['In IEDCR','In Other Labs']

cols = ['#FF6347','#000080']

fig = make_subplots(

    rows=1, cols=2,

    subplot_titles=("Tests Conducted Total", "Tests Conducted 24 Hour"),

    specs=[[{"type": "domain"}, {"type": "domain"}]]

)

fig.add_trace(

    go.Pie(labels = activities,

           values=slices,

           marker_colors=['#FF6347','#000080'],pull=0.05, textinfo="percent+label"),

           row=1, col=1)



iedrcCases = int(CoronaTestCaseNo['Lab'].iloc[0])

otherCases = int(CoronaTestCaseNo['OLab'].iloc[0])

slices = [iedrcCases,otherCases]

activities = ['In IEDCR','In Other Labs']

cols = ['#FF6347','#000080']



fig.add_trace(

    go.Pie(labels = activities,

           values=slices,

           marker_colors=['#FF6347','#000080'],pull=0.05, textinfo="percent+label"),

           row=1, col=2)

fig.update_traces(rotation = 180,hole=.4, hoverinfo="label+percent",textfont_size=15)

fig.update_layout(height=500, showlegend=True, template="presentation")



fig.show()
caseBangladesh = pd.read_csv('../input/bangladesh-covid19-data-may-21-2020/covid-19-PerDayCaseSummaryBD.csv')

result = caseBangladesh.sort_values(['Tested'], ascending= True)

fig = px.area(dataFromCsv, x="Tested", y="Case", 

              template="presentation", 

              labels = {'Tested': 'Total Test', 'Case': 'Positive Cases'},

              title='Tested Vs Positive Cases Trend',

              color_discrete_sequence=['#4169E1'])



fig.update_traces(textposition='top left',mode='lines+markers')

fig.show()
df = pd.read_csv('../input/bangladesh-covid19-data-may-21-2020/covid-19-DistrictWiseCaseMap.csv')

caseBangladesh = df.groupby(['Division']).agg('sum')

caseBangladesh = caseBangladesh.reset_index()

caseBangladesh = caseBangladesh.sort_values(by=['Cases'], ascending=False)

fig = make_subplots(

    rows=1, cols=2,

    specs=[[{"type": "table"}, {"type": "domain"}]]

)

fig.add_trace(

    go.Table(

        header=dict(

            values=[str(caseBangladesh.columns[0]),str(caseBangladesh.columns[1])],

            font=dict(size=15),

            align="center"

        ),

        cells=dict(

            values=[caseBangladesh[k].tolist() for k in caseBangladesh.columns[0:2]],

            align = "center")

    ),

    row=1, col=1

)



slices=caseBangladesh['Cases'].tolist()

activities=caseBangladesh['Division'].tolist()

fig.add_trace(

    go.Pie(labels = activities,

           values=slices,pull=0.05,hole=.5,textinfo="percent", rotation=210),

           row=1, col=2)

# fig.update_traces(rotation = 180,hole=0.5, hoverinfo="label+percent+value",textfont_size=13)

fig.update_layout(height=500, 

                  showlegend=True, 

                  template="presentation",

                  title_text="Division Wise Covid-19 Cases")

fig.show()



caseBangladesh = pd.read_csv('../input/bangladesh-covid19-data-may-21-2020/covid-19-DhakaCityArea.csv')

caseBangladesh

result = caseBangladesh.sort_values(['Number'], ascending= True)

# print(result)

numberOfCases = result[-20:]

fig = px.bar(numberOfCases, x='Area', y='Number',

             hover_data=['Area', 'Number'],

             labels={'Area' : 'Affected Area', 'Number' : 'Number of Cases' },

             template="xgridoff",

             text='Number',

             title="Dhaka City Area Wise Covid-19 Cases(Top 20)",

             height = 600)

fig.show()
caseBangladesh = pd.read_csv('../input/bangladesh-covid19-data-may-21-2020/covid-19-DhakaCityArea.csv')

caseBangladesh

result = caseBangladesh.sort_values(['Number'], ascending= True)

numberOfCases = result[0:20]

fig = px.bar(numberOfCases, x='Area', y='Number',

             hover_data=['Area', 'Number'],

             labels={'Area' : 'Affected Area', 'Number' : 'Number of Cases' },

             template="xgridoff",

             text='Number',

             title="Dhaka City Area Wise Covid-19 Cases(Bottom 20)",

             range_y=[0,10],

             height = 600)

fig.show()
caseBangladesh = pd.read_csv('../input/bangladesh-covid19-data-may-21-2020/covid-19-DistrictWiseCaseMap.csv')

dhaka=caseBangladesh[caseBangladesh['Division']=='Dhaka']

fig = px.treemap(dhaka, path=['District', 'Cases'], values='Cases',

                  color='Cases', hover_data=['District'],

                  color_continuous_scale='Reds',

                  color_continuous_midpoint=np.average(dhaka['Cases'], weights=dhaka['Cases']))

fig.update_layout(height=500, 

                  showlegend=True, 

                  template="presentation",

                  title_text="Dhaka Division Covid-19 Cases")

fig.show()

caseBangladesh = pd.read_csv('../input/bangladesh-covid19-data-may-21-2020/covid-19-DistrictWiseCaseMap.csv')

Chattogram=caseBangladesh[caseBangladesh['Division']=='Chattogram']

fig = px.treemap(Chattogram, path=['District', 'Cases'], values='Cases',

                  color='Cases', hover_data=['District'],

                  color_continuous_scale='Reds',

                  color_continuous_midpoint=np.average(Chattogram['Cases'], weights=Chattogram['Cases']))

fig.update_layout(height=500, 

                  showlegend=True, 

                  template="presentation",

                  title_text="Chattogram Division Covid-19 Cases")

fig.show()
gender = pd.read_csv('../input/bangladesh-covid19-data-may-21-2020/covid-19-GenderWiseCase.csv')

fig = px.pie(gender, values='percentage', 

             names='gender',

             hole=.5,

             color_discrete_sequence=px.colors.sequential.RdBu, 

             template="presentation")

fig.update_layout(title="Gender Wise Covid-19 Cases")

fig.show()



gender = pd.read_csv('../input/bangladesh-covid19-data-may-21-2020/covid-19-AgeWiseCase.csv')

fig = px.pie(gender, values='percentage', 

             names='age_range',

             hole=.5,

             color_discrete_sequence=px.colors.sequential.RdBu,

             template="presentation")

fig.update_layout(title="Age Wise Covid-19 Cases")

fig.show()
# Read Data from dataset

dataset = pd.read_csv('../input/bangladesh-covid19-data-may-21-2020/covid-19-PerDayCaseSummaryBD.csv',parse_dates=["Date"], index_col='Date')

accumulated_count_bd = dataset.cumsum()

accumulated_count_bd = accumulated_count_bd.reset_index()





dataFromCsv = pd.read_csv('../input/bangladesh-covid19-data-may-21-2020/covid-19-WorldData.csv', parse_dates=["date"])

continents = ['United States','Italy','India', 'Russia', 'Brazil']

dataFromCsv = dataFromCsv[dataFromCsv.location.isin(continents)]

dataFromCsv = dataFromCsv.sort_values(['date'], ascending=True)

dataFromCsv = dataFromCsv[25:]



dataFromCsvUS = dataFromCsv[dataFromCsv['location'] == 'United States']

dataFromCsvUS = dataFromCsvUS[21:]



dataFromCsvItaly = dataFromCsv[dataFromCsv['location'] == 'Italy']

dataFromCsvItaly = dataFromCsvItaly[31:]



dataFromCsvIndia = dataFromCsv[dataFromCsv['location'] == 'India']

dataFromCsvIndia = dataFromCsvIndia[30:]



dataFromCsvRussia = dataFromCsv[dataFromCsv['location'] == 'Russia']

dataFromCsvRussia = dataFromCsvRussia[30:]



dataFromCsvBrazil = dataFromCsv[dataFromCsv['location'] == 'Brazil']

dataFromCsvBrazil = dataFromCsvBrazil[30:]
annotations = []

fig = go.Figure()



numOfDays = [i for i in range(1, len(dataFromCsvUS.total_cases)+1)]

fig.add_trace(go.Scatter(x=numOfDays, y=dataFromCsvUS.total_cases,

                    mode='lines',

                    name='US'))

annotations.append(dict(x=numOfDays[-1], y=dataFromCsvUS['total_cases'].iloc[-1],

                      xanchor='left', yanchor='top',

                      text=' {}'.format(dataFromCsvUS['total_cases'].iloc[-1]),

                      font=dict(family='Arial',

                                size=18,

                                color='#4169E1'),

                      showarrow=True,

                      borderwidth = 1))



numOfDays = [i for i in range(1, len(dataFromCsvItaly.total_cases)+1)]

fig.add_trace(go.Scatter(x=numOfDays, y=dataFromCsvItaly.total_cases,

                    mode='lines',

                    name='Italy'))

annotations.append(dict(x=numOfDays[-1], y=dataFromCsvItaly['total_cases'].iloc[-1],

                      xanchor='left', yanchor='top',

                      text=' {}'.format(dataFromCsvItaly['total_cases'].iloc[-1]),

                      font=dict(family='Arial',

                                size=18,

                                color='#FFA500'),

                      showarrow=True,

                      borderwidth = 1))



numOfDays = [i for i in range(1, len(accumulated_count_bd)+1)]

fig.add_trace(go.Scatter(x=numOfDays, y=accumulated_count_bd.Case,

                    mode='lines', 

                    name='Bangladesh'))

annotations.append(dict(x=numOfDays[-1], y=accumulated_count_bd['Case'].iloc[-1],

                      xanchor='left', yanchor='top',

                      text=' {}'.format(accumulated_count_bd['Case'].iloc[-1]),

                      font=dict(family='Arial',

                                size=18,

                                color='#228B22'),

                      showarrow=True,

                      borderwidth = 1))



numOfDays = [i for i in range(1, len(dataFromCsvIndia.total_cases)+1)]

fig.add_trace(go.Scatter(x=numOfDays, y=dataFromCsvIndia.total_cases,

                    mode='lines',

                    name='India'))

annotations.append(dict(x=numOfDays[-1], y=dataFromCsvIndia['total_cases'].iloc[-1],

                      xanchor='left', yanchor='top',

                      text=' {}'.format(dataFromCsvIndia['total_cases'].iloc[-1]),

                      font=dict(family='Arial',

                                size=18,

                                color='#8B0000'),

                      showarrow=True,

                      borderwidth = 1))



numOfDays = [i for i in range(1, len(dataFromCsvRussia.total_cases)+1)]

fig.add_trace(go.Scatter(x=numOfDays, y=dataFromCsvRussia.total_cases,

                    mode='lines',

                    name='Russia'))

annotations.append(dict(x=numOfDays[-1], y=dataFromCsvRussia['total_cases'].iloc[-1],

                      xanchor='left', yanchor='top',

                      text=' {}'.format(dataFromCsvRussia['total_cases'].iloc[-1]),

                      font=dict(family='Arial',

                                size=18,

                                color='#8B0000'),

                      showarrow=True,

                      borderwidth = 1))



numOfDays = [i for i in range(1, len(dataFromCsvBrazil.total_cases)+1)]

fig.add_trace(go.Scatter(x=numOfDays, y=dataFromCsvBrazil.total_cases,

                    mode='lines',

                    name='Brazil'))

annotations.append(dict(x=numOfDays[-1], y=dataFromCsvBrazil['total_cases'].iloc[-1],

                      xanchor='left', yanchor='top',

                      text=' {}'.format(dataFromCsvBrazil['total_cases'].iloc[-1]),

                      font=dict(family='Arial',

                                size=18,

                                color='#8B0000'),

                      showarrow=True,

                      borderwidth = 1))



fig.update_layout(title="Positive Cases Trend",

                  xaxis_title='No of Days',

                  yaxis_title='Number of Cases',

                  template="presentation",

                  annotations=annotations)



fig.show()
annotations = []

numOfDays = [i for i in range(1, len(dataFromCsvUS.total_cases)+1)]

fig = go.Figure()

fig.add_trace(go.Scatter(x=numOfDays, y=dataFromCsvUS.total_deaths,

                    mode='lines',

                    name='US'))

annotations.append(dict(x=numOfDays[-1], y=dataFromCsvUS['total_deaths'].iloc[-1],

                      xanchor='left', yanchor='top',

                      text=' {}'.format(dataFromCsvUS['total_deaths'].iloc[-1]),

                      font=dict(family='Arial',

                                size=18,

                                color='#4169E1'),

                      showarrow=True,

                      borderwidth = 1))



numOfDays = [i for i in range(1, len(dataFromCsvItaly.total_cases)+1)]

fig.add_trace(go.Scatter(x=numOfDays, y=dataFromCsvItaly.total_deaths,

                    mode='lines',

                    name='Italy'))

annotations.append(dict(x=numOfDays[-1], y=dataFromCsvItaly['total_deaths'].iloc[-1],

                      xanchor='left', yanchor='top',

                      text=' {}'.format(dataFromCsvItaly['total_deaths'].iloc[-1]),

                      font=dict(family='Arial',

                                size=18,

                                color='#FFA500'),

                      showarrow=True,

                      borderwidth = 1))

numOfDays = [i for i in range(1, len(accumulated_count_bd)+1)]

fig.add_trace(go.Scatter(x=numOfDays, y=accumulated_count_bd.Death,

                    mode='lines', 

                    name='Bangladesh'))

annotations.append(dict(x=numOfDays[-1], y=accumulated_count_bd['Death'].iloc[-1],

                      xanchor='left', yanchor='top',

                      text=' {}'.format(accumulated_count_bd['Death'].iloc[-1]),

                      font=dict(family='Arial',

                                size=18,

                                color='#228B22'),

                      showarrow=True,

                      borderwidth = 1))



numOfDays = [i for i in range(1, len(dataFromCsvIndia.total_cases)+1)]

fig.add_trace(go.Scatter(x=numOfDays, y=dataFromCsvIndia.total_deaths,

                    mode='lines',

                    name='India'))

annotations.append(dict(x=numOfDays[-1], y=dataFromCsvIndia['total_deaths'].iloc[-1],

                      xanchor='left', yanchor='top',

                      text=' {}'.format(dataFromCsvIndia['total_deaths'].iloc[-1]),

                      font=dict(family='Arial',

                                size=18,

                                color='#8B0000'),

                      showarrow=True,

                      borderwidth = 1))



numOfDays = [i for i in range(1, len(dataFromCsvRussia.total_cases)+1)]

fig.add_trace(go.Scatter(x=numOfDays, y=dataFromCsvRussia.total_deaths,

                    mode='lines',

                    name='Russia'))

annotations.append(dict(x=numOfDays[-1], y=dataFromCsvRussia['total_deaths'].iloc[-1],

                      xanchor='left', yanchor='top',

                      text=' {}'.format(dataFromCsvRussia['total_deaths'].iloc[-1]),

                      font=dict(family='Arial',

                                size=18,

                                color='#8B0000'),

                      showarrow=True,

                      borderwidth = 1))



numOfDays = [i for i in range(1, len(dataFromCsvBrazil.total_cases)+1)]

fig.add_trace(go.Scatter(x=numOfDays, y=dataFromCsvBrazil.total_deaths,

                    mode='lines',

                    name='Brazil'))

annotations.append(dict(x=numOfDays[-1], y=dataFromCsvBrazil['total_deaths'].iloc[-1],

                      xanchor='left', yanchor='top',

                      text=' {}'.format(dataFromCsvBrazil['total_deaths'].iloc[-1]),

                      font=dict(family='Arial',

                                size=18,

                                color='#8B0000'),

                      showarrow=True,

                      borderwidth = 1))



fig.update_layout(title="Death Cases Trend",

                  xaxis_title='No of Days',

                  yaxis_title='Death Cases',

                  template="presentation",

                  annotations=annotations)



fig.show()
dataFromCsv = pd.read_csv('../input/bangladesh-covid19-data-may-21-2020/covid-19-WorldData.csv', parse_dates=["date"])

# print(dataFromCsv)

dataset = pd.read_csv('../input/bangladesh-covid19-data-may-21-2020/covid-19-PerDayCaseSummaryBD.csv',parse_dates=["Date"], index_col='Date')

accumulated_count_bd = dataset.cumsum()

accumulated_count_bd = accumulated_count_bd.reset_index()



dataFromCsvUS = dataFromCsv[dataFromCsv['location'] == 'United States']

dataFromCsvItaly = dataFromCsv[dataFromCsv['location'] == 'Italy']

dataFromCsvIndia = dataFromCsv[dataFromCsv['location'] == 'India']

dataFromCsvPakistan = dataFromCsv[dataFromCsv['location'] == 'Pakistan']

dataFromCsvRussia = dataFromCsv[dataFromCsv['location'] == 'Russia']

# print(dataFromCsvIndia)



dataFromCsvItaly[['total_tests']] = dataFromCsvItaly[['total_tests']].fillna(method='bfill')

dataFromCsvUS[['total_tests']] = dataFromCsvUS[['total_tests']].fillna(method='bfill')

dataFromCsvIndia[['total_tests']] = dataFromCsvIndia[['total_tests']].fillna(method='bfill')

dataFromCsvPakistan[['total_tests']] = dataFromCsvPakistan[['total_tests']].fillna(method='bfill')

dataFromCsvRussia[['total_tests']] = dataFromCsvRussia[['total_tests']].fillna(method='bfill')



dataFromCsvItaly[['total_tests']]= dataFromCsvItaly[['total_tests']].fillna(method='ffill')

dataFromCsvUS[['total_tests']] = dataFromCsvUS[['total_tests']].fillna(method='ffill')

dataFromCsvIndia[['total_tests']] = dataFromCsvIndia[['total_tests']].fillna(method='ffill')

dataFromCsvPakistan[['total_tests']] = dataFromCsvPakistan[['total_tests']].fillna(method='ffill')

dataFromCsvRussia[['total_tests']] = dataFromCsvRussia[['total_tests']].fillna(method='ffill')





fig = go.Figure()

fig.add_trace(go.Scatter(x=dataFromCsvUS.date, y=dataFromCsvUS.total_tests,

                    mode='lines',

                    name='US'))

fig.add_trace(go.Scatter(x=dataFromCsvItaly.date, y=dataFromCsvItaly.total_tests,

                    mode='lines',

                    name='Italy'))

fig.add_trace(go.Scatter(x=accumulated_count_bd.Date, y=accumulated_count_bd.Tested,

                    mode='lines', 

                    name='Bangladesh'))

fig.add_trace(go.Scatter(x=dataFromCsvIndia.date, y=dataFromCsvIndia.total_tests,

                    mode='lines',

                    name='India'))

fig.add_trace(go.Scatter(x=dataFromCsvPakistan.date, y=dataFromCsvPakistan.total_tests,

                    mode='lines', 

                    name='Pakistan'))

fig.add_trace(go.Scatter(x=dataFromCsvRussia.date, y=dataFromCsvRussia.total_tests,

                    mode='lines', 

                    name='Russia'))

fig.update_layout(title='Number of Tests',

                   xaxis_title='Date',

                   yaxis_title='Test Number',

                   template="presentation")



fig.show()

symptoms={'symptom':['Fever',

        'Dry cough',

        'Fatigue',

        'Sputum production',

        'Shortness of breath',

        'Muscle pain',

        'Sore throat',

        'Headache',

        'Chills',

        'Nausea or vomiting',

        'Nasal congestion',

        'Diarrhoea',

        'Haemoptysis',

        'Conjunctival congestion'],'percentage':[87.9,67.7,38.1,33.4,18.6,14.8,13.9,13.6,11.4,5.0,4.8,3.7,0.9,0.8]}



symptoms=pd.DataFrame(data=symptoms,index=range(14))

# symptoms
fig = px.bar(symptoms[['symptom', 'percentage']].sort_values('percentage', ascending=False), 

             y="percentage", x="symptom", color='symptom', 

             log_y=True, template='ggplot2', title='Symptom of  Coronavirus')

fig.show()
fig = px.pie(symptoms,

             values="percentage",

             names="symptom",

             template="ggplot2")

fig.update_traces(rotation=90, pull=0.05, textinfo="percent+label",)

fig.update_layout(

    title={

        'text': "Symptom of  Coronavirus",

        'y':.99,

        'x':0.5,

        'xanchor': 'center',

        'yanchor': 'top'})

fig.show()
dataset = pd.read_csv('../input/bangladesh-covid19-data-may-21-2020/covid-19-PerDayCaseSummaryBD.csv',parse_dates=["Date"], index_col='Date')

dataset

dataset = dataset.resample('D').first().fillna(0).cumsum()

dataset = dataset[22:]

accumulated_count = dataset.cumsum()

x = np.arange(len(dataset)).reshape(-1, 1)

y = dataset.values

regressor = MLPRegressor(hidden_layer_sizes=[32, 32, 10], max_iter=50000, alpha=0.0005, random_state=26)

regressor.fit(x, y)

test = np.arange(len(dataset)+7).reshape(-1, 1)

pred = regressor.predict(test)

prediction = pred.round().astype(int)

# print(prediction)

import datetime as datetime

import dateutil.parser

week = [dataset.index[0] + timedelta(days=i) for i in range(len(prediction))]

dt_idx = pd.DatetimeIndex(week)

predicted_count = pd.Series(dt_idx,name="Date")

# print(predicted_count)

dataFrmae = pd.DataFrame(prediction, columns = ['Case','Day','Death','Recovered','Tested'])

prediction_value = pd.concat([dataFrmae,predicted_count], axis=1)

prediction_value = prediction_value.drop(['Day'], axis=1)

prediction_value.set_index('Date',inplace = True)

accumulated_count = accumulated_count.drop(['Day'], axis=1)

prediction_value.index = pd.to_datetime(prediction_value.index)

updateDateTime = datetime.datetime.now().strftime('%Y-%m-%d')

prediction_value = prediction_value[prediction_value.index >= dateutil.parser.parse(updateDateTime)]

prediction_value = prediction_value.reset_index()

formatted_df = prediction_value["Date"].dt.strftime("%d-%b-%Y")



fig = go.Figure(data=[go.Table(

    header=dict(values=['<b>Date</b>','<b>Case</b>','<b>Death</b>','<b>Recovered</b>', '<b>Tests Conducted</b>'],

                fill_color='blue',

                align='center',

                line_color='darkslategray',

                font = dict(color = 'White', size = 18)),

    cells=dict(values=[formatted_df.values,prediction_value.Case,prediction_value.Death, prediction_value.Recovered, prediction_value.Tested],

               fill_color='White',

               line_color='darkslategray',

               align='center',

               height=40,

               font = dict(color = 'Black', size = 18)))

])

fig.update_layout(

    title={

        'text': "Next 7 days Predictions",

        'y':.9,

        'x':0.5,

        'xanchor': 'center',

        'yanchor': 'top'})



fig.show()
fig = go.Figure(go.Bar(x=prediction_value.Date, 

                       y=prediction_value.Tested, 

                       name='Test Cases',

                       text=prediction_value.Tested,

                       textposition='auto'))

fig.add_trace(go.Bar(x=prediction_value.Date, 

                     y=prediction_value.Case, 

                     name='Positive Cases',

                     text=prediction_value.Case,

                     textposition='auto',

                     marker_color='indianred'))





fig.update_layout(barmode='stack', xaxis={'categoryorder':'total descending'})

fig.update_layout(

    title={

        'text': "Next 7 days Predictions Total- (Test Cases vs Positive Cases)",

        'y':.9,

        'x':0.5,

        'xanchor': 'center',

        'yanchor': 'top'})

fig.show()

fig = go.Figure(go.Bar(x=prediction_value.Date, 

                       y=prediction_value.Recovered, 

                       name='Recovery Cases',

                       text=prediction_value.Recovered,

                       textposition='auto',

                       marker_color='#FF7F50'))

fig.add_trace(go.Bar(x=prediction_value.Date, 

                     y=prediction_value.Death, 

                     name='Death Cases',

                     text=prediction_value.Death,

                     textposition='auto',

                     marker_color='#DC143C'))





fig.update_layout(barmode='stack', xaxis={'categoryorder':'total descending'})

fig.update_layout(

    title={

        'text': "Next 7 days Predictions Total-(Recovery Cases vs Death Cases)",

        'y':.9,

        'x':0.5,

        'xanchor': 'center',

        'yanchor': 'top'})

fig.show()
dataset = pd.read_csv('../input/bangladesh-covid19-data-may-21-2020/covid-19-PerDayCaseSummaryBD.csv',parse_dates=["Date"], index_col='Date')

dataset

# print(dataset)

dataset = dataset.resample('D').first().fillna(0).cumsum()

# print(dataset)

dataset = dataset[22:]

accumulated_count = dataset.cumsum()

model_case = ARIMA(dataset['Case'].values, order=(2, 2, 1))

fit_model_case = model_case.fit(trend='c', full_output=True, disp=True)

fit_model_case.summary()

forcast_case = fit_model_case.forecast(steps=7)

pred_case = forcast_case[0].tolist()

pred_case = [round(num) for num in pred_case]
dataset = pd.read_csv('../input/bangladesh-covid19-data-may-21-2020/covid-19-PerDayCaseSummaryBD.csv',parse_dates=["Date"], index_col='Date')

dataset

dataset = dataset.resample('D').first().fillna(0).cumsum()

dataset = dataset[22:]

# print(dataset['Death'].values)

model_case = ARIMA(dataset['Death'].values, order=(2, 2, 1))

fit_model_case = model_case.fit(trend='c', full_output=True, disp=True)

fit_model_case.summary()

forcast_case = fit_model_case.forecast(steps=7)

pred_death = forcast_case[0].tolist()

pred_death = [round(num) for num in pred_death]
dataset = pd.read_csv('../input/bangladesh-covid19-data-may-21-2020/covid-19-PerDayCaseSummaryBD.csv',parse_dates=["Date"], index_col='Date')

# print(dataset.columns)

dataset = dataset.resample('D').first().fillna(0).cumsum()

dataset = dataset[22:]

# print(dataset['Recovered'].values)

model_case = ARIMA(dataset['Recovered'].values, order=(2, 2, 1))

fit_model_case = model_case.fit(trend='c', full_output=True, disp=True)

fit_model_case.summary()

forcast_case = fit_model_case.forecast(steps=7)

pred_Recovered = forcast_case[0].tolist()

pred_Recovered = [round(num) for num in pred_Recovered]
dataset = pd.read_csv('../input/bangladesh-covid19-data-may-21-2020/covid-19-PerDayCaseSummaryBD.csv',parse_dates=["Date"], index_col='Date')

# print(dataset.columns)

dataset = dataset.resample('D').first().fillna(0).cumsum()

dataset = dataset[22:]

# dataset = dataset.reset_index()

model_case = ARIMA(dataset['Tested'].values, order=(3, 2, 1))

fit_model_case = model_case.fit(trend='c', full_output=True, disp=True)

fit_model_case.summary()

forcast_case = fit_model_case.forecast(steps=7)

pred_test = forcast_case[0].tolist()

pred_test = [round(num) for num in pred_test]

# pd.DataFrame(pred_test)
data = {

    "Case" : pred_case,

    "Death" : pred_death,

    "Recovered": pred_Recovered,

    "Tested" : pred_test

}

dataFrame = pd.DataFrame(data)

# print(dataFrame)

week = [dataset.index[dataset.shape[0]-1] + timedelta(days=i) for i in range(1,8)]

dt_idx = pd.DatetimeIndex(week)

predicted_count = pd.Series(dt_idx,name="Date")

prediction_value = pd.concat([dataFrame,predicted_count], axis=1)

prediction_value.set_index('Date',inplace = True)

#print(prediction_value)

prediction_value = prediction_value.reset_index()

formatted_df = prediction_value["Date"].dt.strftime("%d-%b-%Y")



fig = go.Figure(data=[go.Table(

    header=dict(values=['<b>Date</b>','<b>Case</b>','<b>Death</b>','<b>Recovered</b>', '<b>Tests Conducted</b>'],

                fill_color='blue',

                align='center',

                line_color='darkslategray',

                font = dict(color = 'White', size = 18)),

    cells=dict(values=[formatted_df.values,prediction_value.Case,prediction_value.Death, prediction_value.Recovered, prediction_value.Tested],

               fill_color='White',

               line_color='darkslategray',

               align='center',

               height=40,

               font = dict(color = 'Black', size = 18)))

])

fig.update_layout(

    title={

        'text': "Next 7 days Predictions",

        'y':.9,

        'x':0.5,

        'xanchor': 'center',

        'yanchor': 'top'})



fig.show()


fig = go.Figure(go.Bar(x=prediction_value.Date, 

                       y=prediction_value.Tested, 

                       name='Test Cases',

                       text=prediction_value.Tested,

                       textposition='auto'))

fig.add_trace(go.Bar(x=prediction_value.Date,

                     y=prediction_value.Case,

                     name='Positive Cases',

                     text=prediction_value.Case,

                     textposition='auto',

                     marker_color='indianred'))



fig.update_layout(barmode='stack', xaxis={'categoryorder':'total descending'})

fig.update_layout(

    title={

        'text': "Next 7 days Predictions Total- (Test Cases vs Positive Cases)",

#         'y':.9,

        'x':0.5,

        'xanchor': 'center',

        'yanchor': 'top'})

fig.show()
fig = go.Figure(go.Bar(x=prediction_value.Date, 

                       y=prediction_value.Recovered, 

                       name='Recovery Cases',

                       text=prediction_value.Recovered,

                       textposition='auto',

                       marker_color='#FF7F50'))

fig.add_trace(go.Bar(x=prediction_value.Date, 

                     y=prediction_value.Death, 

                     name='Death Cases',

                     text=prediction_value.Death,

                     textposition='auto',

                     marker_color='#DC143C'))



#DC143C

#FF7F50

fig.update_layout(barmode='stack', xaxis={'categoryorder':'total descending'})

fig.update_layout(

    title={

        'text': "Next 7 days Predictions Total-(Recovery Cases vs Death Cases)",

        'y':.9,

        'x':0.5,

        'xanchor': 'center',

        'yanchor': 'top'})

fig.show()
df_tc=pd.read_csv('../input/bangladesh-covid19-data-may-21-2020/covid-19-PerDayCaseSummaryBD.csv') 

bd_cases = []

bd_tested=[]

bd_dayno=[]



for rate in df_tc.Case:

    bd_cases.append(rate) #extracting confirm cases as an array



for rate in df_tc.Tested:

    bd_tested.append(rate) #extracting conducted  as an array



for rate in df_tc.Day:

    bd_dayno.append(rate) #extracting conducted  as an array

    

    

bd_cases=bd_cases[20:]

bd_tested=bd_tested[20:]

bd_dayno=bd_dayno[20:]



bd_tested_np=np.array(bd_tested).reshape(-1,1) #converting array as numpy array 



X_train_confirmed, X_test_confirmed, y_train_confirmed, y_test_confirmed = train_test_split(bd_tested_np, bd_cases, test_size=0.20, shuffle=False)

cases_in_future = 5

#future_forcast = np.array([bd_tested.append(bd_tested[-1]+100) for i in range(days_in_future)]).reshape(-1, 1)

future_forcast=bd_tested

# future_forcast.append(3600)

for i in range(cases_in_future):

    future_forcast.append(int(future_forcast[-1])+100)

    

future_forcast=np.array(future_forcast).reshape(-1,1)



bayesian_poly = PolynomialFeatures(degree=1)

bayesian_poly_X_train_confirmed = bayesian_poly.fit_transform(X_train_confirmed)

bayesian_poly_X_test_confirmed = bayesian_poly.fit_transform(X_test_confirmed)

bayesian_poly_future_forcast = bayesian_poly.fit_transform(future_forcast)

type(bayesian_poly_X_test_confirmed)



# bayesian ridge polynomial regression

tol = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2]

alpha_1 = [1e-7, 1e-6, 1e-5, 1e-4, 1e-3]

alpha_2 = [1e-7, 1e-6, 1e-5, 1e-4, 1e-3]

lambda_1 = [1e-7, 1e-6, 1e-5, 1e-4, 1e-3]

lambda_2 = [1e-7, 1e-6, 1e-5, 1e-4, 1e-3]

normalize = [True, False]



bayesian_grid = {'tol': tol, 'alpha_1': alpha_1, 'alpha_2' : alpha_2, 'lambda_1': lambda_1, 'lambda_2' : lambda_2, 

                 'normalize' : normalize}



bayesian = BayesianRidge(fit_intercept=False)

bayesian_search = RandomizedSearchCV(bayesian, bayesian_grid, scoring='neg_mean_squared_error', cv=3, return_train_score=True, n_jobs=-1, n_iter=40, verbose=2)

bayesian_search.fit(bayesian_poly_X_train_confirmed, y_train_confirmed)



bayesian_confirmed = bayesian_search.best_estimator_

test_bayesian_pred = bayesian_confirmed.predict(bayesian_poly_X_test_confirmed)

bayesian_pred = bayesian_confirmed.predict(bayesian_poly_future_forcast)



future_case=[float(v) for v in bayesian_pred]



future_forcast=future_forcast.tolist()

data = {

    "Test_Cases" : future_forcast[-5:],

    "Positive_Cases" : future_case[-5:],

    

}

dataFrame = pd.DataFrame(data)

dataFrame.Positive_Cases=dataFrame.Positive_Cases.astype(int)



fig = go.Figure(data=[go.Table(

    header=dict(values=["<b>If tomorrow's conducted test</b>",'<b>Positive Case will be</b>'],

                fill_color='blue',

                align='center',

                line_color='darkslategray',

                font = dict(color = 'White', size = 18)),

    cells=dict(values=[dataFrame.Test_Cases,dataFrame.Positive_Cases],

               fill_color='White',

               line_color='darkslategray',

               align='center',

               height=40,

               font = dict(color = 'Black', size = 18)))

])

fig.update_layout(

    title={

        'text': "Tomorrow's Positive Case Predictions Against Test Cases",

        'y':.9,

        'x':0.5,

        'xanchor': 'center',

        'yanchor': 'top'})



fig.show()

poly = PolynomialFeatures(degree=1)

poly_X_train_confirmed = poly.fit_transform(X_train_confirmed)

poly_X_test_confirmed = poly.fit_transform(X_test_confirmed)

poly_future_forcast = poly.fit_transform(future_forcast)

bd_cases=np.array(bd_cases).reshape(-1, 1)

poly_bd_confirmed = poly.fit_transform(bd_cases)



linear_model = LinearRegression(normalize=True, fit_intercept=False)

linear_model.fit(poly_X_train_confirmed, y_train_confirmed)

test_linear_pred = linear_model.predict(poly_X_test_confirmed)

linear_pred = linear_model.predict(poly_future_forcast)



future_case=[float(v) for v in linear_pred]



# future_forcast=future_forcast.tolist()

data = {

    "Test_Cases" : future_forcast[-5:],

    "Positive_Cases" : future_case[-5:],

    

}

dataFrame = pd.DataFrame(data)

dataFrame.Positive_Cases=dataFrame.Positive_Cases.astype(int)



fig = go.Figure(data=[go.Table(

    header=dict(values=["<b>If tomorrow's conducted test</b>",'<b>Positive Case will be</b>'],

                fill_color='blue',

                align='center',

                line_color='darkslategray',

                font = dict(color = 'White', size = 18)),

    cells=dict(values=[dataFrame.Test_Cases,dataFrame.Positive_Cases],

               fill_color='White',

               line_color='darkslategray',

               align='center',

               height=40,

               font = dict(color = 'Black', size = 20)))

])



fig.update_layout(

    title={

        'text': "Tomorrow's Positive Case Predictions Against Test Cases",

        'y':.9,

        'x':0.5,

        'xanchor': 'center',

        'yanchor': 'top'})



fig.show()

dataset = pd.read_csv('../input/bangladesh-covid19-data-may-21-2020/covid-19-PerDayCaseSummaryBD.csv', parse_dates = ['Date'])

dataset

testVSCase = dataset[['Tested','Case']].sort_values('Tested')

test = testVSCase.iloc[:,:-1].values

case = testVSCase.iloc[:,1].values

from sklearn.model_selection import train_test_split



test_train,test_test, case_train, case_test = train_test_split(test,case, test_size = 0.2, random_state = 1)

from sklearn.linear_model import LinearRegression

regressor = LinearRegression()

regressor.fit(test_train,case_train)

case_pred = regressor.predict(test_test)

# print(case_pred)

# try:

#     test = int(input("Enter Number of Tests Made: "))

#     case_pred = int(regressor.predict([[test]])[0])

#     if case_pred < 0:

#         case_pred = 0        

#     print("Possible Positive Cases   : {}".format(case_pred))

# except ValueError:

#     print("Invalid Test Number")