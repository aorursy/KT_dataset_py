import math, time, datetime

from datetime import date

import numpy as np, pandas as pd



from fbprophet import Prophet

from sklearn.metrics import mean_absolute_error



from fbprophet.plot import plot_plotly, plot_components_plotly

import matplotlib.pyplot as plt 

import seaborn as sns

import plotly.graph_objects as go

import plotly.express as px
data = pd.read_csv("../input/covid19-morocco/covid19-morocco.csv")

data["Date"] = pd.to_datetime(data['Date'], dayfirst=True)

data = data[data["Date"] <= pd.Timestamp(date.today())]

active = data['Confirmed'] - data['Recovered'] - data['Deaths']

data['Active'] = active

total_Confirmed=data['Confirmed']

# data.info()
data.tail()
cnf, dth, rec, act = '#393e46', '#ff2e63', '#21bf73', '#fe9801' 

temp = data[['Date','Deaths', 'Recovered', 'Active']].tail(1)

temp = temp.melt(id_vars="Date", value_vars=['Active', 'Deaths', 'Recovered'])

fig = px.treemap(temp, path=["variable"], values="value", height=225, 

                 color_discrete_sequence=[rec, act, dth])

fig.data[0].textinfo = 'label+text+value'

fig.show()
line_data = data.groupby('Date').sum().reset_index()



line_data = line_data.melt(id_vars='Date', 

                 value_vars=['Confirmed', 'Recovered', 'Deaths'], 

                 var_name='Ratio', 

                 value_name='Value')



fig = px.line(line_data, x="Date", y="Value", color='Ratio', 

              title='Total Cumulative COVID-19 Cases in Morocco')

fig.show()
line_data = data.groupby('Date').sum().reset_index()



line_data = line_data.melt(id_vars='Date', 

                 value_vars=['Active'], 

                 var_name='Ratio', 

                 value_name='Value')



fig = px.line(line_data, x="Date", y="Value", color='Ratio', 

              title='Total Evolution of COVID-19 Active')

fig.show()
# Treemaps
dictionary_labels = {

       'TTH': 'Tanger-Tétouane-Al Hoceima',

       'OR': 'Oriental',

       'FM': 'Fès-Meknès',

       'RSK': 'Rabat-Salé-Kénitra',

       'BK': 'Béni Mellal-Khénifra',

       'CS': 'Casablanca-Settat',

       'MS': 'Marrakech-Safi',

       'DT': 'Drâa-Tafilalet',

       'SM': 'Souss-Massa',

       'GO': 'Guelmim-Oued Noun',

       'LS': 'Laâyoune-Sakia El Hamra',

       'DO': 'Dakhla-Oued Ed-Dahab'

}



data = data.set_index('Date')

regions = []

for x in dictionary_labels:

    regions.append(data[x][-1])



labels = []

for x in dictionary_labels.values():

    labels.append(x)
plt.figure(figsize=(25,10))

plt.pie(regions, labels = labels)

plt.title('Pie Chart of Total COVID-19 Cases per Region')
line_data = data.groupby('Date').sum().reset_index()

line_data = line_data.melt(id_vars='Date', 

                 value_vars=['TTH', 'OR', 'FM', 'RSK', 'BK', 'CS', 'MS', 'DT', 'SM', 'GO', 'LS', 'DO' ], 

                 var_name='Ratio', 

                 value_name='Value')



fig = px.line(line_data, x="Date", y="Value", color='Ratio', 

              title='Total Evolution of COVID-19 Cases per Region')

fig.show()
labels = list(dictionary_labels.keys())

regions = list(dictionary_labels.values())

plt.figure(figsize=(25,10))

for i in range (12):

       ax1 = plt.subplot(4, 3, i+1)

       sns.lineplot(data = data[labels[i]],label=regions[i])
new_cases = []

# data = data.set_index('Date')



for i in range(len(total_Confirmed)):

    if i == 0:

        new_cases.append(0)

    elif total_Confirmed[i] < total_Confirmed[i-1]:

        new_cases.append(0)

    else:

        temp = int(total_Confirmed[i] - total_Confirmed[i-1])

        new_cases.append(temp)

    

new_cases = np.array(new_cases)

data['New cases'] = new_cases

new_cases = data.groupby('Date').sum()['New cases'].reset_index()

new_cases.columns = ['ds','y']



model = Prophet(interval_width = 0.95)

model.fit(new_cases)



future = model.make_future_dataframe(periods = 5)

future.tail()



forecast = model.predict(future)

forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()
# print("Mean Absolute Error:", mean_absolute_error(forecast['yhat'].head(215).round(), data['New cases']).round())
fig = go.Figure()

fig.add_trace(go.Scatter(x=forecast.ds, y = abs(forecast['yhat'].round()),

                         mode= 'lines+markers',name='Forecasting of Daily Cases'))

fig.add_trace(go.Scatter(x=data.index, y = data['New cases'],

                         mode= 'lines+markers',name='Daily cases'))

fig.update_layout(title="Forecasting of COVID-19 Daily Cases",

                 xaxis_title="Date",yaxis_title="Number of Cases",legend=dict(x=0,y=1,traceorder="normal"))

fig.show()
plot_plotly(model, forecast)