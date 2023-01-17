# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input/covid19-in-india/'):

    for filename in filenames:

        tmpfile = os.path.join(dirname, filename)

        print(tmpfile)

        

        

        



# Any results you write to the current directory are saved as output.
agegroup = pd.read_csv('/kaggle/input/covid19-in-india/AgeGroupDetails.csv')

covid19india = pd.read_csv('/kaggle/input/covid19-in-india/covid_19_india.csv')

hospitalbeds = pd.read_csv('/kaggle/input/covid19-in-india/HospitalBedsIndia.csv')

#icmrdetails = pd.read_csv('/kaggle/input/covid19-in-india/ICMRTestingDetails.csv')

individualdetails = pd.read_csv('/kaggle/input/covid19-in-india/IndividualDetails.csv')

populationdetails = pd.read_csv('/kaggle/input/covid19-in-india/population_india_census2011.csv')





agegroup.head()
import csv

json_total_data = pd.read_json('/kaggle/input/covidind/all_totals.json')

#json_total_data.to_csv('all_totals.csv')

#csv_total_data = pd.read_csv('/kaggle/working/all_totals.csv')

report_times = []

#print(csv_total_data['rows'])

tmp_active = []

for i in range(len(json_total_data['rows'])):

    report_times.append(json_total_data['rows'][i]['key'][0])

    

row = []

with open('tmp_csv.csv', 'w') as f:

    csvwriter = csv.writer(f)

    csvwriter.writerow(['report_time', 'type', 'number'])

    for i in range(len(json_total_data['rows'])):

        row = []

        row.append(json_total_data['rows'][i]['key'][0])

        row.append(json_total_data['rows'][i]['key'][1])

        row.append(json_total_data['rows'][i]['value'])

        csvwriter.writerow(row)



csv_total_data = pd.read_csv('/kaggle/working/tmp_csv.csv') 

#print(csv_total_data)



#report_times

tmp_report_time = set(report_times)

#tmp_report_time

    









import json

totalcases = csv_total_data



#totalcases.tail(30)



unique_report_time = csv_total_data['report_time'].unique()



active_cases = []

cured_cases = []

death_cases = []

total_confirmed_cases = []



#print(unique_report_time)

for report_time in unique_report_time:

    #print(report_time)

    tmpentry = totalcases[totalcases['report_time'] == report_time]

    #print(tmpentry)

    active_cases.append(tmpentry.loc[tmpentry['type'] == 'active_cases', 'number'].values[0])

    cured_cases.append(tmpentry.loc[tmpentry['type'] == 'cured', 'number'].values[0])

    death_cases.append(tmpentry.loc[tmpentry['type'] == 'death', 'number'].values[0])

    total_confirmed_cases.append(tmpentry.loc[tmpentry['type'] == 'total_confirmed_cases', 'number'].values[0])







report_time = unique_report_time.tolist()

report_time = list(map(str, report_time))

#print(report_time)

#print(active_cases)

#unique_report_time
d = {'Date':report_time, 'Confirmed':total_confirmed_cases}

df = pd.DataFrame(d)

confirmed = df.groupby('Date').sum()['Confirmed'].reset_index()



confirmed.columns = ['ds','y']

confirmed['ds'] = pd.to_datetime(confirmed['ds'])

confirmed['ds'] = confirmed['ds'].dt.tz_localize(None)

plot_cases = confirmed.diff(axis=0)

plot_cases = plot_cases[1:]

days = confirmed['ds']

days = days[1:]

cases = plot_cases['y']

d = {'Date':days, 'Confirmed':cases}

testdf = pd.DataFrame(d)

testdf['Date'] = pd.to_numeric(testdf['Date'], errors='coerce')

df = testdf.to_numpy()

# print(days)

# print(cases)



import plotly.graph_objs as go



trace_new_cases = go.Scatter(x=days, y=cases,

                    mode='lines+markers',

                    name='new cases',

                    marker=dict(color='Orange'))



data = [trace_new_cases]



layout = dict(title='Covid-19 new cases in India')



fig = go.Figure(data=data, layout=layout)



fig.show()
d = {'Date':report_time, 'Confirmed_death':death_cases}

df = pd.DataFrame(d)

confirmed = df.groupby('Date').sum()['Confirmed_death'].reset_index()



confirmed.columns = ['ds','y']

confirmed['ds'] = pd.to_datetime(confirmed['ds'])

confirmed['ds'] = confirmed['ds'].dt.tz_localize(None)

plot_cases = confirmed.diff(axis=0)

plot_cases = plot_cases[1:]

days = confirmed['ds']

days = days[1:]

cases = plot_cases['y']

d = {'Date':days, 'Confirmed_death':cases}

testdf = pd.DataFrame(d)

testdf['Date'] = pd.to_numeric(testdf['Date'], errors='coerce')

df = testdf.to_numpy()

# print(days)

# print(cases)



import plotly.graph_objs as go



trace_new_cases = go.Scatter(x=days, y=cases,

                    mode='lines+markers',

                    name='new deaths',

                    marker=dict(color='Orange'))



data = [trace_new_cases]



layout = dict(title='Covid-19 new Deaths in India')



fig = go.Figure(data=data, layout=layout)



fig.show()
from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score

Xtrain, Xtest, ytrain, ytest = train_test_split(df, cases)

from sklearn.naive_bayes import GaussianNB # 1. choose model class

model = GaussianNB()                       # 2. instantiate model

model.fit(Xtrain, ytrain)                  # 3. fit model to data

y_model = model.predict(Xtest)



accuracy_score(ytest, y_model)
import plotly.graph_objs as go



trace_active_cases = go.Scatter(x=report_time, y=active_cases,

                    mode='lines+markers',

                    name='active cases',

                    marker=dict(color='Orange'))

                                

trace_cured_cases = go.Scatter(x=report_time, y=cured_cases,

                    mode='lines+markers',

                    name='cured cases',

                    marker=dict(color='Green'))



trace_death_cases = go.Scatter(x=report_time, y=death_cases,

                    mode='lines+markers',

                    name='death cases',

                    marker=dict(color='Red'))



trace_total_confirmed_cases = go.Scatter(x=report_time, y=total_confirmed_cases,

                    mode='lines+markers',

                    name='total confirmed cases',

                    marker=dict(color='Blue'))

                               

data = [trace_active_cases, trace_cured_cases, trace_death_cases, trace_total_confirmed_cases]



layout = dict(title='Covid-19 cases in India')



fig = go.Figure(data=data, layout=layout)



fig.show()



#TRIAL prediction, will not reconcile with actual values.



from fbprophet import Prophet



import plotly.graph_objs as go



trace_total_confirmed_cases = go.Scatter(x=report_time, y=total_confirmed_cases,

                    mode='lines+markers',

                    name='total confirmed cases',

                    marker=dict(color='Blue'))

                               

data = [trace_total_confirmed_cases]



layout = dict(title='Covid-19 cases in India')



fig = go.Figure(data=data, layout=layout)



#fig.show()



#print(report_time)

d = {'Date':report_time, 'Confirmed':total_confirmed_cases}

df = pd.DataFrame(d)

confirmed = df.groupby('Date').sum()['Confirmed'].reset_index()

confirmed.columns = ['ds','y']

confirmed['ds'] = pd.to_datetime(confirmed['ds'])

confirmed['ds'] = confirmed['ds'].dt.tz_localize(None)

#print(confirmed.tail(10))



m = Prophet(interval_width=0.90)

m.fit(confirmed)

future = m.make_future_dataframe(periods=10)

future.tail()



#predicting the future with date, and upper and lower limit of y value

forecast = m.predict(future)

print(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(15))



confirmed_forecast_plot = m.plot(forecast)



import csv

json_total_data = pd.read_json('/kaggle/input/covidind/mohfw.json')



row = []

with open('tmp_mohfw_csv.csv', 'w') as f:

    csvwriter = csv.writer(f)

    csvwriter.writerow(['report_time', 'state', 'confirmed'])

    for i in range(len(json_total_data['rows'])):

        row = []

        row.append(json_total_data['rows'][i]['value']['report_time'])

        row.append(json_total_data['rows'][i]['value']['state'])

        row.append(json_total_data['rows'][i]['value']['confirmed'])

        csvwriter.writerow(row)



csv_state_info = pd.read_csv('/kaggle/working/tmp_mohfw_csv.csv')



#print(csv_state_info)



state_data_dict = {}

state = csv_state_info['state'].unique()

#print(state)



for st in state:

    state_data_dict[str(st)]= dict([('report_time', []), ('confirmed', [])])



#(state_data_dict['kl']['confirmed']).append(10)



for col, val in csv_state_info.iterrows():

    #print(val)

    #print(val.values[1])

    (state_data_dict[str(val.values[1])]['report_time']).append(str(val.values[0]))

    (state_data_dict[str(val.values[1])]['confirmed']).append(str(val.values[2]))

        

    





#(state_data_dict['kl']['confirmed']).append(10)

#print(state_data_dict)







# for i in range(len(json_total_data['rows'])):

# #     print(json_total_data['rows'][i])

# #     if i > 20:

# #     break

#     mohfw_report_times.append(json_total_data['rows'][i]['value']['report_time'])

#     state.append(json_total_data['rows'][i]['value']['state'])

#     confirmed_cases.append(json_total_data['rows'][i]['value']['confirmed'])



# tmp_state = state

# tmp_state = set(tmp_state)

# tmp_state

# # report_times = []

# #print(csv_total_data['rows'])

# tmp_active = []

# for i in range(len(json_total_data['rows'])):

#     report_times.append(json_total_data['rows'][i]['key'][0])

    

# row = []

# with open('tmp_csv.csv', 'w') as f:

#     csvwriter = csv.writer(f)

#     csvwriter.writerow(['report_time', 'type', 'number'])

#     for i in range(len(json_total_data['rows'])):

#         row = []

#         row.append(json_total_data['rows'][i]['key'][0])

#         row.append(json_total_data['rows'][i]['key'][1])

#         row.append(json_total_data['rows'][i]['value'])

#         csvwriter.writerow(row)



# csv_total_data = pd.read_csv('/kaggle/working/tmp_csv.csv') 

# #print(csv_total_data)



# #report_times

# tmp_report_time = set(report_times)

# #tmp_report_time
import plotly.graph_objs as go

N = len(state)

values = list(range(N))

trace_confirmed_cases = []

c= ['hsl('+str(h)+',50%'+',50%)' for h in np.linspace(0, 360, N)]

#print(N)

i = 0

for st in state:    

    trace_confirmed_cases.append(go.Scatter(x=state_data_dict[st]['report_time'], y=state_data_dict[st]['confirmed'],

                        mode='lines+markers',

                        name=st,

                        marker=dict(color=c[i])))

    i = i+1

                               

data =  trace_confirmed_cases



layout = dict(title='Covid-19 confirmed cases in India - state wise categorisation')



fig = go.Figure(data=data, layout=layout)



fig.show()