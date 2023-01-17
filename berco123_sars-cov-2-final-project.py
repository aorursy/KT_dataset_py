# This code compares the death people data in Turkey for SARS- Cov- 2 viruses

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import plotly.graph_objects as go #it uses for the graph of the data 
from plotly.offline import init_notebook_mode, iplot
import plotly.express as px
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
        

        import warnings
warnings.filterwarnings('ignore')   


        

province_data = pd.read_csv("/kaggle/input/number-of-cases-in-the-city-covid19-turkey/number_of_cases_in_the_city.csv")
data = pd.read_csv("/kaggle/input/covid19-in-turkey/covid_19_data_tr.csv")
data_test_numbers = pd.read_csv("/kaggle/input/covid19-in-turkey/test_numbers.csv")
data_intensive_care = pd.read_csv("/kaggle/input/covid19-in-turkey/time_series_covid_19_intensive_care_tr.csv")
data_intubated = pd.read_csv("/kaggle/input/covid19-in-turkey/time_series_covid_19_intubated_tr.csv")
data.tail()
data.info()
#run the section below once if you run it twice it will fail because it cannot find anything as province state

data.rename(columns = {"Last_Update" : "Date", "Confirmed" : "Case", "Deaths" : "Demise", "Recovered" : "Convalescent"}, inplace = True)
data.drop("Province/State", axis = 1, inplace = True)
list1 = data_test_numbers.keys()
list2 = []
for i in range (4,len(list1)):
    list2.append(data_test_numbers[list1[i]][0])

data['number_of_tests'] = list2 #Days without number of tests will be considered as 0
list3 = data_intensive_care.keys()
list4 = []
for i in range (4,len(list3)):
    list4.append(data_test_numbers[list3[i]][0])
data['totalIntensiveCare'] = list4 #the number of patients in intensive care unit will be considered as 0, unspecified days
list5 = data_intubated.keys()
list6 = []
for i in range (4,len(list5)):
    list6.append(data_test_numbers[list5[i]][0])

data['totalnumberofIntubates'] = list6 #days without intubation number will be considered as 0
test_rate = [0] 
test_increase = [0] 
case_rate = [0] #we accepted 0 because there is no specified value
death_rate = [0]
case_increase = [0]
death_increase = [0]
test_rate = [0]
test_increase= [0]
intensivecare_rate = [0]
intensivecare_increase = [0]
intubates_rate = [0]
intubates_increase = [0]
intensivecare_case_rate = [0]
intubates_case_rate = [0]
active_case = [0]

for i in range(len(data)-1):
    testRate =  round((data["number_of_tests"][i+1] - data["number_of_tests"][i]) / data["number_of_tests"][i], 2)
    testIncrease = data["number_of_tests"][i+1] - data["number_of_tests"][i] 
    caseRate = round((data["Case"][i+1] - data["Case"][i]) / data["Case"][i], 2)
    caseIncrease = data["Case"][i+1] - data["Case"][i]
    deathRate =  round((data["Demise"][i+1] - data["Demise"][i]) / data["Demise"][i], 2)
    deathIncrease = data["Demise"][i+1] - data["Demise"][i]
    intensivecareRate =  round((data["totalIntensiveCare"][i+1] - data["totalIntensiveCare"][i]) / data["totalIntensiveCare"][i], 2)
    intensivecareIncrease = data["totalIntensiveCare"][i+1] - data["totalIntensiveCare"][i] 
    intubatesRate =  round((data["totalnumberofIntubates"][i+1] - data["totalnumberofIntubates"][i]) / data["totalnumberofIntubates"][i], 2)
    intubatesIncrease = data["totalnumberofIntubates"][i+1] - data["totalnumberofIntubates"][i]
    intensivecarecaseRate = round((data["totalIntensiveCare"][i+1] - data["totalIntensiveCare"][i]) / data["Case"][i], 5)
    intubatescaseRate = round((data["totalnumberofIntubates"][i+1] - data["totalnumberofIntubates"][i]) / data["Case"][i], 5)
    activeCase = data["Case"][i] - data["Convalescent"][i]
    
    
    test_rate.append(testRate)
    test_increase.append(testIncrease)
    case_rate.append(caseRate)
    death_rate.append(deathRate)
    case_increase.append(caseIncrease)
    death_increase.append(deathIncrease)
    intensivecare_rate.append(intensivecareRate)
    intensivecare_increase.append(intensivecareIncrease)
    intubates_rate.append(intubatesRate)
    intubates_increase.append(intubatesIncrease)
    intensivecare_case_rate.append(intensivecarecaseRate)
    intubates_case_rate.append(intubatescaseRate)
    active_case.append(activeCase)

    
data["Test Increase"] = test_increase
data["Test Increase Rate"] = test_rate
data["Case Increase"] = case_increase
data["Case Increase Rate"] = case_rate
data["Death Increase"] = death_increase
data["Death Increase Rate"] = death_rate
data["Intensivecare Increase"] = intensivecare_increase
data["Intensivecare Increase Rate"] = intensivecare_rate
data["Intubates Increase"] = intubates_increase
data["Intubates Increase Rate"] = intubates_rate
data["Intensivecare/Case Rate"] = intensivecare_case_rate
data["Intubates/Case Rate"] = intubates_case_rate
data['Active Case'] = active_case

data.fillna(0, inplace = True) #We updated NaN values to 0.
data = data.replace([np.inf, -np.inf], np.nan) #number/0 infinity updated 0.
test_positive=[]
for i in range(len(data)):
    test_positive_rate = round((data["Case Increase Rate"][i] / data["number_of_tests"][i]), 2)
    test_positive.append(test_positive_rate)
    
data["Positive/Test Rate"] = test_positive
data = data.replace([np.inf, -np.inf], np.nan)
data.fillna(0, inplace = True)
data
fig = go.Figure(data=[
    go.Bar(name='Case', x=data['Date'], y=data['Case'], marker_color='rgba(135, 206, 250, 0.8)'),
    go.Bar(name='Demise', x=data['Date'], y = data['Demise'], marker_color='rgba(255, 0, 0, 0.8)'),
    go.Bar(name='Convalescent', x=data['Date'], y=data['Convalescent'], marker_color='rgba(0, 255, 0, 0.8)')
])
fig.update_layout(barmode='group', title_text='Turkey Daily Case, Death and Convalescent Number', xaxis_tickangle=-45)
fig.show()
fig = go.Figure(data=[
    go.Bar(name='Demise', x=data['Date'], y = data['Demise'], marker_color='rgb(255, 0, 0)'),
    go.Bar(name='Convalescent', x=data['Date'] , y=data['Convalescent'], marker_color='rgb(0, 255, 0)')
])
fig.update_layout(barmode='group', title_text='Turkey Daily Death and Convalescent Number', xaxis_tickangle=-45)
fig.show()
fig = go.Figure(data=[go.Bar(name='Active Case', x=data['Date'], y=data['Active Case'], marker_color='rgba(135, 206, 250, 0.8)'),])
fig.update_layout(barmode='group', title_text='Turkey Daily Active Case Number', xaxis_tickangle=-45, xaxis= dict(title= 'Date / Date'),
                  yaxis= dict(title= 'Number of Person'))
fig.show()
after17thofmarch = data.loc[data['Date'] > '3/17/2020']
after15thofmarch = data.loc[data['Date'] > '3/15/2020']
after26thofmarch = data.loc[data['Date'] > '3/27/2020']
case = go.Scatter(x = data.Date,
                    y = data.Case,
                    mode = "lines+markers",
                    name = "Case / Cases",
                    marker = dict(color = 'rgba(135, 206, 250, 0.8)'),
                    text= data.Case
                   )
death = go.Scatter(x = data.Date,
                    y = data.Demise,
                    mode = "lines+markers",
                    name = "Demise / Death",
                    marker = dict(color = 'rgba(255, 0, 0, 0.8)'),
                    text= data.Demise
                   )
convalescent = go.Scatter(x = data.Date,
                    y = data.Convalescent,
                    mode = "lines+markers",
                    name = "Convalescent/Recovered",
                    marker = dict(color = 'rgba(0, 255, 0, 0.8)'),
                    text= data.Convalescent
                   )
data2 = [case, death, convalescent]
layout = dict(title = "Turkey SARS-Cov-2 Case, Demise and Convalescent -  Covid-19 Number of Case and Deaths in Turkey", 
              xaxis= dict(title= 'Date / Date'), yaxis= dict(title= 'Number of Person'), xaxis_tickangle=-45)
fig = dict(data = data2, layout = layout)
iplot(fig)
death = go.Scatter(x = data.Date,
                    y = data.Demise,
                    mode = "lines+markers",
                    name = "Demise / Death",
                    marker = dict(color = 'rgba(255, 0, 0, 0.8)'),
                    text= data.Demise
                   )
convalescent = go.Scatter(x = data.Date,
                    y = data.Convalescent,
                    mode = "lines+markers",
                    name = "Convalescent/Recovered",
                    marker = dict(color = 'rgba(0, 255, 0, 0.8)'),
                    text= data.Convalescent
                   )
data2 = [death,convalescent]
layout = dict(title = "SARS- Cov- 2 Number of Deaths and Recovered in Turkey", 
              xaxis= dict(title= 'Date / Date'), yaxis= dict(title= 'Number of Person'), xaxis_tickangle=-45)
fig = dict(data = data2, layout = layout)
iplot(fig)
case = go.Scatter(x =after15thofmarch.Date,
                    y = after15thofmarch.Case,
                    mode = "lines+markers",
                    name = "Case / Cases",
                    marker = dict(color = 'rgba(135, 206, 250, 0.8)'),
                    text= after15thofmarch.Case
                   )
layout = dict(title = "SARS- Cov- 2 Number of Case in Turkey (Logarithmic)", 
              xaxis= dict(title= 'Date / Date'), yaxis= dict(title= 'Number of Person'), xaxis_tickangle=-45, yaxis_type="log")
fig = dict(data = case, layout = layout)
iplot(fig)
demise = go.Scatter(x =after17thofmarch.Date,
                    y = after17thofmarch.Demise,
                    mode = "lines+markers",
                    name = "Demise / Death",
                    marker = dict(color = 'rgba(255, 0, 0, 0.8)'),
                    text= after17thofmarch.Demise
                   )
layout = dict(title = "SARS- Cov- 2 Number of Death in Turkey (Logarithmic)", 
              xaxis= dict(title= 'Date / Date'), yaxis= dict(title= 'Number of Person'), xaxis_tickangle=-45, yaxis_type="log")
fig = dict(data = demise, layout = layout)
iplot(fig)
intensivecare = go.Scatter(x =after26thofmarch.Date,
                    y = after26thofmarch.totalIntensiveCare,
                    mode = "lines+markers",
                    name = "totalIntensivecare",
                    marker = dict(color = 'rgba(72,118 ,255, 0.8)'),
                    text= after26thofmarch.totalIntensiveCare
                   )
intubates = go.Scatter(x =after26thofmarch.Date,
                    y = after26thofmarch.totalnumberofIntubates,
                    mode = "lines+markers",
                    name = "totalIntubates",
                    marker = dict(color = 'rgba(139, 90, 43, 0.8)'),
                    text= after26thofmarch.totalnumberofIntubates
                   )
data2 = [intensivecare,intubates]
layout = dict(title = "SARS- Cov- 2 Intensivecare and Intubates Patient Number in Turkey ", 
              xaxis= dict(title= 'Date / Date'), yaxis= dict(title= 'Number of Person'), xaxis_tickangle=-45)
fig = dict(data = data2, layout = layout)
iplot(fig)

intensivecare = go.Scatter(x =after26thofmarch.Date,
                    y = after26thofmarch.totalIntensiveCare,
                    mode = "lines+markers",
                    name = "totalIntensivecare",
                    marker = dict(color = 'rgba(72,118 ,255, 0.8)'),
                    text= after26thofmarch.totalIntensiveCare
                   )
intubates = go.Scatter(x =after26thofmarch.Date,
                    y = after26thofmarch.totalnumberofIntubates,
                    mode = "lines+markers",
                    name = "totalIntubates",
                    marker = dict(color = 'rgba(139, 90, 43, 0.8)'),
                    text= after26thofmarch.totalnumberofIntubates
                   )
data2 = [intensivecare,intubates]
layout = dict(title = "SARS- Cov- 2 Intensivecare and Intubates Patient Number in Turkey ", 
              xaxis= dict(title= 'Date / Date'), yaxis= dict(title= 'Number of Person'), xaxis_tickangle=-45)
fig = dict(data = data2, layout = layout)
iplot(fig)
intensivecare = go.Scatter(x =after26thofmarch.Date,
                    y = after26thofmarch.totalIntensiveCare,
                    mode = "lines+markers",
                    name = "totalIntensivecare",
                    marker = dict(color = 'rgba(72,118 ,255, 0.8)'),
                    text= after26thofmarch.totalIntensiveCare
                   )
intubates = go.Scatter(x =after26thofmarch.Date,
                    y = after26thofmarch.totalnumberofIntubates,
                    mode = "lines+markers",
                    name = "totalIntubates",
                    marker = dict(color = 'rgba(139, 90, 43, 0.8)'),
                    text= after26thofmarch.totalnumberofIntubates
                   )
data2 = [intensivecare,intubates]
layout = dict(title = "SARS- Cov- 2 Intensivecare and Intubates Patient Number in Turkey ", 
              xaxis= dict(title= 'Date / Date'), yaxis= dict(title= 'Number of Person'), xaxis_tickangle=-45)
fig = dict(data = data2, layout = layout)
iplot(fig)
intensivecare = go.Scatter(x =after26thofmarch.Date,
                    y = after26thofmarch.totalIntensiveCare,
                    mode = "lines+markers",
                    name = "totalIntensivecare",
                    marker = dict(color = 'rgba(72,118 ,255, 0.8)'),
                    text= after26thofmarch.totalIntensiveCare
                   )
intubates = go.Scatter(x =after26thofmarch.Date,
                    y = after26thofmarch.totalnumberofIntubates,
                    mode = "lines+markers",
                    name = "totalIntubates",
                    marker = dict(color = 'rgba(139, 90, 43, 0.8)'),
                    text= after26thofmarch.totalnumberofIntubates
                   )
data2 = [intensivecare,intubates]
layout = dict(title = "SARS- Cov- 2 Intensivecare and Intubates Patient Number in Turkey ", 
              xaxis= dict(title= 'Date / Date'), yaxis= dict(title= 'Number of Person'), xaxis_tickangle=-45)
fig = dict(data = data2, layout = layout)
iplot(fig)
case_rate_ = go.Scatter(x = data.Date,
                    y = data['Case Increase Rate'],
                    mode = "lines+markers",
                    name = "Case Increase Rate",
                    marker = dict(color = 'rgba(135, 206, 250, 0.8)'),
                    text= data['Case Increase Rate']
                   )
death_rate_ = go.Scatter(x = data.Date,
                    y = data['Death Increase Rate'],
                    mode = "lines+markers",
                    name = "Death Increase Rate",
                    marker = dict(color = 'rgba(255, 0, 0, 0.8)'),
                    text= data['Death Increase Rate']
                   )
positive_test_rate = go.Scatter(x = data.Date,
                    y = data['Positive/Test Rate'],
                    mode = "lines+markers",
                    name = "Positive/Test Rate",
                    marker = dict(color = 'rgba(220, 218, 68, 0.8)'),
                    text= data['Positive/Test Rate']
                   )
data2 = [case_rate_, death_rate_, positive_test_rate]
layout = dict(title = "SARS- Cov- 2 Daily Rates in Turkey", 
              xaxis= dict(title= 'Date / Date'), yaxis= dict(title= 'Rate / Rate'), xaxis_tickangle=-45)
fig = dict(data = data2, layout = layout)
iplot(fig)
case_rate_ = go.Scatter(x = data.Date,
                    y = data['Case Increase Rate'],
                    mode = "lines+markers",
                    name = "Case Increase Rate",
                    marker = dict(color = 'rgba(135, 206, 250, 0.8)'),
                    text= data['Case Increase Rate']
                   )
layout = dict(title = "SARS- Cov- 2 Turkey Daily Case Increase Rate", 
              xaxis= dict(title= 'Date / Date'), yaxis= dict(title= 'Rate / Rate'), xaxis_tickangle=-45)
fig = dict(data = case_rate_, layout = layout)
iplot(fig)
death_rate_ = go.Scatter(x = data.Date,
                    y = data['Death Increase Rate'],
                    mode = "lines+markers",
                    name = "Death Increase Rate",
                    marker = dict(color = 'rgba(255, 0, 0, 0.8)'),
                    text= data['Death Increase Rate']
                   )
layout = dict(title = "SARS- Cov- 2 Turkey Daily Death Increase Rate", 
              xaxis= dict(title= 'Date / Date'), yaxis= dict(title= 'Rate / Rate'), xaxis_tickangle=-45)
fig = dict(data = death_rate_, layout = layout)
iplot(fig)
test = go.Scatter(x = data.Date,
                    y = data['number_of_tests'],
                    mode = "lines+markers",
                    name = "Test Number",
                    marker = dict(color = 'rgba(220, 218, 68, 0.8)'),
                    text= data['number_of_tests']
                   )
layout = dict(title = "SARS- Cov- 2 Turkey Daily Test Number", 
              xaxis= dict(title= 'Date / Date'), yaxis= dict(title= 'Number'), xaxis_tickangle=-45)
fig = dict(data = test, layout = layout)
iplot(fig)
positive_test_rate = go.Scatter(x = data.Date,
                    y = data['Positive/Test Rate'],
                    mode = "lines+markers",
                    name = "Positive/Test Rate",
                    marker = dict(color = 'rgba(220, 218, 68, 0.8)'),
                    text= data['Positive/Test Rate']
                   )
layout = dict(title = "SARS- Cov- ' Turkey Daily Positive/Test Rate'", 
              xaxis= dict(title= 'Date / Date'), yaxis= dict(title= 'Rate / Rate'), xaxis_tickangle=-45)
fig = dict(data = positive_test_rate, layout = layout)
iplot(fig)
case_increase = go.Scatter(x = data.Date,
                        y = data['Case Increase'],
                    mode = "lines+markers",
                    name = "Case Increase Number",
                    marker = dict(color = 'rgba(135, 206, 250, 0.8)'),
                    text= data['Case Increase']
                   )
test_increase = go.Scatter(x = data.Date,
                        y = data['Test Increase'],
                    mode = "lines+markers",
                    name = "Test Increase Number",
                    marker = dict(color = 'rgba(220, 218, 68, 0.8)'),
                    text= data['Test Increase']
                   )
data2 =[case_increase,test_increase] 
layout = dict(title = "SARS- Cov- 2 Turkey Case Increase and Test Increase Number", 
              xaxis= dict(title= 'Date / Date'), yaxis= dict(title= 'Rate / Rate'), xaxis_tickangle=-45)
fig = dict(data = data2, layout = layout)
iplot(fig)
province_data.head()
province_data.info()
province_data.rename(columns = {"Province" : "City", "Number of Case" : "Number of Case"}, inplace = True)
province_data.sort_values(by=['Number of Case'], ascending=False, inplace = True)
#province_df = province_data.head(10)
fig = px.pie(province_data.head(10), values='Number of Case', names='City', title='Number of Cases in the Citys')
fig.show()
province_df2 = province_data[1:]
fig = px.pie(province_df2, values='Number of Case', names='City', title='Number of Cases of Cities Outside of Istanbul', 
             hover_data=['Number of Case'])
fig.update_traces(textposition='inside', textinfo='percent+label')
fig.show()
fig = px.bar(province_data, x="City", y="Number of Case", title='Number of Cases in the Citys')
fig.update_layout(barmode='group')
fig.show()
fig = px.bar(province_df2, x="City", y="Number of Case", title='Number of Cases of Cities Outside of Istanbul')
fig.update_layout(barmode='group')
fig.show()
fig = px.bar(province_df2.head(15), x="City", y="Number of Case", title='Case Numbers of Top 15 Cities Outside of Istanbul')
fig.update_layout(barmode='group')
fig.show()