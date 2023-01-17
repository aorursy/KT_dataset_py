
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
pd.set_option('display.max_rows', None)
import datetime
from plotly.subplots import make_subplots
from scipy.optimize import curve_fit
import pylab
from datetime import timedelta
import warnings
warnings.filterwarnings("ignore")
## https://api.covid19india.org/csv/latest/state_wise.csv
# read data
latest = pd.read_csv('https://api.covid19india.org/csv/latest/state_wise.csv')
# save as a .csv file`
latest.to_csv('state_level_latest.csv', index=False)
# read data
state_wise_daily = pd.read_csv('https://api.covid19india.org/csv/latest/state_wise_daily.csv')
# melt dataframe
state_wise_daily = state_wise_daily.melt(id_vars=['Date', 'Status'], 
                                         value_vars=state_wise_daily.columns[2:], 
                                         var_name='State', value_name='Count')
# pivot table
state_wise_daily = state_wise_daily.pivot_table(index=['Date', 'State'], 
                                                columns=['Status'], 
                                                values='Count').reset_index()
# map state names to state codes
state_codes = {code:state for code, state in zip(latest['State_code'], latest['State'])}
state_codes['DD'] = 'Daman and Diu'
state_wise_daily['State_Name'] = state_wise_daily['State'].map(state_codes)
state_wise_daily=state_wise_daily[state_wise_daily.State_Name!="Total"]
state_wise_daily['Date'] = pd.to_datetime(state_wise_daily['Date'], dayfirst=True)
state_wise_daily.sort_values('Date', ascending=True,inplace=True)
state_wise=state_wise_daily.groupby("State_Name").sum().reset_index()
state_wise["Mortality Rate Per 100"] =np.round(100*state_wise["Deceased"]/state_wise["Confirmed"],2)
state_wise['Mortality Rate Per 100'] = state_wise['Mortality Rate Per 100'].fillna(0)
state_wise.sort_values(by='Mortality Rate Per 100',ascending=False).style.background_gradient(cmap='Blues',subset=["Confirmed"])\
                        .background_gradient(cmap='Greens',subset=["Recovered"])\
                        .background_gradient(cmap='Reds',subset=["Deceased"])\
                        .background_gradient(cmap='YlOrBr',subset=["Mortality Rate Per 100"]).hide_index()
def stanalysis(statename,typ):
    definestate=state_wise_daily[state_wise_daily.State_Name==statename]
    finalstate= definestate.groupby(["Date","State_Name"])[["Confirmed","Deceased","Recovered"]].sum().reset_index().reset_index(drop=True)
    createfigure(finalstate,typ,statename)
    
def createfigure(dataframe,typ,statename):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dataframe["Date"], y=dataframe["Confirmed"],
                    mode="lines+text",
                    name='Confirmed',
                    marker_color='orange',
                        ))
    
    fig.add_trace(go.Scatter(x=dataframe["Date"], y=dataframe["Recovered"],
                    mode="lines+text",
                    name='Recovered',
                    marker_color='Green',
                        ))
    fig.add_trace(go.Scatter(x=dataframe["Date"], y=dataframe["Deceased"],
                    mode="lines+text",
                    name='Deceased',
                    marker_color='Red',
                        ))
      
    fig.add_shape(
        # Line Vertical
        dict(
            type="line",
            x0="2020-03-24",
            y0=dataframe[typ].max(),
            x1="2020-03-24",
    
            line=dict(
                color="red",
                width=5)))
    fig.add_annotation(
            x="2020-03-24",
            y=dataframe[typ].max(),
            text="Lockdown Period",
             font=dict(
            family="Courier New, monospace",
            size=14,
            color="red"
            ),)
    fig.add_annotation(
            x="2020-04-24",
            y=dataframe[typ].max(),
            text="Month after lockdown",
             font=dict(
            family="Courier New, monospace",
            size=14,
            color="Green"
            ),)
    fig.add_shape(
        # Line Vertical
        dict(
            type="line",
            x0="2020-04-24",
            y0=dataframe[typ].max(),
            x1="2020-04-24",
    
            line=dict(
                color="Green",
                width=5)))
    fig
    fig.update_layout(
    title='Evolution of Confirmed-Recovered-Deceased cases over time in '+statename,
        template='gridon')
    fig.show()
    
    
stanalysis("Delhi",'Recovered')
stanalysis("Maharashtra",'Recovered')
stanalysis("Madhya Pradesh",'Recovered')
stanalysis("West Bengal",'Recovered')
# #Run this code for all states visualisation
for states in state_wise_daily.State_Name.unique().tolist():
    if(states!='Daman and Diu'):
        stanalysis(states,'Recovered')
population=state_wise_daily.groupby(["Date"])[["Confirmed","Deceased","Recovered"]].sum().reset_index()
population["day_count"]=list(range(1,len(population)+1))
fig = px.bar(population, x='day_count', y='Confirmed',text='Confirmed')
fig.update_traces(texttemplate='%{text:.2s}', textposition='outside')
fig.update_layout(
xaxis_title="Day",
yaxis_title="Population Effected",
title='Evaluation of Confirmed Cases In India',template='gridon')
fig.show()
def sigmoid(x,c,a,b):
    y = c*1 / (1 + np.exp(-a*(x-b)))
    return y
indiapopulation=1380004385
fmodel=population[population.Confirmed>=50]
fmodel['day_count']=list(range(1,len(fmodel)+1))
fmodel['increase'] = (fmodel.Confirmed-fmodel.Confirmed.shift(1)).fillna(0).astype(int)
fmodel['increaserate']=(fmodel['increase']/fmodel["Confirmed"])
fmodel['Active']=fmodel['Confirmed']-fmodel['Deceased']-fmodel['Recovered']
xdata = np.array(list(abs(fmodel.day_count)))
ydata = np.array(list(abs(fmodel.Active)))
cof,cov = curve_fit(sigmoid, xdata, ydata, method='trf',bounds=([0.,0., 0.],[indiapopulation,1, 100.]))
#‘trf’ : Trust Region Reflective algorithm, particularly suitable for large sparse problems with bounds. Generally robust method.
x = np.linspace(-1, fmodel.day_count.max()+20, 20)
y = sigmoid(x,cof[0],cof[1],cof[2])
fig = go.Figure()
fig.add_trace(go.Scatter(x=x, y=y,
                    mode="lines+text",
                    name='Active Cases Approx',
                    marker_color='orange',
                        ))
    
fig.add_trace(go.Scatter(x=xdata, y=ydata,
                    mode="markers",
                    name='Active Cases',
                    marker_color='Green',
                    marker_line_width=2, marker_size=10
                        ))
fig
fig.update_layout(
title='Daily Active Cases in India is approx '+ str(int(cof[0])) +', Active cases curve started flatten from day ' + str(int(cof[2])) +" and will flatten by day "+str(round(int(cof[2])*2.5)),
        template='gridon', font=dict(
        family="Courier New, monospace",
        size=10,
        color="blue"
    ))
fig.show()
round(fmodel.Active.sum()+((fmodel.day_count.max()+40-fmodel.day_count.max())*y[11:20].mean()))

xdata = np.array(list(abs(fmodel.day_count)))
ydata = np.array(list(abs(fmodel.Confirmed)))
cof,cov = curve_fit(sigmoid, xdata, ydata, method='trf',bounds=([0.,0., 0.],[indiapopulation,1, 100.]))
#‘trf’ : Trust Region Reflective algorithm, particularly suitable for large sparse problems with bounds. Generally robust method.
x = np.linspace(-1, fmodel.day_count.max()+40, 40)
y = sigmoid(x,cof[0],cof[1],cof[2])
fig = go.Figure()
fig.add_trace(go.Scatter(x=x, y=y,
                    mode="lines+text",
                    name='Confirmed Cases Approx',
                    marker_color='Orange',
                        ))
    
fig.add_trace(go.Scatter(x=xdata, y=ydata,
                    mode="markers",
                    name='Confirm Cases',
                    marker_color='Red',
                    marker_line_width=2, marker_size=10
                        ))
fig
fig.update_layout(
title='Daily Confirmed Cases in India is approx '+ str(int(cof[0])) +', Confirm case curve started flatten from day ' + str(int(cof[2])) +" and will flatten by day "+str(round(int(cof[2])*2)),
        template='gridon',
 font=dict(
        family="Courier New, monospace",
        size=7,
        color="blue"
    ))
fig.show()
round(fmodel.Confirmed.sum()+((fmodel.day_count.max()+40-fmodel.day_count.max())*y[15:40].mean()))
