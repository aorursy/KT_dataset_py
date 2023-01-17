# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.tools as tls
df = pd.read_csv("../input/rainfall in india 1901-2015.csv") # Reading the given csv format data.
df.head(3)
df.set_index("YEAR", inplace=True) # Setting the Year as an index.
df.columns = df.columns.str.upper() # making all the column headers upper.
print("The list of all states : ", df["SUBDIVISION"].unique())
print("\nThe number of all states : ", df["SUBDIVISION"].nunique())
# It is useful data to pass as an argument for the upcoming functions.
avg_df = df["ANNUAL"].groupby(df["SUBDIVISION"])
avg = pd.DataFrame(avg_df.mean())
avg["ANNUAL"] = avg["ANNUAL"].round() 
avg["STATES"] = avg.index

trace = go.Bar(
    x=avg["STATES"],
    y=avg["ANNUAL"])

layout = go.Layout(title='Average Rainfall from 1901 to 2015 in Indian States',
                  yaxis = {"title": "Rainfall in (mm)"})

data = [trace]

fig = go.Figure(data=data, layout=layout)
py.iplot(fig)
def Rainfall_mon(subdivision, month):
    sub = subdivision.upper() # Making all the arguments passed upper.
    mon = month.upper()
    
    try: # Exception Handeling
        subdivision = df[df["SUBDIVISION"] == sub]
        month_data = pd.DataFrame(subdivision[mon])
        month_data["YEAR"] = month_data.index
        month_data.fillna(method='ffill',inplace=True)
        
        trace = go.Scatter(marker = ({"color":"#26A69A"}),
            x=month_data["YEAR"],
            y=month_data[mon])

        layout = go.Layout(title='Rainfall from 1901 to 2015 in ' + mon + ' Month in ' + sub,
                           yaxis = {"title" : "Rainfall in (mm)"},
                           xaxis = {"title" : "Year"})

        data = [trace]

        fig = go.Figure(data=data, layout=layout)
        py.iplot(fig)
    except Exception:
        print("Please enter correect values")

Rainfall_mon("MADHYA MAHARASHTRA", "may")
Rainfall_mon("MADHYA MAHARASHTRA", "aug")
# Rainfall_mon("MADHYA MAHARASHTRA", "nov")
# Rainfall_mon("PUNJAB", "apr") 
# Rainfall_mon("PUNJAB", "jul")
def pct_change_before(ipt_state):
    state = ipt_state.upper()
    
    subdivision = df[df["SUBDIVISION"] == state]

    month_data = pd.DataFrame(subdivision["ANNUAL"])
    month_data.fillna(method='ffill',inplace=True)
    month_data = month_data.pct_change() # This line of code modifies the annual column to the pct_change. 

    trace = go.Scatter(x = month_data.index, y = month_data["ANNUAL"])

    layout = go.Layout(title = "Percent Change in Rainfall in " + state + " w.r.t. Year Before Current Year",
                    xaxis = {"title": "Year"},
                    yaxis = {"title": "Percentage Rainfall in (mm)"})

    data = [trace]
    fig = go.Figure(data=data, layout=layout)
    return py.iplot(fig)
   
# pct_change_before("BIHAR")
pct_change_before("MADHYA MAHARASHTRA")
# pct_change_before("KONKAN & GOA")
def get_annual_rf(ipt_state):
    
    state = ipt_state.upper()

    subdivision = df[df["SUBDIVISION"] == state]["ANNUAL"]
    change_1 = pd.DataFrame(subdivision)

    new_data = []
    for an in subdivision:
        new = (an - subdivision.iloc[0]) / subdivision.iloc[0] * 100.0 
        new_data.append(new)  

    change_1["PCT_CHANGE"] = new_data
    change_1["YEAR"] = subdivision.index
    
    trace = go.Scatter(x = change_1["YEAR"],
                       y = change_1["PCT_CHANGE"] )
    
    layout = go.Layout(title = "Percent Change in Rainfall in " + state + " from 1901",
                      xaxis = {"title": "Year"},
                      yaxis = {"title": "Percentage Rainfall in (mm)"},)
    
    data = [trace]
    fig = go.Figure(data=data, layout=layout)
    py.iplot(fig)
    
get_annual_rf("MADHYA MAHARASHTRA")
def get_annual_rain_report(ipt_state):
    
    state = ipt_state.upper()
    
    # Chossing a specific column to analyze.
    subdivision = df[df["SUBDIVISION"] == state]
    
    # Forwardly filling the empty values.
    subdivision.fillna(method='ffill', inplace=True)

    dframe = pd.DataFrame(subdivision)

    fig = tls.make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing = 0.05)

    for col in ['JAN', 'FEB', 'MAR', 'APR']:
        fig.append_trace({'x': dframe.index, 'y': dframe[col], 'type': 'scatter', 'name': col}, 1, 1)

    for col in ['MAY', 'JUN', 'JUL', 'AUG']:
        fig.append_trace({'x': dframe.index, 'y': dframe[col], 'type': 'scatter', 'name': col}, 2, 1)    

    for col in ['SEP', 'OCT', 'NOV', 'DEC']:
        fig.append_trace({'x': dframe.index, 'y': dframe[col], 'type': 'scatter', 'name': col}, 3, 1)    

    fig['layout']['yaxis2'].update(title='Rainfall in (mm)')

    fig['layout'].update(height=800, width=820,
                         title='Rainfall in ' + state)
    py.iplot(fig)

get_annual_rain_report('PUNJAB')
def get_annual_rain_report(ipt_state):
    
    state = ipt_state.upper()
    
    # Chossing a specific column to analyze.
    subdivision = df[df["SUBDIVISION"] == state]
    
    # Forwardly filling the empty values.
    subdivision.fillna(method='ffill', inplace=True)

    dframe = pd.DataFrame(subdivision)

    trace1 = go.Bar(
    y= dframe.index,
    x= dframe["JAN-FEB"], orientation = "h",
    name='JAN-FEB')
    
    trace2 = go.Bar(
    y= dframe.index,
    x= dframe["MAR-MAY"] , orientation = "h",
    name='MAR-MAY')
    
    trace3 = go.Bar(
    y= dframe.index,
    x= dframe["JUN-SEP"] , orientation = "h",
    name='JUN-SEP')
    
    trace4 = go.Bar(
    y= dframe.index,
    x= dframe["OCT-DEC"] , orientation = "h",
    name='OCT-DEC')
    
    data = [trace1, trace2, trace3,trace4]
    layout = go.Layout(barmode='stack')
    
    fig = go.Figure(data=data, layout=layout)
    fig['layout'].update(height=1500, width=820,
                        title = "Rainfall in " + state)
    py.iplot(fig)

get_annual_rain_report('MADHYA MAHARASHTRA')

values = [df["JAN"].sum(), df["FEB"].sum(), df["MAR"].sum(), df["APR"].sum(),
            df["MAY"].sum(), df["JUN"].sum(), df["JUL"].sum(), df["AUG"].sum(),
            df["SEP"].sum(), df["OCT"].sum(), df["NOV"].sum(), df["DEC"].sum()]

labels = [ 'JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC']


fig = {
      "data": [
        {
          "values": values,
          "labels": labels,
          "hoverinfo":"label+percent+name",
          "name":"Rainfall %",
          "hole": .6,
          "type": "pie"
        }],
          "layout": {
            "annotations": [
                {
                    "font": {
                        "size": 16
                    },
                    "showarrow": False,
                    "text": "<b>Month-wise <br>Rainfall <br>Percentage in <br> INDIA <br>from 1901 to 2015"
                }
            ]
        }
    }
py.iplot(fig, filename='donut')