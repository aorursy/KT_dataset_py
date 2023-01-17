import pandas as pd

import plotly.express as px

import plotly.graph_objects as go

import numpy as np
source = pd.read_csv("/kaggle/input/covid19-in-india/covid_19_india.csv", usecols=['Date', 'State/UnionTerritory', 'Confirmed'])

source['last10days'] = 1





for i in source['State/UnionTerritory'].unique():

    idx = source[source['State/UnionTerritory'] == i].index

    source.loc[idx, 'last10days'] = source.loc[idx, 'Confirmed'].diff(periods=10)

    source.loc[idx[:10], 'last10days'] = source.loc[idx[:10], 'Confirmed']

    

    

source['Date'] = pd.to_datetime(source['Date'], format="%d/%m/%y")



subdata = source[source['Date'] >= pd.Timestamp(2020, 3, 20)]



# Calculation of Rate of Growth required for corresponding doubling-period.



d = np.array([2, 5, 7, 10, 15, 20, 30])

r = 1 - np.exp(-10*np.log(2)/d)



scope = ['Maharashtra', 'Delhi', 'Rajasthan', 'Gujarat', 'Tamil Nadu', 

         'Kerala', 'West Bengal', 'Uttar Pradesh']





shortform = {'Maharashtra':'MH', 'Gujarat':'GJ', 'Delhi':'DL', 'Rajasthan':'RJ', 'Madhya Pradesh':'MP', 

             'Tamil Nadu':'TN', 'Uttar Pradesh':'UP', 'Telengana':'TS', 'Kerala':'KL', 'West Bengal':'WB'}



#scope = ['Uttar Pradesh']



colorlist = (px.colors.qualitative.Plotly[:7] + px.colors.qualitative.Vivid[:2] + 

             px.colors.qualitative.Vivid[3:] + px.colors.qualitative.Set2 + px.colors.sequential.Peach)





''' First Frame '''





firstdatalist = []



counter = -1



for i in range(len(d)):

    

    firstdatalist.append(

            go.Scatter(

            x = np.array([2, 5*10**4]),

            y = np.array([2, 5*10**4])*r[i],

            mode="lines", line_width=1.6, line_color=colorlist[counter], line_dash='solid', 

            opacity=1.0, name='double in {} days'.format(d[i])))

    

    counter = counter-1



counter = 0

for i in scope:

        

    firstdatalist.append(

        go.Scatter(

            x = subdata.loc[(subdata['State/UnionTerritory'] == i) & 

                            (subdata['Date'] == pd.Timestamp(2020, 3, 21)), 'Confirmed'],

            y = subdata.loc[(subdata['State/UnionTerritory'] == i) & 

                            (subdata['Date'] == pd.Timestamp(2020, 3, 21)), 'last10days'],

            mode="markers+text", marker_size=8, marker_color=colorlist[counter],

            opacity=1.0, text=shortform[i], textposition='middle right',  showlegend=False))

    

    firstdatalist.append(

        go.Scatter(

            x = subdata.loc[(subdata['State/UnionTerritory'] == i) & 

                            (subdata['Date'] <= pd.Timestamp(2020, 3, 21)), 'Confirmed'],

            y = subdata.loc[(subdata['State/UnionTerritory'] == i) & 

                            (subdata['Date'] <= pd.Timestamp(2020, 3, 21)), 'last10days'],

            mode="lines", line_shape='spline', line_dash='4, 2, 4, 2', line_width=2, marker_size=3,

            opacity=1.0, marker_color=colorlist[counter], name=i))  

    

    counter = counter+1

    



''' Animation Frames '''    

    

    

framelist=[]



for k in pd.date_range(start='3/22/2020', end=subdata['Date'].max()):

    datalist =[]

        

    counter = -1



    for i in range(len(d)):

    

        datalist.append(

            go.Scatter(

            x = np.array([2, 5*10**4]),

            y = np.array([2, 5*10**4])*r[i],

            mode="lines", line_width=1.6, line_color=colorlist[counter], line_dash='solid', 

            opacity=0.8, name='double in {} days'.format(d[i]), showlegend=False))

    

        counter = counter-1

    

    counter = 0

    

    for i in scope:

        

        datalist.append(

            go.Scatter(

            x = subdata.loc[(subdata['State/UnionTerritory'] == i) & (subdata['Date'] == k), 'Confirmed'],

            y = subdata.loc[(subdata['State/UnionTerritory'] == i) & (subdata['Date'] == k), 'last10days'],

            mode="markers+text", marker_size=8, marker_color=colorlist[counter],

            opacity=1.0, text=shortform[i], textposition='middle right', showlegend=False))

        

        datalist.append(

            go.Scatter(

            x = subdata.loc[(subdata['State/UnionTerritory'] == i) & (subdata['Date'] <= k), 'Confirmed'],

            y = subdata.loc[(subdata['State/UnionTerritory'] == i) & (subdata['Date'] <= k), 'last10days'],

            mode="lines", line_shape='spline', line_dash='4, 2, 4, 2', line_width=2, marker_size=3, 

            opacity=0.7, marker_color=colorlist[counter], name=shortform[i], showlegend=False))

        

        counter = counter+1

        

    framelist.append(go.Frame(data = datalist))

    

    
fig = go.Figure(

                data = firstdatalist,



                layout = go.Layout(height=650, template='plotly_white',

                                   xaxis_type="log", yaxis_type="log", xaxis_range=[1.3, 4.602], yaxis_range=[1.3, 4.578],

                                   title="Covid-19 : Trajectory and efficacy of mitigation steps by {} States in India".format(len(scope)),

                                   xaxis_title="Total Cases", yaxis_title="New Cases in last 10 days",

                                   transition_duration=10000,

                                   updatemenus=[dict( type="buttons", buttons=[dict(label="Play", method="animate", 

                                                args=[None, {"frame": {"duration": 500, "redraw": False},

                                                "fromcurrent": True, 

                                                "transition": {"duration": 50, "easing": "linear-out"}}])])]),



                frames = framelist 

                

                )



go.Figure.write_html(fig, file='animation1.html')

fig.show()