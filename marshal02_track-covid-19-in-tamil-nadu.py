import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

!pip install cufflinks plotly

!pip install chart_studio

from plotly.offline import iplot, init_notebook_mode

import cufflinks

cufflinks.go_offline(connected=True)

init_notebook_mode(connected=True)

import plotly.express as px

import chart_studio.plotly as py

from plotly.graph_objs import *

import plotly.graph_objs as go

from plotly.subplots import make_subplots



import os



for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

#Load the data

India_population = pd.read_csv('/kaggle/input/covid19-in-india/population_india_census2011.csv')



#covid = pd.read_csv('/kaggle/input/covid19-in-india/covid_19_india.csv',parse_dates={'DateTime' : ['Date', 'Time']}, 

                    #infer_datetime_format=True, index_col='DateTime')



India_covid = pd.read_csv('/kaggle/input/covid19-in-india/covid_19_india.csv',parse_dates=['Date'], 

                    infer_datetime_format=True)





ICMRTestingLabs = pd.read_csv('/kaggle/input/covid19-in-india/ICMRTestingLabs.csv')



HospitalBeds = pd.read_csv('/kaggle/input/covid19-in-india/HospitalBedsIndia.csv')



TestingDetails = pd.read_csv('/kaggle/input/covid19-in-india/StatewiseTestingDetails.csv', parse_dates = ['Date'],

                            infer_datetime_format=True,  index_col='Date')



AgeGroupDetails = pd.read_csv('/kaggle/input/covid19-in-india/AgeGroupDetails.csv')





import json 

f = open('/kaggle/input/covid19-cases-in-india/non_virus_deaths.json',) 

data = json.load(f) 
#Pick the data of Tamil Nadu



covid = India_covid[India_covid['State/UnionTerritory']=='Tamil Nadu'].reset_index(drop=True).drop(['State/UnionTerritory', 'Sno', 'Time'], axis = 1)



population = India_population[India_population['State / Union Territory']=='Tamil Nadu'].reset_index(drop=True).drop(['State / Union Territory', 'Sno'], axis = 1)



ICMRTestingLabs = ICMRTestingLabs[ICMRTestingLabs['state']=='Tamil Nadu'].reset_index(drop=True).drop('state', axis = 1)



HospitalBeds = HospitalBeds[HospitalBeds['State/UT']=='Tamil Nadu'].reset_index(drop=True).drop(['State/UT', 'Sno'], axis = 1)



TestingDetails = TestingDetails[TestingDetails['State']=='Tamil Nadu'].drop('State', axis = 1)



#Preprocessing

#In covid data



for i in range(0, len(covid)):

    if((i>=0 and i<=5) or (i>=25 and i<=36) or (i>=55 and i<=66) or (i>=86 and i<=97)):

        covid['Date'].iloc[i] = (covid['Date'].iloc[i].strftime("%Y-%d-%m"))

        #print(covid['Date'].iloc[i])

    else:

        covid['Date'].iloc[i] = (covid['Date'].iloc[i].strftime("%Y-%m-%d"))

        #print(covid['Date'].iloc[i])

               

#
Total = India_population['Population'].sum()

TN = population['Population']

labels = ["Rest of the India", 'TN']

values = [float(Total - TN), float(TN)]

colors = ['rgb(211,211,211)', 'rgba(25, 140, 229, 0.8)']

fig = go.Figure(data = [go.Pie(title = 'Population Percentage', labels= labels, values = values, textinfo='label+ percent', marker_colors = colors,

                              textfont = dict(color = 'white'), pull = [.05,0])])



fig.update_layout(title_text= 'Population of Tamil Nadu', showlegend = False)



fig.show()
Total_confirmed = India_covid.loc[India_covid.groupby('State/UnionTerritory').Confirmed.idxmax()].Confirmed.sum()

Total_deaths = India_covid.loc[India_covid.groupby('State/UnionTerritory').Deaths.idxmax()].Deaths.sum()

Total_cured = India_covid.loc[India_covid.groupby('State/UnionTerritory').Cured.idxmax()].Cured.sum()





TN_confirmed = covid.Confirmed.iloc[-1]

TN_deaths = covid.Deaths.iloc[-1]

TN_cured = covid.Cured.iloc[-1]





confirm = [Total_confirmed, TN_confirmed]

cured = [Total_cured, TN_cured]

deaths = [Total_deaths, TN_deaths]



specs = [[{'type':'domain'}, {'type':'domain'}, {'type':'domain'}]]

fig = make_subplots(rows=1, cols=3, specs = specs)

labels = ["Rest of the India", 'TN']

colors = ['rgb(211,211,211)', 'rgba(25, 140, 229, 0.8)']



# Define pie charts

fig.add_trace(go.Pie(title = dict(text = 'Confirmed', 

                                  font = dict(color = 'rgb(0,0,0)')),labels=labels, values=confirm, name='Confirmed',textinfo='label+ percent',

                     marker_colors=colors, pull = [.05,0], insidetextorientation = 'horizontal'), 1, 1)



fig.add_trace(go.Pie(title = dict(text = 'Cured', font = dict(color = 'rgb(0,0,0)')), labels=labels, values= cured, name='Cured', textinfo='label+ percent',

                     marker_colors=colors, pull = [.05,0], insidetextorientation = 'horizontal'), 1, 2)

fig.add_trace(go.Pie(title = dict(text = 'Deaths', font = dict(color = 'rgb(0,0,0)')), labels=labels, values=deaths, name='Deaths', textinfo='label+ percent',

                     marker_colors=colors, pull = [.05,0], insidetextorientation = 'horizontal'), 1, 3)



fig = go.Figure(fig)



fig.update_layout(

             showlegend = False

             )



fig.show()
fig = go.Figure()

fig.add_trace(go.Scatter(

                    x = covid.Date,

                    y = covid.Confirmed,

                    mode = "lines",

                    name = "Confirmed Cases",

                    marker = dict(color = 'rgba(16, 112, 2, 0.8)'),

                    text= covid.Confirmed,

                    showlegend= True,

                    line = dict(color = 'rgba(25, 140, 229, 0.8)', width = 3)

                    ))





fig.add_trace(go.Scatter(

        x=[covid.Date.iloc[-1]],

        y=[covid.Confirmed.iloc[-1]],

        mode='markers',

        name = "Confirmed Cases",

        text = [covid.Date.iloc[-1], covid.Confirmed.iloc[-1]],

        marker=dict(color='rgba(25, 140, 229, 0.8)', size= 8),

        showlegend= False,

        

    ))





#data = [trace1]

fig.update_layout(

              xaxis= dict(title= 'Months',ticklen= 5,zeroline= False),

              legend=dict(x=.05, y=1, traceorder='normal', font=dict(size=12,),),

              plot_bgcolor='rgba(0,0,0,0)'

             )





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

        ),

    ),

    yaxis=dict(

        showgrid=False,

        zeroline=False,

        showline=False,

        showticklabels=False,

    ),

    autosize=False,

    margin=dict(

        autoexpand=False,

        l=100,

        r=20,

        t=110,

    ),

    plot_bgcolor='white'

)





annotations = []





annotations.append(dict(xref='paper', x=0.95, y=covid.Confirmed.iloc[-1],

                                  xanchor='left', yanchor='bottom',

                                  text= str(covid.Confirmed.iloc[-1]),

                                  font=dict(family='Arial',

                                            size=16),

                                  showarrow=False))



fig.update_layout(annotations=annotations)







fig.show()
fig = go.Figure()



active = covid.Confirmed - covid.Cured - covid.Deaths



fig.add_trace(go.Scatter(

                    x = covid.Date,

                    y = active,

                    mode = "lines",

                    name = "Active Cases",

                    marker = dict(color = 'rgba(16, 112, 2, 0.8)'),

                    text= active,

                    showlegend= True,

                    line = dict(color = 'rgba(25, 140, 229, 0.8)', width = 3)

                    ))





fig.add_trace(go.Scatter(

        x=[covid.Date.iloc[-1]],

        y=[active.iloc[-1]],

        mode='markers',

        name = "Active Cases",

        text = [covid.Date.iloc[-1], active.iloc[-1]],

        marker=dict(color='rgba(25, 140, 229, 0.8)', size= 8),

        showlegend= False,

        

    ))





#data = [trace1]

fig.update_layout(

              xaxis= dict(title= 'Months',ticklen= 5,zeroline= False),

              legend=dict(x=.05, y=1, traceorder='normal', font=dict(size=12,),),

              plot_bgcolor='rgba(0,0,0,0)'

             )





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

        ),

    ),

    yaxis=dict(

        showgrid=False,

        zeroline=False,

        showline=False,

        showticklabels=False,

    ),

    autosize=False,

    margin=dict(

        autoexpand=False,

        l=100,

        r=20,

        t=110,

    ),

    plot_bgcolor='white'

)





annotations = []





annotations.append(dict(xref='paper', x=0.95, y=active.iloc[-1],

                                  xanchor='left', yanchor='bottom',

                                  text= str(active.iloc[-1]),

                                  font=dict(family='Arial',

                                            size=16),

                                  showarrow=False))



fig.update_layout(annotations=annotations)







fig.show()
fig = go.Figure()

fig.add_trace(go.Scatter(

                    x = covid.Date,

                    y = covid.Deaths,

                    mode = "lines",

                    name = "Deaths",

                    marker = dict(color = 'rgba(16, 112, 2, 0.8)'),

                    text= covid.Deaths,

                    showlegend= True,

                    line = dict(color = 'rgba(25, 140, 229, 0.8)', width = 3)

                    ))





fig.add_trace(go.Scatter(

        x=[covid.Date.iloc[-1]],

        y=[covid.Deaths.iloc[-1]],

        mode='markers',

        name = "COVID Deaths",

        text = [covid.Date.iloc[-1], covid.Deaths.iloc[-1]],

        marker=dict(color='rgba(25, 140, 229, 0.8)', size= 8),

        showlegend= False,

        

    ))





#data = [trace1]

fig.update_layout(

              xaxis= dict(title= 'Months',ticklen= 5,zeroline= False),

              legend=dict(x=.05, y=1, traceorder='normal', font=dict(size=12,),),

              plot_bgcolor='rgba(0,0,0,0)'

             )





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

        ),

    ),

    yaxis=dict(

        showgrid=False,

        zeroline=False,

        showline=False,

        showticklabels=False,

    ),

    autosize=False,

    margin=dict(

        autoexpand=False,

        l=100,

        r=20,

        t=110,

        

    ),

    plot_bgcolor='white'

)





annotations = []





annotations.append(dict(xref='paper', x=0.95, y=covid.Deaths.iloc[-1],

                                  xanchor='left', yanchor='bottom',

                                  text= str(covid.Deaths.iloc[-1]),

                                  font=dict(family='Arial',

                                            size=16),

                                  showarrow=False))



fig.update_layout(annotations=annotations)







fig.show()
fig = go.Figure()

fig.add_trace(go.Scatter(

                    x = covid.Date,

                    y = covid.Cured,

                    mode = "lines",

                    name = "Cured",

                    marker = dict(color = 'rgba(16, 112, 2, 0.8)'),

                    text= covid.Cured,

                    showlegend= True,

                    line = dict(color = 'rgba(25, 140, 229, 0.8)', width = 3)

                    ))





fig.add_trace(go.Scatter(

        x=[covid.Date.iloc[-1]],

        y=[covid.Cured.iloc[-1]],

        mode='markers',

        name = "COVID Cured",

        text = [covid.Date.iloc[-1], covid.Cured.iloc[-1]],

        marker=dict(color='rgba(25, 140, 229, 0.8)', size= 8),

        showlegend= False,

        

    ))





#data = [trace1]

fig.update_layout(

              xaxis= dict(title= 'Months',ticklen= 5,zeroline= False),

              legend=dict(x=.05, y=1, traceorder='normal', font=dict(size=12,),),

              plot_bgcolor='rgba(0,0,0,0)'

             )





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

        ),

    ),

    yaxis=dict(

        showgrid=False,

        zeroline=False,

        showline=False,

        showticklabels=False,

    ),

    autosize=False,

    margin=dict(

        autoexpand=False,

        l=100,

        r=20,

        t=110,

        

    ),

    plot_bgcolor='white'

)





annotations = []





annotations.append(dict(xref='paper', x=0.95, y=covid.Cured.iloc[-1],

                                  xanchor='left', yanchor='bottom',

                                  text= str(covid.Cured.iloc[-1]),

                                  font=dict(family='Arial',

                                            size=16),

                                  showarrow=False))



fig.update_layout(annotations=annotations)







fig.show()
HospitalBeds.rename(columns = {'NumPrimaryHealthCenters_HMIS' : 'Primary Health Centers', 'NumCommunityHealthCenters_HMIS' : 'Community Health Centers',

                    'NumSubDistrictHospitals_HMIS' : 'Sub District Hospitals', 'NumDistrictHospitals_HMIS' : 'District Hospitals',

                    'TotalPublicHealthFacilities_HMIS' : 'Total Public Health Facilities', 'NumPublicBeds_HMIS' : 'Public Beds',

                    'NumRuralHospitals_NHP18' : 'Rural Hospitals', 'NumRuralBeds_NHP18' : 'Rural Beds',

                    'NumUrbanHospitals_NHP18' : 'Urban Hospitals', 'NumUrbanBeds_NHP18' : 'Urban Beds'},

                    index = {0 : 'Count'}, inplace = True)



HospitalBeds = HospitalBeds.T
fig = go.Figure(data = [go.Table(

    columnwidth = [.5,2],

    header=dict(values=['<b>Health Centers</b><br>     <b>& Beds</b>','<b>'+str(HospitalBeds.columns[0])+'</b>'],

                fill_color='paleturquoise', align = 'left'),

    cells=dict(values=[HospitalBeds.index, HospitalBeds.Count],

               fill_color='lavender',

               align='left'))])



fig.show()
non_virus_deaths = []

for i in range(data['total_rows']):

    non_virus_deaths.append(data['rows'][i]['value'])

    

non_virus_deaths = pd.DataFrame(non_virus_deaths)
Total = non_virus_deaths.groupby('state')['deaths'].sum().sum()

TN = non_virus_deaths.groupby('state')['deaths'].sum()['TN']





labels = ["Rest of the India", 'TN']

values = [float(Total - TN), float(TN)]

colors = ['rgb(211,211,211)', 'rgba(25, 140, 229, 0.8)']

fig = go.Figure(data = [go.Pie(

                               labels= labels, values = values, textinfo='label+ percent', marker_colors = colors,

                               pull = [.05,0], insidetextorientation = 'horizontal')])



fig.update_layout( showlegend = False)



fig.show()

from collections import Counter 



Total = non_virus_deaths['reason'].sum()

TN = non_virus_deaths[non_virus_deaths['state']=='TN']['reason'].sum()



def group_list(lst): 

    return list(zip(Counter(lst).keys(), Counter(lst).values()))



Total_reason = pd.DataFrame(group_list(Total), columns = ['Reason', 'Count'])

Total_reason.sort_values(by = ['Count'], ascending = False, inplace = True)

Total_reason.reset_index(drop = True, inplace = True)





TN_reason = pd.DataFrame(group_list(TN), columns = ['Reason', 'Count'])

TN_reason.sort_values(by = ['Count'], ascending = False, inplace = True)

TN_reason.reset_index(drop = True, inplace = True)





Total_total_count = Total_reason['Count'].sum()

TN_total_count = TN_reason['Count'].sum()





Total_other = Total_total_count - (Total_reason['Count'][0] + Total_reason['Count'][1]

                                   + Total_reason['Count'][2] + Total_reason['Count'][3] + Total_reason['Count'][4])

TN_other = TN_total_count - (TN_reason['Count'][0] + TN_reason['Count'][1]

                             + TN_reason['Count'][2] + TN_reason['Count'][3] + TN_reason['Count'][4])







Total_y_data = [int(Total_reason['Count'][0]/Total_total_count*100), int(Total_reason['Count'][1]/Total_total_count*100),

           int(Total_reason['Count'][2]/Total_total_count*100), int(Total_reason['Count'][3]/Total_total_count*100),

           int(Total_reason['Count'][4]/Total_total_count*100), int(Total_other/Total_total_count*100)]





TN_y_data = [int(TN_reason['Count'][0]/TN_total_count*100), int(TN_reason['Count'][1]/TN_total_count*100),

           int(TN_reason['Count'][2]/TN_total_count*100), int(TN_reason['Count'][3]/TN_total_count*100),

           int(TN_reason['Count'][4]/TN_total_count*100), int(TN_other/TN_total_count*100)]









#x_data = [[23, ]]



Total_x_data = [Total_reason['Reason'][0], Total_reason['Reason'][1], Total_reason['Reason'][2], Total_reason['Reason'][3], Total_reason['Reason'][4], 'Other']

TN_x_data = [TN_reason['Reason'][0], TN_reason['Reason'][1], TN_reason['Reason'][2], TN_reason['Reason'][3], TN_reason['Reason'][4], 'Other']







# Creating two subplots

fig = make_subplots(rows=1, cols=2, specs=[[{}, {}]], shared_xaxes=True,

                    shared_yaxes=False, vertical_spacing=0.001)





fig.append_trace(go.Bar(

    x=Total_y_data,

    y=Total_x_data,

    marker=dict(

        color='rgba(25, 140, 229, 0.8)',

    ),

    name='Non Virus Deaths in India',

    orientation='h',

), 1, 1)





fig.append_trace(go.Bar(

    x=TN_y_data,

    y=TN_x_data,

    marker=dict(

        color='rgba(128, 0, 128, 0.8)',

        line=dict(

            color='rgba(128, 0, 128, 0.8)',

            width=1),

    ),

    name='Non Virus Deaths in Tamil Nadu',

    orientation='h',

), 1, 2)







fig.update_layout(

    

    yaxis=dict(

        showgrid=False,

        showline=False,

        showticklabels=True,

        domain=[0, 0.85],

        autorange = 'reversed',

    ),

    yaxis2=dict(

        showgrid=False,

        showline=False,

        showticklabels=True,

        domain=[0, 0.85],

        autorange = 'reversed',

    ),



    xaxis=dict(

        zeroline=False,

        showline=False,

        showticklabels=False,

        

        domain=[0, 0.42],

    ),

    

    xaxis2=dict(

        zeroline=True,

        showline=True,

        showticklabels=False,

        

        domain=[0.55, 1],

    ),



    legend=dict(x=0.029, y=1.038, font_size=10),

    #margin=dict(l=100, r=20, t=70, b=70),

    #paper_bgcolor='rgba(255,255, 255)',

    plot_bgcolor= 'white'

    #'rgba(255, 255 , 255)',

)



#248, 248, 255

annotations = []



Total_y = np.array(Total_y_data)

TN_y = np.array(TN_y_data)



# Adding labels

for ydn, yd, xdn, xd in zip(TN_y, Total_y, TN_x_data,  Total_x_data):

    # labeling the bar net worth

    annotations.append(dict(xref='x1', yref='y1',

                            y=xd, x=yd + 5,

                            text=str(yd) + '%',

                            font=dict(family='Arial', size=12,

                                      color='rgb(0, 0, 0)'),

                            showarrow=False))

    # labeling the bar net worth

    annotations.append(dict(xref='x2', yref='y2',

                            y=xdn, x=ydn + 4,

                            text=str(ydn) + '%',

                            font=dict(family='Arial', size=12,

                                      color='rgb(0, 0, 0)'),

                            showarrow=False))

# Source





fig.update_layout(annotations=annotations)



fig.show()
