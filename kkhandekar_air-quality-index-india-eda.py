# -- Libraries --

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import heapq



import plotly.express as px

import plotly.graph_objects as go

import matplotlib.pyplot as plt

import seaborn as sns



# Warning

import warnings

warnings.filterwarnings("ignore")



from tabulate import tabulate
# -- Data Load --

url = '../input/air-quality-data-india-from-20152020/city_day_new.csv'

data = pd.read_csv(url, header='infer')



# Drop Null/Missing Records

data.dropna(inplace=True)



# Reset Index

data.reset_index(inplace=True,drop=True)
#Inspect

data.head()
# Stat Summary

data.describe().transpose()
# Function to label AQI Bucket based on AQI Values

def AQI_Calc(x):

    if 0 <= x <= 50:

        return "Good"

    elif 51 <= x <= 100:

        return "Satisfactory"

    elif 100 <= x <= 200:

        return "Moderate"

    elif 201 <= x <= 300:

        return "Poor"

    elif 301 <= x <= 400:

        return "Very Poor"

    else:

        return "Severe"



# Function to aggregate monthly data & get the average for Particular City

def aggData (city):

    df = data[data['City'] == city]

    df.reset_index(inplace=True,drop=True)

    

    #Removing AQI_Bucket Column

    df.drop(['AQI_Bucket'], axis=1,inplace=True)     

    

    #convert to DateTime

    df['Date'] = pd.to_datetime(df['Date'])   

    

    df.index = df['Date'] 

    df = df.resample('M').mean()

    

    # Drop Null/Missing Records

    df.dropna(inplace=True)



    # Calculate AQI Bucket from AQI

    df['AQI_Bucket'] = df['AQI'].apply(lambda x: AQI_Calc(x))

    

    #Convert Date to Column 

    df.reset_index(level=0, inplace=True)

    

    df['Date'] = df["Date"].dt.strftime('%d-%b-%Y')

    

    return df

# Seperate dataframe for Hyderabad with monthly aggregated values

df  = aggData('Hyderabad')
# Inspect New Dataframe

df.head()
# Utility Function to Plot Density Distribution Plot



def distPlot():

   

    hi = heapq.nlargest(1, zip(df['AQI'], df['Date']))

    lw = heapq.nsmallest(1, zip(df['AQI'], df['Date']))

    

    hi_val, hi_dt = zip(*hi)  # Unzip high value list

    lw_val, lw_dt = zip(*lw)  # Unzip low value list

    

    # Define a new list

    tab = []

        

    # Iterating over unzipped list (high val)

    for i,j in zip(hi_dt,lw_dt):

        tab.append(["AQI Index",i,j])

    

    print("DATES ON WHICH THE AQI WAS EXTREME \n")

    print(tabulate(tab, headers=['','Poor','Good']))

    

    fig = plt.figure(figsize=(20, 15))

    plt.subplots_adjust(hspace = 0.3)

    sns.set_palette('muted')

        

    plt.subplot(221)

    ax1 = sns.distplot(df['PM2.5'], color = 'r',label='pm2.5')

    ax1 = sns.distplot(df['PM10'], color = 'b',label='pm10')

    plt.title('Particulate Matter Distribution')

    ax1.legend(loc='upper right')

    ax1.set_xlabel('')  

    

    

    plt.subplot(222)

    ax2 = sns.distplot(df['NO'], color = 'r',label='NO')

    ax2 = sns.distplot(df['NO2'], color = 'b',label='NO2')

    ax2 = sns.distplot(df['NOx'], color = 'g',label='NOx')

    plt.title('Nitrous Oxides Distribution')

    ax2.legend(loc='upper right')

    ax2.set_xlabel('')  

    

    plt.subplot(223)

    ax3 = sns.distplot(df['CO'], color = 'r',label='CO')

    ax3 = sns.distplot(df['SO2'], color = 'b',label='SO2')

    ax3 = sns.distplot(df['O3'], color = 'g',label='O3')

    plt.title('CO, SO2 & Ground Level Ozone Distribution')

    ax3.legend(loc='upper right')

    ax3.set_xlabel('')  

    

    plt.subplot(224)

    ax4 = sns.distplot(df['Benzene'], color = 'r',label='Benzene')

    ax4 = sns.distplot(df['Toluene'], color = 'b',label='Toulene')

    ax4 = sns.distplot(df['Xylene'], color = 'g',label='Xylene')

    plt.title('Benzene, Toluene & Xylene Distribution')

    ax4.legend(loc='upper right')

    ax4.set_xlabel('')  

    

    plt.show()

# Plotting Distribution Plot for Hyderabad

distPlot()
# Function to Plot Particulate Matter

def PM_Plot():

    title = 'Particulate Matter [2015-2020]'

    labels = ['PM2.5', 'PM10','AQI']

    colors = ['rgb(67,67,67)', 'rgb(115,115,115)','rgb(49,130,189)']

    

    mode_size = [8, 8, 10]

    line_size = [2, 2, 3]

    

    x_data = np.vstack((np.array(df['Date']),)*3)

    y_data = np.array([ df['PM2.5'],df['PM10'], df['AQI'] ])

    

    fig = go.Figure()



    for i in range(0, 3):

        fig.add_trace(go.Scatter(x=x_data[i], y=y_data[i], mode='lines',name=labels[i],line=dict(color=colors[i], width=line_size[i]),

        connectgaps=True,))

    

        fig.add_trace(go.Scatter(x=[x_data[i][0], x_data[i][-1]],y=[y_data[i][0], y_data[i][-1]],mode='markers',

        marker=dict(color=colors[i], size=mode_size[i])

    ))

    

    fig.update_layout(xaxis=dict(showline=True,showgrid=False,showticklabels=True,linecolor='rgb(204, 204, 204)',

                      linewidth=2,ticks='outside',tickfont=dict(family='Arial',size=12,color='rgb(82, 82, 82)',),),

                      yaxis=dict(showgrid=False,zeroline=False,showline=False,showticklabels=False,),

                      autosize=False,margin=dict(autoexpand=False,l=100,r=20,t=110,),

                      showlegend=False,plot_bgcolor='white')

    

    annotations = []



    # Adding labels

    for y_trace, label, color in zip(y_data, labels, colors):

        

        # labeling the left_side of the plot

        annotations.append(dict(xref='paper', x=0.05, y=y_trace[0],xanchor='right', yanchor='middle',

                                text=label,font=dict(family='Arial',size=12),

                                showarrow=False))

        

        # labeling the right_side of the plot

        annotations.append(dict(xref='paper', x=0.95, y=y_trace[11],xanchor='left', yanchor='middle',

                                text='',font=dict(family='Arial',size=12),

                                showarrow=False))



        # Title

        annotations.append(dict(xref='paper', yref='paper', x=0.0, y=1.05,xanchor='left', yanchor='middle',

                                text=title, font=dict(family='Arial',size=25,color='rgb(37,37,37)'),

                                showarrow=False))



    fig.update_layout(annotations=annotations,autosize=True)

    fig.update_xaxes(tickangle=45)

    fig.show()

# Function to Plot Nitrogen Oxides



def NO_Plot():

    title = 'Nitrogen Oxides [2015-2020]'

    labels = ['NO', 'NO2','NOx']

    colors = ['rgb(67,67,67)', 'rgb(115,115,115)','rgb(49,130,189)']

    

    mode_size = [8, 8, 10]

    line_size = [2, 2, 3]

    

    x_data = np.vstack((np.array(df['Date']),)*3)

    y_data = np.array([ df['NO'],df['NO2'], df['NOx'] ])

    

    fig = go.Figure()



    for i in range(0, 3):

        fig.add_trace(go.Scatter(x=x_data[i], y=y_data[i], mode='lines',name=labels[i],line=dict(color=colors[i], width=line_size[i]),

        connectgaps=True,))

    

        fig.add_trace(go.Scatter(x=[x_data[i][0], x_data[i][-1]],y=[y_data[i][0], y_data[i][-1]],mode='markers',

        marker=dict(color=colors[i], size=mode_size[i])

    ))

    

    fig.update_layout(xaxis=dict(showline=True,showgrid=False,showticklabels=True,linecolor='rgb(204, 204, 204)',

                      linewidth=2,ticks='outside',tickfont=dict(family='Arial',size=12,color='rgb(82, 82, 82)',),),

                      yaxis=dict(showgrid=False,zeroline=False,showline=False,showticklabels=False,),

                      autosize=False,margin=dict(autoexpand=False,l=100,r=20,t=110,),

                      showlegend=False,plot_bgcolor='white')

    

    annotations = []



    # Adding labels

    for y_trace, label, color in zip(y_data, labels, colors):

        

        # labeling the left_side of the plot

        annotations.append(dict(xref='paper', x=0.05, y=y_trace[0],xanchor='right', yanchor='middle',

                                text=label,font=dict(family='Arial',size=12),

                                showarrow=False))

        

        # labeling the right_side of the plot

        annotations.append(dict(xref='paper', x=0.95, y=y_trace[11],xanchor='left', yanchor='middle',

                                text='',font=dict(family='Arial',size=12),

                                showarrow=False))



        # Title

        annotations.append(dict(xref='paper', yref='paper', x=0.0, y=1.05,xanchor='left', yanchor='middle',

                                text=title, font=dict(family='Arial',size=25,color='rgb(37,37,37)'),

                                showarrow=False))



    fig.update_layout(annotations=annotations, autosize=True)

    fig.update_xaxes(tickangle=45)

    fig.show()

# Function to Plot Various Toxic Gases



def VTG_Plot():

    title = 'Various Toxic Gases [2015-2020]'

    labels = ['Ammonia', 'CarbonMonoOX','SulphurOx','GL Ozone']

    colors = ['rgb(67,67,67)', 'rgb(115,115,115)','rgb(49,130,189)','rgb(189,189,189)']

    

    mode_size = [8, 8, 8,8]

    line_size = [2, 2, 2,2]

    

    x_data = np.vstack((np.array(df['Date']),)*4)

    y_data = np.array([ df['NH3'],df['CO'], df['SO2'], df['O3'] ])

    

    fig = go.Figure()



    for i in range(0, 4):

        fig.add_trace(go.Scatter(x=x_data[i], y=y_data[i], mode='lines',name=labels[i],line=dict(color=colors[i], width=line_size[i]),

        connectgaps=True,))

    

        fig.add_trace(go.Scatter(x=[x_data[i][0], x_data[i][-1]],y=[y_data[i][0], y_data[i][-1]],mode='markers',

        marker=dict(color=colors[i], size=mode_size[i])

    ))

    

    fig.update_layout(xaxis=dict(showline=True,showgrid=False,showticklabels=True,linecolor='rgb(204, 204, 204)',

                      linewidth=2,ticks='outside',tickfont=dict(family='Arial',size=12,color='rgb(82, 82, 82)',),),

                      yaxis=dict(showgrid=False,zeroline=False,showline=False,showticklabels=False,),

                      autosize=False,margin=dict(autoexpand=False,l=100,r=20,t=110,),

                      showlegend=False,plot_bgcolor='white')

    

    annotations = []



    # Adding labels

    for y_trace, label, color in zip(y_data, labels, colors):

        

        # labeling the left_side of the plot

        annotations.append(dict(xref='paper', x=0.05, y=y_trace[0],xanchor='right', yanchor='middle',

                                text=label,font=dict(family='Arial',size=12),

                                showarrow=False))

        

        # labeling the right_side of the plot

        annotations.append(dict(xref='paper', x=0.95, y=y_trace[11],xanchor='left', yanchor='middle',

                                text='',font=dict(family='Arial',size=12),

                                showarrow=False))



        # Title

        annotations.append(dict(xref='paper', yref='paper', x=0.0, y=1.05,xanchor='left', yanchor='middle',

                                text=title, font=dict(family='Arial',size=25,color='rgb(37,37,37)'),

                                showarrow=False))



    fig.update_layout(annotations=annotations, autosize=True)

    fig.update_xaxes(tickangle=45)

    fig.show()

# Function to Plot HydroCarbon Gases



def HC_Plot():

    title = 'Toxic Hydrocarbon Gases [2015-2020]'

    labels = ['Benzene', 'Toluene','Xylene']

    colors = ['rgb(67,67,67)', 'rgb(115,115,115)','rgb(49,130,189)']

    

    mode_size = [8, 8, 8]

    line_size = [2, 2, 2]

    

    x_data = np.vstack((np.array(df['Date']),)*3)

    y_data = np.array([ df['Benzene'],df['Toluene'], df['Xylene'] ])

    

    fig = go.Figure()



    for i in range(0, 3):

        fig.add_trace(go.Scatter(x=x_data[i], y=y_data[i], mode='lines',name=labels[i],line=dict(color=colors[i], width=line_size[i]),

        connectgaps=True,))

    

        fig.add_trace(go.Scatter(x=[x_data[i][0], x_data[i][-1]],y=[y_data[i][0], y_data[i][-1]],mode='markers',

        marker=dict(color=colors[i], size=mode_size[i])

    ))

    

    fig.update_layout(xaxis=dict(showline=True,showgrid=False,showticklabels=True,linecolor='rgb(204, 204, 204)',

                      linewidth=2,ticks='outside',tickfont=dict(family='Arial',size=12,color='rgb(82, 82, 82)',),),

                      yaxis=dict(showgrid=False,zeroline=False,showline=False,showticklabels=False,),

                      autosize=False,margin=dict(autoexpand=False,l=100,r=20,t=110,),

                      showlegend=False,plot_bgcolor='white')

    

    annotations = []



    # Adding labels

    for y_trace, label, color in zip(y_data, labels, colors):

        

        # labeling the left_side of the plot

        annotations.append(dict(xref='paper', x=0.05, y=y_trace[0],xanchor='right', yanchor='middle',

                                text=label,font=dict(family='Arial',size=12),

                                showarrow=False))

        

        # labeling the right_side of the plot

        annotations.append(dict(xref='paper', x=0.95, y=y_trace[11],xanchor='left', yanchor='middle',

                                text='',font=dict(family='Arial',size=12),

                                showarrow=False))



        # Title

        annotations.append(dict(xref='paper', yref='paper', x=0.0, y=1.05,xanchor='left', yanchor='middle',

                                text=title, font=dict(family='Arial',size=25,color='rgb(37,37,37)'),

                                showarrow=False))



    fig.update_layout(annotations=annotations, autosize=True)

    fig.update_xaxes(tickangle=45)

    fig.show()

# Plot for Particulate Matter

PM_Plot()
# Plot for Nitrogen Oxides

NO_Plot()
# Plot for Various Toxic Gases

VTG_Plot()
# Plot for Toxic Hydrocarbon Gas

HC_Plot()
# Seperate dataframe for Delhi with monthly aggregated values

df  = aggData('Delhi')
# Inspect New Dataframe

df.head()
# Plotting Distribution Plot for Delhi

distPlot()
#Plot for Particulate Matter

PM_Plot()
# Plot for Nitrogen Oxides

NO_Plot()
# Plot for Various Toxic Gases

VTG_Plot()
# Plot for HydroCarbon Gases

HC_Plot()
# Seperate dataframe for Kolkata with monthly aggregated values

df  = aggData('Kolkata')
# Inspect the New Dataframe

df.head()
# Plotting Distribution Plot for Kolkata

distPlot()
#Plot for Particulate Matter

PM_Plot()
# Plot for Nitrogen Oxide

NO_Plot()
# Plot for Various Toxic Gases

VTG_Plot()
# Plot for HydroCarbon

HC_Plot()
# Seperate dataframe for Amaravati with monthly aggregated values

df  = aggData('Amaravati')
# Inspect New Dataframe

df.head()
# Plotting Distribution Plot for Amaravati

distPlot()
#Particulate Matter Plot

PM_Plot()
# Nitrogen Oxides Plot

NO_Plot()
# Various Toxic Gases Plot

VTG_Plot()
# HydroCarbon Gases

HC_Plot()
# Seperate dataframe for Amaravati with monthly aggregated values

df  = aggData('Visakhapatnam')
# Inspect 

df.head()
# Plotting Distribution Plot for Visakhapatnam

distPlot()
# Particulate Matter Plot

PM_Plot()
# NItrogen Oxide Plot

NO_Plot()
# Various Toxic Gases

VTG_Plot()
# Hydrocarbon gases

HC_Plot()