# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import plotly.graph_objects as go

from sklearn.preprocessing import StandardScaler

from sklearn.decomposition import PCA



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
dataEarthquakes = pd.read_csv('/kaggle/input/global-significant-earthquake-database-from-2150bc/Worldwide-Earthquake-database.csv')

dataEarthquakes = dataEarthquakes.replace('    ', np.nan) # Some fields are empty string: '    ' it must be replaced by NaN

dataEarthquakes.head()
#dataEarthquakes.info()
# Calculate occurence of earthquake to plot cumulative distribution

earthquakeOccurence = dataEarthquakes[['YEAR','I_D']].groupby('YEAR').count().reset_index()

earthquakeOccurence['percent'] = round(earthquakeOccurence['I_D']/earthquakeOccurence['I_D'].sum()*100,2)

earthquakeOccurence['cumPercent'] = earthquakeOccurence['percent'].cumsum()

#earthquakeOccurence.head()
fig = go.Figure()

fig.add_trace(go.Scatter(x=earthquakeOccurence.YEAR, y=earthquakeOccurence.cumPercent, line=dict(color='rgb(33,125,196)')))

fig.update_layout(plot_bgcolor='white',

                  xaxis=dict(title='Year',ticks="outside", tickwidth=2, tickcolor='grey', ticklen=10,showline=True, linewidth=2, linecolor='grey'),

                  yaxis=dict(title='Percentile [%ile]',ticks="outside", tickwidth=2, tickcolor='grey', ticklen=10,showline=True, linewidth=2, linecolor='grey'),

                  title=dict(text="Cumulative distribution of earthquake events per year", font=dict(family="Verdana",size=25,color="Black")))

fig.update_xaxes(tickfont=dict(family='Verdana', color='grey', size=14), titlefont=dict(family='Verdana', color='black', size=16))

fig.update_yaxes(tickfont=dict(family='Verdana', color='grey', size=14), titlefont=dict(family='Verdana', color='black', size=16))

fig.show()
missingValues = pd.DataFrame(dataEarthquakes['YEAR'])

missingValues['countMissing'] = dataEarthquakes.isna().sum(axis=1)

missingValues = missingValues[['YEAR','countMissing']].groupby('YEAR').sum().reset_index()

missingValues = missingValues.merge(earthquakeOccurence[['YEAR','I_D']], on='YEAR')

missingValues['countMissingPercent'] = round(missingValues['countMissing']/(missingValues['I_D']*47)*100,2)

#missingValues.head()
fig = go.Figure()

fig.add_trace(go.Bar(x=missingValues.YEAR, y=missingValues.countMissingPercent))

fig.update_layout(plot_bgcolor='white',

                  xaxis=dict(title='Year',ticks="outside", tickwidth=2, tickcolor='grey', ticklen=10,showline=True, linewidth=2, linecolor='grey'),

                  yaxis=dict(title='Percentage of missing data [%]',ticks="outside", tickwidth=2, tickcolor='grey', ticklen=10,showline=True, linewidth=2, linecolor='grey'),

                  title=dict(text="Percentage of missing data per year", font=dict(family="Verdana",size=25,color="Black")))

fig.update_xaxes(tickfont=dict(family='Verdana', color='grey', size=14), titlefont=dict(family='Verdana', color='black', size=16))

fig.update_yaxes(tickfont=dict(family='Verdana', color='grey', size=14), titlefont=dict(family='Verdana', color='black', size=16))

fig.show()
fig = go.Figure()

fig.add_trace(go.Bar(x=[(dataEarthquakes['FLAG_TSUNAMI'] == 'Yes').sum(),(dataEarthquakes['FLAG_TSUNAMI'] == 'No').sum()],

                     y=['Yes','No'],

                     text=[str(round((dataEarthquakes['FLAG_TSUNAMI'] == 'Yes').sum()/len(dataEarthquakes)*100,2))+'%',str(round((dataEarthquakes['FLAG_TSUNAMI'] == 'No').sum()/len(dataEarthquakes)*100,2))+'%'],

                     textposition='inside',

                     orientation='h',

                     marker=dict(

        color='rgb(33,125,196)')))

fig.update_layout(plot_bgcolor='white',

                  xaxis=dict(showticklabels=False),

                  title=dict(text="Earthquake triggering a tsunami?", font=dict(family="Verdana",size=25,color="Black")),

                  font=dict(family="Verdana",

                            size=18,

                            color="grey"),

                 margin=dict(b=10),

                 height=300, width=800)

fig.show()
allData = []

color = ['blue','orange']



for y in dataEarthquakes['FLAG_TSUNAMI'].unique():

    data=go.Scattermapbox(lon = dataEarthquakes['LONGITUDE'][dataEarthquakes['FLAG_TSUNAMI'] == y],

                           lat=dataEarthquakes['LATITUDE'][dataEarthquakes['FLAG_TSUNAMI'] == y],

                       text = ['Year: ' + str(list(dataEarthquakes['YEAR'][dataEarthquakes['FLAG_TSUNAMI'] == y])[x]) for x in range(len(dataEarthquakes[dataEarthquakes['FLAG_TSUNAMI'] == y]))],

                       hoverinfo = 'text',

                       mode = 'markers',

                       name= 'No tsunami triggerd' if y == 'No' else 'Tsunami triggered',

                       marker=go.scattermapbox.Marker(size=[5]*len(dataEarthquakes['LATITUDE']),

                                                      sizeref = 1,

                                                      color= ['blue']*len(dataEarthquakes['FLAG_TSUNAMI'] == y) if y == 'Yes' else ['orange']*len(dataEarthquakes['FLAG_TSUNAMI'] == y),

                                                      opacity=0.5))

    allData.append(data)



layout = go.Layout(autosize = True, showlegend=True,

                   title=dict(text="Earthquake localisation per tsunami triggered", font=dict(family="Verdana",size=25,color="Black")),

                   margin=go.layout.Margin(l=50,r=20,b=10,t=50,pad=4),

                   mapbox=go.layout.Mapbox(bearing = 0, # orientation

                                           pitch=0, # inclinaison

                                           zoom=0.6,

                                           style='basic'),

                   mapbox_style="stamen-terrain",

                   width=1050,height=550)

fig = go.Figure(data = allData,layout=layout)

fig.show()
fig = go.Figure()

fig.add_trace(go.Densitymapbox(lat=dataEarthquakes['LATITUDE'][dataEarthquakes['EQ_PRIMARY'].dropna().index],

                               lon=dataEarthquakes['LONGITUDE'][dataEarthquakes['EQ_PRIMARY'].dropna().index],

                               z=dataEarthquakes['EQ_PRIMARY'][dataEarthquakes['EQ_PRIMARY'].dropna().index],

                               radius=5))



fig.update_layout(autosize = True, showlegend=True,

                   title=dict(text="Earthquake localisation by magnitude", font=dict(family="Verdana",size=25,color="Black")),

                   margin=go.layout.Margin(l=50,r=20,b=10,t=50,pad=4),

                   mapbox=go.layout.Mapbox(zoom=0.6,

                                           style='basic'),

                   mapbox_style="stamen-terrain",

                   width=950,height=550)

fig.show()
fig = go.Figure()

fig.add_trace(go.Densitymapbox(lat=dataEarthquakes['LATITUDE'][dataEarthquakes['FOCAL_DEPTH'].dropna().index],

                               lon=dataEarthquakes['LONGITUDE'][dataEarthquakes['FOCAL_DEPTH'].dropna().index],

                               z=dataEarthquakes['FOCAL_DEPTH'][dataEarthquakes['FOCAL_DEPTH'].dropna().index],

                               radius=5))



fig.update_layout(autosize = True, showlegend=True,

                   title=dict(text="Earthquake localisation by depth", font=dict(family="Verdana",size=25,color="Black")),

                   margin=go.layout.Margin(l=50,r=20,b=10,t=50,pad=4),

                   mapbox=go.layout.Mapbox(zoom=0.6,

                                           style='basic'),

                   mapbox_style="stamen-terrain",

                   width=950,height=550)

fig.show()
depthBinSize = 25 # in km

magnitudeBinSize = 0.2

binDepth = list(range(0,700,depthBinSize))

binMagnitude = np.arange(0,10,magnitudeBinSize)

dfMagnitudeDepth = dataEarthquakes[['EQ_PRIMARY','FOCAL_DEPTH']].dropna()

dfMagnitudeDepth['DepthBin'] = np.floor(dfMagnitudeDepth['FOCAL_DEPTH']/depthBinSize)*depthBinSize

dfMagnitudeDepth = dfMagnitudeDepth.groupby(['DepthBin', pd.cut(dfMagnitudeDepth['EQ_PRIMARY'], binMagnitude)]).size().unstack()

# Create matrix with count of all event based on depth bin and magnitude bin

matrixCountEvent = []

for x in range(len(dfMagnitudeDepth.index)):

    matrixCountEvent.append(dfMagnitudeDepth.iloc[x].values.tolist())
fig = go.Figure(data=go.Heatmap(x=binMagnitude,

                                y=binDepth,

                                z=matrixCountEvent,

                                colorscale=[[0,'rgb(255,255,255)'], [0.01,"rgb(170,210,255)"], [1,"rgb(0,30,250)"]]))

fig.update_layout(plot_bgcolor='white', width = 1000,

                  xaxis=dict(title='Magnitude',ticks="outside", tickwidth=2, tickcolor='grey', ticklen=10,showline=True, linewidth=2, linecolor='grey'),

                  yaxis=dict(title='Depth [km]',ticks="outside", tickwidth=2, tickcolor='grey', ticklen=10,showline=True, linewidth=2, linecolor='grey'),

                  title=dict(text="Count earthquake per depth and magnitude", font=dict(family="Verdana",size=25,color="Black")))

fig.update_xaxes(tickfont=dict(family='Verdana', color='grey', size=14), titlefont=dict(family='Verdana', color='black', size=16))

fig.update_yaxes(tickfont=dict(family='Verdana', color='grey', size=14), titlefont=dict(family='Verdana', color='black', size=16))

fig.show()
magnitudeBinSize = 0.2

binDamage = list(range(1,4))

binMagnitude = np.arange(0,10,magnitudeBinSize)

dfMagnitudeDamage = dataEarthquakes[['EQ_PRIMARY','DAMAGE_DESCRIPTION']].dropna()

dfMagnitudeDamage = dfMagnitudeDamage.groupby(['DAMAGE_DESCRIPTION', pd.cut(dfMagnitudeDamage['EQ_PRIMARY'], binMagnitude)]).size().unstack()

# Create matrix with count of all event based on depth bin and magnitude bin

matrixCountEvent = []

for x in range(len(dfMagnitudeDamage.index)):

       matrixCountEvent.append(dfMagnitudeDamage.iloc[x].values.tolist())
fig = go.Figure(data=go.Heatmap(x=binMagnitude,

                                y=binDamage,

                                z=matrixCountEvent,

                                colorscale=[[0,'rgb(255,255,255)'], [0.01,"rgb(170,210,255)"], [1,"rgb(0,30,250)"]]))

fig.update_layout(plot_bgcolor='white', width = 1000,

                  xaxis=dict(title='Magnitude',ticks="outside", tickwidth=2, tickcolor='grey', ticklen=10,showline=True, linewidth=2, linecolor='grey'),

                  yaxis=dict(title='Damage',ticks="outside", tickwidth=2, tickcolor='grey', ticklen=10,showline=True, linewidth=2, linecolor='grey'),

                  title=dict(text="Count earthquake per damage and magnitude", font=dict(family="Verdana",size=25,color="Black")))

fig.update_xaxes(tickfont=dict(family='Verdana', color='grey', size=14), titlefont=dict(family='Verdana', color='black', size=16))

fig.update_yaxes(tickfont=dict(family='Verdana', color='grey', size=14), titlefont=dict(family='Verdana', color='black', size=16))

fig.show()
pcaData = dataEarthquakes[['YEAR','FLAG_TSUNAMI','FOCAL_DEPTH','EQ_PRIMARY','REGION_CODE','DAMAGE_DESCRIPTION']]

pcaData = pcaData.dropna(axis=0).reset_index(drop=True) # Remove NaN

pcaData = pcaData.replace({'FLAG_TSUNAMI' : { 'No' : 0, 'Yes' : 1}}) # Replace categorical values by numerical values

#pcaData
# Standardize the data

scaler = StandardScaler()

pcaData_scaled = scaler.fit_transform(pcaData)

pcaData_scaled



# Create the PCA model and fit standardised data

pca = PCA(n_components=6) # Use the maximum number of component that explains 100% of variance

pca.fit(pcaData_scaled)

pcaData_projected = pca.transform(pcaData_scaled) # for scatter plots
fig = go.Figure()

[fig.add_trace(go.Scatter(x=[0, pca.components_[0,x]],y=[0,pca.components_[1,x]],name=pcaData.columns[x])) for x in range(len(pca.components_[0,:]))]

fig.update_layout(plot_bgcolor='white',height=600, width=600,

                  showlegend=False,

                  shapes=[dict(type="circle",xref="x",yref="y",x0=-1,y0=-1,x1=1,y1=1,line_color="LightSeaGreen",)],

                  xaxis=dict(ticks="outside", tickwidth=2, tickcolor='grey', ticklen=10,showline=True, linewidth=2, linecolor='grey'),

                  yaxis=dict(ticks="outside", tickwidth=2, tickcolor='grey', ticklen=10,showline=True, linewidth=2, linecolor='grey'),

                  title=dict(text="Principal Component Analysis", font=dict(family="Verdana",size=25,color="Black")))

fig.update_xaxes(title=dict(text='PC1', font=dict(size=18)),showgrid=True, linecolor='black', ticks='outside')

fig.update_yaxes(title=dict(text='PC2', font=dict(size=18)),showgrid=True, linecolor='black', ticks='outside')

fig.show()
fig = go.Figure()

fig.add_trace(go.Scatter(x=[pcaData_projected[x][0] for x in range(len(pcaData_projected)) if pcaData['FLAG_TSUNAMI'][x]==0],y=[pcaData_projected[x][1] for x in range(len(pcaData_projected)) if pcaData['FLAG_TSUNAMI'][x]==0], name='No tsunami triggered', mode='markers'))

fig.add_trace(go.Scatter(x=[pcaData_projected[x][0] for x in range(len(pcaData_projected)) if pcaData['FLAG_TSUNAMI'][x]==1],y=[pcaData_projected[x][1] for x in range(len(pcaData_projected)) if pcaData['FLAG_TSUNAMI'][x]==1], name='Tsunami triggered', mode='markers'))

fig.update_layout(plot_bgcolor='white',height=500, width=1000, showlegend=True,

                  xaxis=dict(ticks="outside", tickwidth=2, tickcolor='grey', ticklen=10,showline=True, linewidth=2, linecolor='grey'),

                  yaxis=dict(ticks="outside", tickwidth=2, tickcolor='grey', ticklen=10,showline=True, linewidth=2, linecolor='grey'),

                  title=dict(text="Tsunami triggered with PCA", font=dict(family="Verdana",size=25,color="Black")))

fig.update_xaxes(title=dict(text='PC1', font=dict(size=18)),showgrid=True, linecolor='black', ticks='outside')

fig.update_yaxes(title=dict(text='PC2', font=dict(size=18)),showgrid=True, linecolor='black', ticks='outside')

fig.show()