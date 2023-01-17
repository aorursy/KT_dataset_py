import pandas as pd 
import numpy as np 
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
import plotly.graph_objs as go
#import cufflinks as cf
#cf.go_offline()
init_notebook_mode(connected=True)
df = pd.read_csv('../input/CO_level_2000_-.csv')
del df['Unnamed: 0']
df.head()
df = df.groupby(['Address','State','County','City','Date']).mean().reset_index()
df.info()
df.describe()
df['State'].unique()
df = df[~df['State'].isin(['Puerto Rico', 'Country Of Mexico'])]
tempYear = []
tempMonth = []
totalTuples = df.count()['State']
for i in range(totalTuples):
    delement = (df['Date'].iloc[i]).split('-')
    tempYear.append(int(delement[0]))
    tempMonth.append(delement[0]+'-'+delement[1])
df['Year'] = tempYear
df['Month'] = tempMonth
df.head()
stateData = {}
addrDict = {}
for i in df['State'].unique():
    #create a dicionary of data frames for state-wise record
    stateData[i] = df[df['State'] == i]
    addrDict[i] = stateData[i]['Address'].nunique()
addrdf = pd.DataFrame.from_dict(addrDict, orient = 'index', columns = ['Address Count']).reset_index().rename(columns = {'index' : 'State'})

# addrdf
addrdf.head()
data = go.Bar(x = addrdf['State'], 
              y = addrdf['Address Count'], 
              text = addrdf['State'])
layout = go.Layout(dict(title = 'Number of unique addresses per State', 
                        xaxis = dict(title = 'State'),
                        yaxis = dict(title = 'Count')))
fig = dict(data = [data],layout = layout)
iplot(fig)

#addrdf.iplot(kind='bar', x='State', y='Address Count', title='Number of unique addresses per State (Zoom in or hover over)', orientation='h')

yeardf = df.groupby('Year').count().reset_index()

data = go.Bar(x = yeardf['Year'], 
              y = yeardf['Address'], 
              text = yeardf['Year'])
layout = go.Layout(dict(title = 'Number of records per year', 
                        xaxis = dict(title='Year'),
                        yaxis = dict(title = 'Count')))
fig = dict(data=[data],layout = layout)
iplot(fig)

# df.groupby('Year').count().reset_index().iplot(kind='bar', x='Year', y='Address', title='Number of records per year')
df[df['Year'] == 2018]['Month'].unique()
datesAddr = ['3847 W EARLL DR-WEST PHOENIX STATION', '6767 Ojo De Agua', 'LFC #1-LAS FLORES CANYON', 'NW Corner Interstate 10 & Etiwanda Ave', '700 North Bullis Road', '10 S. 11th St/ Evansville- Lloyd', '1301 E. 9TH ST.', '4113 SHUTTLESWORTH DRIVE']
datesState = ['Arizona', 'Texas', 'California', 'California', 'California', 'Indiana', 'Ohio', 'Alabama']
datesStart = []
datesEnd = []
for i in range(len(datesAddr)):
    datesStart.append(df[df['Address'] == datesAddr[i]]['Date'].min())
    datesEnd.append(df[df['Address'] == datesAddr[i]]['Date'].max())
    datesAddr[i] += ', '+datesState[i]
datesDF = pd.DataFrame([datesAddr, datesStart, datesEnd],index=['Address','Start date','Last Date']).transpose()
datesDF#.head()
maximumYear = df[['Year','Arithmetic Mean']]
maximumYear = maximumYear.groupby('Year').max().reset_index()

maxTable = pd.DataFrame()
for i in range(19):
    x = maximumYear.iloc[i]['Arithmetic Mean']
    record = df[df['Year'] == (2000 + i)]
    record = record[record['Arithmetic Mean'] == x].head(1) # pick only one
    maxTable = maxTable.append(record)
maxTable = maxTable[['Address','State','Arithmetic Mean','Month']]
maxTable
# lets extract minimum records year - wise. Same logic upside - down.
minimumYear = df[['Year','Arithmetic Mean']]
# neglect 0 and negative values
minimumYear = minimumYear[minimumYear['Arithmetic Mean'] > 0]
minimumYear = minimumYear.groupby('Year').min().reset_index()

minTable = pd.DataFrame()
for i in range(19):
    x = minimumYear.iloc[i]['Arithmetic Mean']
    record = df[df['Year'] == (2000 + i)]
    record = record[record['Arithmetic Mean'] == x].head(1) # pick one record
    minTable = minTable.append(record)
minTable = minTable[['Address','State','Arithmetic Mean','Month']]
minTable
addr = '3847 W EARLL DR-WEST PHOENIX STATION'
addr
addrdf = stateData['Arizona'][stateData['Arizona']['Address'] == addr]

tempdf = addrdf.groupby('Month').count().reset_index()
data = go.Bar(x = tempdf['Month'], 
              y = tempdf['Address'], 
              text = tempdf['Month'])
layout = go.Layout(dict(title = 'Number of records for selected address month wise', 
                        xaxis = dict(title = 'Year'),
                        yaxis = dict(title = 'Count')))
fig = dict(data = [data],layout = layout)
iplot(fig)

#addrdf.groupby('Month').count().reset_index().iplot(kind='bar', x='Month', y='Address', title='Number of records for selected address month wise')
seasonrange = ['2016-12', '2017-01', '2017-02', '2017-03', '2017-04', '2017-05', '2017-06', '2017-07', '2017-08', '2017-09', '2017-10', '2017-11', '2017-12']

for i in range(len(seasonrange)):
    tempdf = addrdf[addrdf['Month'] == seasonrange[i]]
    data = go.Scatter(x = tempdf['Date'], 
                           y = tempdf['Arithmetic Mean'],
                           text = tempdf['Date'], 
                           mode = 'lines+markers', 
                           name = seasonrange[i])
    fig = dict(data = [data],layout = go.Layout(dict(title = seasonrange[i], 
                                                 xaxis = dict(title = 'Days'), 
                                                yaxis = dict(range = [0, 2], title = 'CO (in ppm)'))))
    iplot(fig)

#for i in range(len(seasonrange)):
#    addrdf[addrdf['Month']==seasonrange[i]].iplot(x='Date',y='Arithmetic Mean',layout=go.Layout(yaxis=dict(range=[0,2]),title=seasonrange[i]))
winterdf = addrdf[addrdf['Month'].isin(['2016-12', '2017-01', '2017-02'])][['Arithmetic Mean', '1st Max Hour']]
springdf = addrdf[addrdf['Month'].isin(['2017-03', '2017-04', '2017-05'])][['Arithmetic Mean', '1st Max Hour']]
summerdf = addrdf[addrdf['Month'].isin(['2017-06', '2017-07', '2017-08'])][['Arithmetic Mean', '1st Max Hour']]
autumndf = addrdf[addrdf['Month'].isin(['2017-09', '2017-10', '2017-11'])][['Arithmetic Mean', '1st Max Hour']]
seasondf = [winterdf, springdf, summerdf, autumndf]
dftext = ['Winter', 'Spring', 'Summer', 'Autumn']

data = []
for i in range(len(seasondf)):
    data.append(go.Box(y = seasondf[i]['Arithmetic Mean'], 
                  name = dftext[i]))
layout = go.Layout(title = 'Distribution of Arithmetic Mean across different season (2016-12 to 2017-11)', 
                   xaxis = dict(title = 'Season'), 
                   yaxis = dict(title = 'CO (in ppm)'))
fig = dict(data = data, layout = layout)
iplot(fig)

#pd.concat([winterdf['Arithmetic Mean'], springdf['Arithmetic Mean'], summerdf['Arithmetic Mean'], autumndf['Arithmetic Mean']], axis=1, keys=['Winter','Spring','Summer','Autumn']).iplot(kind='box')
data = []
for i in range(len(seasondf)):
    data.append(go.Box(y = seasondf[i]['1st Max Hour'], 
                  name = dftext[i]))
layout = go.Layout(title = 'Distribution of Hour values at which maximum reading was taken', 
                   xaxis = dict(title = 'Season'), 
                   yaxis = dict(title = 'Hours'))
fig = dict(data = data, layout = layout)
iplot(fig)

#pd.concat([winterdf['1st Max Hour'], springdf['1st Max Hour'], summerdf['1st Max Hour'], autumndf['1st Max Hour']], axis=1, keys=['Winter','Spring','Summer','Autumn']).iplot(kind='box')
winterdf = df[df['Month'].isin(['2016-12', '2017-01', '2017-02'])][['Arithmetic Mean', 'State']]
springdf = df[df['Month'].isin(['2017-03', '2017-04', '2017-05'])][['Arithmetic Mean', 'State']]
summerdf = df[df['Month'].isin(['2017-06', '2017-07', '2017-08'])][['Arithmetic Mean', 'State']]
autumndf = df[df['Month'].isin(['2017-09', '2017-10', '2017-11'])][['Arithmetic Mean', 'State']]

# group by and sort by State to map it easily ahead
winterdf = winterdf.groupby('State').mean().reset_index().sort_values('State')
springdf = springdf.groupby('State').mean().reset_index().sort_values('State')
summerdf = summerdf.groupby('State').mean().reset_index().sort_values('State')
autumndf = autumndf.groupby('State').mean().reset_index().sort_values('State')
                                                                      
abbState = ['US State:', 'Alabama', 'Alaska', 'Arizona', 'Arkansas', 'California', 'Colorado', 'Connecticut', 'Delaware', 'Florida', 'Georgia', 'Hawaii', 'Idaho', 'Illinois', 'Indiana', 'Iowa', 'Kansas', 'Kentucky', 'Louisiana', 'Maine', 'Maryland', 'Massachusetts', 'Michigan', 'Minnesota', 'Mississippi', 'Missouri', 'Montana', 'Nebraska', 'Nevada', 'New Hampshire', 'New Jersey', 'New Mexico', 'New York', 'North Carolina', 'North Dakota', 'Ohio', 'Oklahoma', 'Oregon', 'Pennsylvania', 'Rhode Island', 'South Carolina', 'South Dakota', 'Tennessee', 'Texas', 'Utah', 'Vermont', 'Virginia', 'Washington', 'West Virginia', 'Wisconsin', 'Wyoming', 'Commonwealth/Territory:', 'American Samoa', 'District Of Columbia', 'Federated States of Micronesia', 'Guam', 'Marshall Islands', 'Northern Mariana Islands', 'Palau', 'Puerto Rico', 'Virgin Islands', 'Military "State":', 'Armed Forces Africa', 'Armed Forces Americas', 'Armed Forces Canada', 'Armed Forces Europe', 'Armed Forces Middle East', 'Armed Forces Pacific']
abbAB = ['Abbreviation:', 'AL', 'AK', 'AZ', 'AR', 'CA', 'CO', 'CT', 'DE', 'FL', 'GA', 'HI', 'ID', 'IL', 'IN', 'IA', 'KS', 'KY', 'LA', 'ME', 'MD', 'MA', 'MI', 'MN', 'MS', 'MO', 'MT', 'NE', 'NV', 'NH', 'NJ', 'NM', 'NY', 'NC', 'ND', 'OH', 'OK', 'OR', 'PA', 'RI', 'SC', 'SD', 'TN', 'TX', 'UT', 'VT', 'VA', 'WA', 'WV', 'WI', 'WY', 'Abbreviation:', 'AS', 'DC', 'FM', 'GU', 'MH', 'MP', 'PW', 'PR', 'VI', 'Abbreviation:', 'AE', 'AA', 'AE', 'AE', 'AE', 'AP']
abbDF = pd.DataFrame([abbState,abbAB]).transpose()

#small correction, so things go smooth ahead
abbDF.iloc[53][0] = 'District Of Columbia'

# creating label to display when hovered over
mapA = []
mapS = []
for i in winterdf.index:
    mapA.append(str(winterdf['Arithmetic Mean'].iloc[i])[:5]+' ppm')
    mapS.append(abbDF[abbDF[0] == winterdf['State'].iloc[i]][1].values[0])
winterdf['text'] = mapA
winterdf['code'] = mapS

mapA = []
mapS = []
for i in springdf.index:
    mapA.append(str(springdf['Arithmetic Mean'].iloc[i])[:5]+' ppm')
    mapS.append(abbDF[abbDF[0] == springdf['State'].iloc[i]][1].values[0])
springdf['text'] = mapA
springdf['code'] = mapS

mapA = []
mapS = []
for i in summerdf.index:
    mapA.append(str(summerdf['Arithmetic Mean'].iloc[i])[:5]+' ppm')
    mapS.append(abbDF[abbDF[0] == summerdf['State'].iloc[i]][1].values[0])
summerdf['text'] = mapA
summerdf['code'] = mapS

mapA = []
mapS = []
for i in autumndf.index:
    mapA.append(str(autumndf['Arithmetic Mean'].iloc[i])[:5]+' ppm')
    mapS.append(abbDF[abbDF[0] == autumndf['State'].iloc[i]][1].values[0])
autumndf['text'] = mapA
autumndf['code'] = mapS
data = dict(type='choropleth',
            locations = winterdf['code'],
            z = winterdf['Arithmetic Mean'],
            locationmode = 'USA-states',
            text = winterdf['text'],
            marker = dict(line = dict(color = 'rgb(255,255,255)',width = 2)),
            colorbar = {'title':"CO Mean in ppm"}) 
layout = dict(title = 'Arithmetic Mean Value in Winter by State',
              geo = dict(scope = 'usa',
                         showlakes = True,
                         lakecolor = 'rgb(85,173,240)'))
choromap = go.Figure(data = [data],layout = layout)
iplot(choromap)
data = dict(type='choropleth',
            locations = springdf['code'],
            z = springdf['Arithmetic Mean'],
            locationmode = 'USA-states',
            text = springdf['text'],
            marker = dict(line = dict(color = 'rgb(255,255,255)',width = 2)),
            colorbar = {'title':"CO Mean in ppm"}) 
layout = dict(title = 'CO Mean Value in Spring by State',
              geo = dict(scope = 'usa',
                         showlakes = True,
                         lakecolor = 'rgb(85,173,240)'))
choromap = go.Figure(data = [data],layout = layout)
iplot(choromap)
data = dict(type='choropleth',
            locations = summerdf['code'],
            z = summerdf['Arithmetic Mean'],
            locationmode = 'USA-states',
            text = summerdf['text'],
            marker = dict(line = dict(color = 'rgb(255,255,255)',width = 2)),
            colorbar = {'title':"CO Mean in ppm"}) 
layout = dict(title = 'CO Mean Value in Summer by State',
              geo = dict(scope = 'usa',
                         showlakes = True,
                         lakecolor = 'rgb(85,173,240)'))
choromap = go.Figure(data = [data],layout = layout)
iplot(choromap)
data = dict(type='choropleth',
            locations = autumndf['code'],
            z = autumndf['Arithmetic Mean'],
            locationmode = 'USA-states',
            text = autumndf['text'],
            marker = dict(line = dict(color = 'rgb(255,255,255)',width = 2)),
            colorbar = {'title':"CO Mean in ppm"}) 
layout = dict(title = 'CO Mean Value in Autumn by State',
              geo = dict(scope = 'usa',
                         showlakes = True,
                         lakecolor = 'rgb(85,173,240)'))
choromap = go.Figure(data = [data],layout = layout)
iplot(choromap)
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
chosenAddress = addrdf[['Month','Arithmetic Mean']]
#aggregate them
chosenAddress = chosenAddress.groupby('Month').mean().reset_index().reset_index()
# we will use monthID on X-axis such that the first month in record will have monthID = 0
chosenAddress = chosenAddress.rename(columns = {'index':'monthID'})
start = addrdf['Month'].min()
# start is first month in record of given address is of form '2000-01‘
# tofind is input feature for which the CO Mean values is to be predicted is of form '2018-11‘
def toID(tofind,start = start):
    startY = int(start.split('-')[0])
    startM = int(start.split('-')[1])
    tofindY = int(tofind.split('-')[0])
    tofindM = int(tofind.split('-')[1])
    id = 12 - startM
    id += ((tofindY - startY) - 1 ) * 12
    id += tofindM
    return id
# init our model
lm = LinearRegression()
# lets get data ready
X = chosenAddress[['monthID']] #feature
Y = chosenAddress[['Arithmetic Mean']] #label
# split train and test data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=101)
# train our model
lm.fit(X_train,Y_train)
print("Intercept is "+str(lm.intercept_))
print("Coefficient is "+str(lm.coef_))
# predictions (Y) for trained data
line = X_train['monthID'] * lm.coef_[0] + lm.intercept_[0]

# display values
#annotation = go.Annotation(x = 3.5, y = 3, text = '$R^2 = 0.9551,\\Y = 0.716X + 19.18$',  showarrow = False, font = go.Font(size=16))

# actual points
train = go.Scatter(x = X_train['monthID'],
                   y = Y_train['Arithmetic Mean'],
                   mode = 'markers',
                   marker = dict(color = 'rgb(255, 127, 14)'),
                   name = 'Data')

# fitted line
fit = go.Scatter(x = X_train['monthID'],
                 y = line,
                 mode = 'lines',
                 marker = dict(color = 'rgb(31, 119, 180)'),
                 name = 'Fit')

layout = go.Layout(title = 'Linear Fit Model',
                   xaxis = dict(title = 'Month ID'),
                   yaxis = dict(title = 'CO (in ppm)'))
                   
data = [train, fit]

fig = go.Figure(data=data, layout=layout)

iplot(fig)
predictions = lm.predict(X_test)
data = go.Histogram(x = (Y_test - predictions)['Arithmetic Mean'],
                    xbins = dict(start = -6, end = 6, size = 0.1))
layout = go.Layout(xaxis = dict(title = 'Error'))

fig = dict(data = [data], layout = layout)
iplot(fig)

#(Y_test - predictions).iplot(kind='hist', bins=10)
inMonth = '2018-08'
print('In month '+inMonth+', predicted value of CO Mean is '+str(lm.predict([[toID(inMonth)]])[0][0])[:5])

