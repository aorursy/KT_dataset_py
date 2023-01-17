# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
df = pd.read_csv('../input/pollution_us_2000_2016.csv')

df.head()
codf = df[['Address','State','County','City','Date Local','CO Units','CO Mean','CO 1st Max Value','CO 1st Max Hour','CO AQI']]

codf.head()
codf = codf.groupby(['Address','State','County','City','Date Local']).mean().reset_index()

codf.head()
codf['State'].unique()
codf = codf[codf['State']!='Country Of Mexico']

# get our total number

totalTuples = codf.count()['State']

#totalTuples

# add year and month columns

tempYear = []

tempMonth = []

for i in range(totalTuples):

    delement = (codf['Date Local'].iloc[i]).split('-')

    tempYear.append(int(delement[0]))

    tempMonth.append(delement[0]+'-'+delement[1])

codf['Year'] = tempYear

codf['Month'] = tempMonth

codf.head()
stateData = {}

addrlabel = []

acountlabel = []

for i in codf['State'].unique():

    #create a dicionary of data frames for state-wise record

    stateData[i] = codf[codf['State'] == i]

    addrlabel.append(i)

    acountlabel.append(stateData[i]['Address'].nunique())

aCountdf=pd.DataFrame(addrlabel,columns=['State'])

aCountdf['Address Count'] = acountlabel

aCountdf.head()
plt.figure(figsize=(13,8))

splot = sns.barplot(y='State',x='Address Count',data=aCountdf,estimator=sum)

for p in splot.patches:

        splot.annotate(p.get_width(), (p.get_width(),p.get_y()+0.5), ha='center', va='center', xytext=(13,0), textcoords='offset points')  
datesAddr = ['1415 Hinton Street','14306 PARK AVE., VICTORVILLE, CA','NO. B\'HAM,SOU R.R., 3009 28TH ST. NO.','2 YARMOUTH ROAD, RG&E Substation','200TH STREET AND SOUTHERN BOULDVARD Pfizer Lab']

datesState = ['Texas','California','Alabama','New York','New York']

datesStart = []

datesEnd = []

for i in range(5):

    datesStart.append(codf[codf['Address'] == datesAddr[i]]['Date Local'].min())

    datesEnd.append(codf[codf['Address'] == datesAddr[i]]['Date Local'].max())

    datesAddr[i] += ', '+datesState[i]

datesDF = pd.DataFrame([datesAddr, datesStart, datesEnd],index=['Address','Start date','Last Date']).transpose()

datesDF.head()
plt.figure(figsize=(18,9))

splot = sns.countplot(data=codf,x='Year')

for p in splot.patches:

        splot.annotate(p.get_height(), (p.get_x() + p.get_width() / 2., p.get_height()), ha='center', va='center', rotation=0, xytext=(0, 10), textcoords='offset points') 
maximumYear = codf[['Year','CO Mean']]

maximumYear = maximumYear.groupby('Year').max().reset_index()

x = maximumYear.iloc[0]['CO Mean']

record = codf[codf['Year'] == (2000)]

maxTable = record[record['CO Mean'] == x]

for i in range(1,17):

    x = maximumYear.iloc[i]['CO Mean']

    record = codf[codf['Year'] == (2000 + i)]

    maxTable = maxTable.append(record[record['CO Mean'] == x])

    #record

maxTable = maxTable[['Address','State','CO Mean','Month']]

maxTable
minimumYear = codf[['Year','CO Mean']]

# neglect 0 and negative values

minimumYear = minimumYear[minimumYear['CO Mean'] > 0]

minimumYear = minimumYear.groupby('Year').min().reset_index()



x = minimumYear.iloc[0]['CO Mean']

record = codf[codf['Year'] == (2000)].head(1) # pick one record

minTable = record[record['CO Mean'] == x]

for i in range(1,17):

    x = minimumYear.iloc[i]['CO Mean']

    record = codf[codf['Year'] == (2000 + i)]

    record = record[record['CO Mean'] == x].head(1) # pick one record

    minTable = minTable.append(record)

    #record

minTable = minTable[['Address','State','CO Mean','Month']]

minTable
addr = '14306 PARK AVE., VICTORVILLE, CA'

addr
cal1 = stateData['California'][stateData['California']['Address'] == addr]

plt.figure(figsize=(75,16))

splot = sns.countplot(x='Month',data=cal1)
tempcal1 = stateData['California'][stateData['California']['Address'] == addr]

tempcal1 = tempcal1[tempcal1['Year'] == 2015]

plt.figure(figsize=(16,7))

splot = sns.countplot(x='Month',data=tempcal1)

for p in splot.patches:

        splot.annotate(p.get_height(), (p.get_x() + p.get_width() / 2., p.get_height()), ha='center', va='center', rotation=0, xytext=(0, 10), textcoords='offset points') 
tempcal = cal1[cal1['Month'] == '2014-12']

tempcal['Day'] = list(range(1,32))

plt.figure(figsize=(12,3))

splot = sns.pointplot(data = tempcal, y = 'CO Mean', x = 'Day')

splot.set_ylim(0,1.5)

tempcal = cal1[cal1['Month'] == '2015-01']

tempcal['Day'] = list(range(1,32))

plt.figure(figsize=(12,3))

splot = sns.pointplot(data = tempcal, y = 'CO Mean', x = 'Day')

splot.set_ylim(0,1.5)
tempcal = cal1[cal1['Month'] == '2015-02']

tempcal['Day'] = list(range(1,29))

plt.figure(figsize=(12,3))

splot = sns.pointplot(data = tempcal, y = 'CO Mean', x = 'Day')

splot.set_ylim(0,1.5)
tempcal = cal1[cal1['Month'] == '2015-03']

tempcal['Day'] = list(range(1,32))

plt.figure(figsize=(12,3))

splot = sns.pointplot(data = tempcal, y = 'CO Mean', x = 'Day')

splot.set_ylim(0,1.5)
tempcal = cal1[cal1['Month'] == '2015-04'][['Date Local','CO Mean']]

tempApril = tempcal['CO Mean'].mean()

aprilDF = pd.DataFrame([['2015-04-07','2015-04-08','2015-04-09','2015-04-10','2015-04-11','2015-04-12'],[tempApril,tempApril,tempApril,tempApril,tempApril,tempApril]],

                      ['Date Local','CO Mean']).transpose()



tempcal = pd.concat([tempcal,aprilDF]).sort_values('Date Local')

tempcal['Day'] = list(range(1,31))

plt.figure(figsize=(12,3))

splot = sns.pointplot(data = tempcal, y = 'CO Mean', x = 'Day')

splot.set_ylim(0,1.5)
tempcal = cal1[cal1['Month'] == '2015-05']

tempcal['Day'] = list(range(1,32))

plt.figure(figsize=(12,3))

splot = sns.pointplot(data = tempcal, y = 'CO Mean', x = 'Day')

splot.set_ylim(0,1.5)
tempcal = cal1[cal1['Month'] == '2015-06']

tempcal['Day'] = list(range(1,31))

plt.figure(figsize=(12,3))

splot = sns.pointplot(data = tempcal, y = 'CO Mean', x = 'Day')

splot.set_ylim(0,1.5)
tempcal = cal1[cal1['Month'] == '2015-07']

tempcal['Day'] = list(range(1,32))

plt.figure(figsize=(12,3))

splot = sns.pointplot(data = tempcal, y = 'CO Mean', x = 'Day')

splot.set_ylim(0,1.5)
tempcal = cal1[cal1['Month'] == '2015-08']

tempcal['Day'] = list(range(1,32))

plt.figure(figsize=(12,3))

splot = sns.pointplot(data = tempcal, y = 'CO Mean', x = 'Day')

splot.set_ylim(0,1.5)
tempcal = cal1[cal1['Month'] == '2015-09']

tempcal['Day'] = list(range(1,31))

plt.figure(figsize=(12,3))

splot = sns.pointplot(data = tempcal, y = 'CO Mean', x = 'Day')

splot.set_ylim(0,1.5)
tempcal = cal1[cal1['Month'] == '2015-10']

tempcal['Day'] = list(range(1,32))

plt.figure(figsize=(12,3))

splot = sns.pointplot(data = tempcal, y = 'CO Mean', x = 'Day')

splot.set_ylim(0,1.5)
tempcal = cal1[cal1['Month'] == '2015-11']

tempcal['Day'] = list(range(1,31))

plt.figure(figsize=(12,3))

splot = sns.pointplot(data = tempcal, y = 'CO Mean', x = 'Day')

splot.set_ylim(0,1.5)
tempcal = cal1[cal1['Month'] == '2015-12']

tempcal['Day'] = list(range(1,32))

plt.figure(figsize=(12,3))

splot = sns.pointplot(data = tempcal, y = 'CO Mean', x = 'Day')

splot.set_ylim(0,1.5)
#tempcal1 = pd.DataFrame(pd.concat([cal1[cal1['Month'] == '2014-12'],cal1[cal1['Month'] == '2015-01'],cal1[cal1['Month'] == '2015-02']])['CO Mean'],columns=['CO Mean'])

#tempcal2 = pd.DataFrame(pd.concat([cal1[cal1['Month'] == '2015-03'],cal1[cal1['Month'] == '2015-04'],cal1[cal1['Month'] == '2015-05']])['CO Mean'],columns=['CO Mean'])

#tempcal3 = pd.DataFrame(pd.concat([cal1[cal1['Month'] == '2015-06'],cal1[cal1['Month'] == '2015-07'],cal1[cal1['Month'] == '2015-08']])['CO Mean'],columns=['CO Mean'])

#tempcal4 = pd.DataFrame(pd.concat([cal1[cal1['Month'] == '2015-09'],cal1[cal1['Month'] == '2015-10'],cal1[cal1['Month'] == '2015-11']])['CO Mean'],columns=['CO Mean'])

# optimized 



tempcal1 = cal1[cal1['Month'].isin(['2014-12', '2015-01', '2015-03'])][['CO Mean']]

tempcal2 = cal1[cal1['Month'].isin(['2015-03', '2015-04', '2015-05'])][['CO Mean']]

tempcal3 = cal1[cal1['Month'].isin(['2015-06', '2015-07', '2015-08'])][['CO Mean']]

tempcal4 = cal1[cal1['Month'].isin(['2015-09', '2015-10', '2015-12'])][['CO Mean']]



tempcal1['Season'] = ['Winter'] * tempcal1.count()['CO Mean']

tempcal2['Season'] = ['Spring'] * tempcal2.count()['CO Mean']

tempcal3['Season'] = ['Summer'] * tempcal3.count()['CO Mean']

tempcal4['Season'] = ['Autumn'] * tempcal4.count()['CO Mean']



tempcal = pd.concat([tempcal1,tempcal2,tempcal3,tempcal4])#.sort_values('Season')#['Season']#.count()



plt.figure(figsize=(8,5))

splot = sns.boxplot(data=tempcal,x='Season',y='CO Mean')

splot.set_ylim(0,1)



# medians = tempcal.groupby(['Season'])['CO Mean'].median()#.values

# group by sorts by season. and so, actual sequence in which seasons occur is lost.



medians = [tempcal1['CO Mean'].median(),tempcal2['CO Mean'].median(),tempcal3['CO Mean'].median(),tempcal4['CO Mean'].median()]

median_labels = [str(np.round(s, 2)) for s in medians]



pos = range(len(medians))

for tick,label in zip(pos,splot.get_xticklabels()):

    splot.text(pos[tick], medians[tick] + 0.01 , median_labels[tick], horizontalalignment='center', size='large', color='w', weight='semibold')

#it is high in winter because of inversion. Check -->> https://en.wikipedia.org/wiki/Inversion_(meteorology)    
#tempcal1 = pd.DataFrame(pd.concat([cal1[cal1['Month'] == '2014-12'],cal1[cal1['Month'] == '2015-01'],cal1[cal1['Month'] == '2015-02']])['CO 1st Max Hour'],columns=['CO 1st Max Hour'])

#tempcal2 = pd.DataFrame(pd.concat([cal1[cal1['Month'] == '2015-03'],cal1[cal1['Month'] == '2015-04'],cal1[cal1['Month'] == '2015-05']])['CO 1st Max Hour'],columns=['CO 1st Max Hour'])

#tempcal3 = pd.DataFrame(pd.concat([cal1[cal1['Month'] == '2015-06'],cal1[cal1['Month'] == '2015-07'],cal1[cal1['Month'] == '2015-08']])['CO 1st Max Hour'],columns=['CO 1st Max Hour'])

#tempcal4 = pd.DataFrame(pd.concat([cal1[cal1['Month'] == '2015-09'],cal1[cal1['Month'] == '2015-10'],cal1[cal1['Month'] == '2015-11']])['CO 1st Max Hour'],columns=['CO 1st Max Hour'])



tempcal1 = cal1[cal1['Month'].isin(['2014-12', '2015-01', '2015-03'])][['CO 1st Max Hour']]

tempcal2 = cal1[cal1['Month'].isin(['2015-03', '2015-04', '2015-05'])][['CO 1st Max Hour']]

tempcal3 = cal1[cal1['Month'].isin(['2015-06', '2015-07', '2015-08'])][['CO 1st Max Hour']]

tempcal4 = cal1[cal1['Month'].isin(['2015-09', '2015-10', '2015-12'])][['CO 1st Max Hour']]



tempcal1['Season'] = ['Winter'] * tempcal1.count()['CO 1st Max Hour']

tempcal2['Season'] = ['Spring'] * tempcal2.count()['CO 1st Max Hour']

tempcal3['Season'] = ['Summer'] * tempcal3.count()['CO 1st Max Hour']

tempcal4['Season'] = ['Autumn'] * tempcal4.count()['CO 1st Max Hour']



tempcal = pd.concat([tempcal1,tempcal2,tempcal3,tempcal4])#.sort_values('Season')



plt.figure(figsize=(8,5))

splot = sns.boxplot(data=tempcal,x='Season',y='CO 1st Max Hour')

splot.set_ylim(0,24)



# medians = tempcal.groupby(['Season'])['CO 1st Max Hour'].median()#.values

# group by sorts by season. and so, actual sequence in which seasons occur is lost.



medians = [tempcal1['CO 1st Max Hour'].median(),tempcal2['CO 1st Max Hour'].median(),tempcal3['CO 1st Max Hour'].median(),tempcal4['CO 1st Max Hour'].median()]

median_labels = [str(np.round(s, 2)) for s in medians]



pos = range(len(medians))

for tick,label in zip(pos,splot.get_xticklabels()):

    splot.text(pos[tick], medians[tick] + 0.5 , median_labels[tick], horizontalalignment='center', size='large', color='w', weight='semibold')

    
import plotly.plotly as py

import plotly.graph_objs as go 

from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

init_notebook_mode(connected=True) 
mapData = codf[(codf['Month'] >= '2014-12') & (codf['Month'] <= '2015-11')]



#winterdf = pd.concat([mapData[mapData['Month'] == '2014-12'],mapData[mapData['Month'] == '2015-01'],mapData[mapData['Month'] == '2015-02']])[['State','CO Mean']].groupby('State').mean().reset_index().sort_values('State')

#springdf = pd.concat([mapData[mapData['Month'] == '2015-03'],mapData[mapData['Month'] == '2015-04'],mapData[mapData['Month'] == '2015-05']])[['State','CO Mean']].groupby('State').mean().reset_index().sort_values('State')

#summerdf = pd.concat([mapData[mapData['Month'] == '2015-06'],mapData[mapData['Month'] == '2015-07'],mapData[mapData['Month'] == '2015-08']])[['State','CO Mean']].groupby('State').mean().reset_index().sort_values('State')

#autumndf = pd.concat([mapData[mapData['Month'] == '2015-09'],mapData[mapData['Month'] == '2015-10'],mapData[mapData['Month'] == '2015-11']])[['State','CO Mean']].groupby('State').mean().reset_index().sort_values('State')



winterdf = mapData[mapData['Month'].isin(['2014-12', '2015-01', '2015-02'])].groupby('State').mean().reset_index().sort_values('State')

springdf = mapData[mapData['Month'].isin(['2015-03', '2015-04', '2015-05'])].groupby('State').mean().reset_index().sort_values('State')

summerdf = mapData[mapData['Month'].isin(['2015-06', '2015-07', '2015-08'])].groupby('State').mean().reset_index().sort_values('State')

autumndf = mapData[mapData['Month'].isin(['2015-09', '2015-10', '2015-11'])].groupby('State').mean().reset_index().sort_values('State')





#abbDF = pd.read_html('https://www.50states.com/abbreviations.htm')[0]

# above line gives URLError. However, works on local notebook.

# extracted values by

# adict = abbDF.to_dict()

#abbState = list(adict[0].values())

#abbAB = list(adict[1].values())



abbState = ['US State:', 'Alabama', 'Alaska', 'Arizona', 'Arkansas', 'California', 'Colorado', 'Connecticut', 'Delaware', 'Florida', 'Georgia', 'Hawaii', 'Idaho', 'Illinois', 'Indiana', 'Iowa', 'Kansas', 'Kentucky', 'Louisiana', 'Maine', 'Maryland', 'Massachusetts', 'Michigan', 'Minnesota', 'Mississippi', 'Missouri', 'Montana', 'Nebraska', 'Nevada', 'New Hampshire', 'New Jersey', 'New Mexico', 'New York', 'North Carolina', 'North Dakota', 'Ohio', 'Oklahoma', 'Oregon', 'Pennsylvania', 'Rhode Island', 'South Carolina', 'South Dakota', 'Tennessee', 'Texas', 'Utah', 'Vermont', 'Virginia', 'Washington', 'West Virginia', 'Wisconsin', 'Wyoming', 'Commonwealth/Territory:', 'American Samoa', 'District Of Columbia', 'Federated States of Micronesia', 'Guam', 'Marshall Islands', 'Northern Mariana Islands', 'Palau', 'Puerto Rico', 'Virgin Islands', 'Military "State":', 'Armed Forces Africa', 'Armed Forces Americas', 'Armed Forces Canada', 'Armed Forces Europe', 'Armed Forces Middle East', 'Armed Forces Pacific']

abbAB = ['Abbreviation:', 'AL', 'AK', 'AZ', 'AR', 'CA', 'CO', 'CT', 'DE', 'FL', 'GA', 'HI', 'ID', 'IL', 'IN', 'IA', 'KS', 'KY', 'LA', 'ME', 'MD', 'MA', 'MI', 'MN', 'MS', 'MO', 'MT', 'NE', 'NV', 'NH', 'NJ', 'NM', 'NY', 'NC', 'ND', 'OH', 'OK', 'OR', 'PA', 'RI', 'SC', 'SD', 'TN', 'TX', 'UT', 'VT', 'VA', 'WA', 'WV', 'WI', 'WY', 'Abbreviation:', 'AS', 'DC', 'FM', 'GU', 'MH', 'MP', 'PW', 'PR', 'VI', 'Abbreviation:', 'AE', 'AA', 'AE', 'AE', 'AE', 'AP']

abbDF = pd.DataFrame([abbState,abbAB]).transpose()



#small correction, so things go smooth ahead

abbDF.iloc[53][0] = 'District Of Columbia'



mapA = []

mapS = []

for i in winterdf.index:

    mapA.append(str(winterdf['CO Mean'].iloc[i])[:5]+' ppm')

    mapS.append(abbDF[abbDF[0] == winterdf['State'].iloc[i]][1].values[0])

winterdf['text'] = mapA

winterdf['code'] = mapS



mapA = []

mapS = []

for i in springdf.index:

    mapA.append(str(springdf['CO Mean'].iloc[i])[:5]+' ppm')

    mapS.append(abbDF[abbDF[0] == springdf['State'].iloc[i]][1].values[0])

springdf['text'] = mapA

springdf['code'] = mapS



mapA = []

mapS = []

for i in summerdf.index:

    mapA.append(str(summerdf['CO Mean'].iloc[i])[:5]+' ppm')

    mapS.append(abbDF[abbDF[0] == summerdf['State'].iloc[i]][1].values[0])

summerdf['text'] = mapA

summerdf['code'] = mapS



mapA = []

mapS = []

for i in autumndf.index:

    mapA.append(str(autumndf['CO Mean'].iloc[i])[:5]+' ppm')

    mapS.append(abbDF[abbDF[0] == autumndf['State'].iloc[i]][1].values[0])

autumndf['text'] = mapA

autumndf['code'] = mapS
data = dict(type='choropleth',

            locations = winterdf['code'],

            z = winterdf['CO Mean'],

            locationmode = 'USA-states',

            text = winterdf['text'],

            marker = dict(line = dict(color = 'rgb(255,255,255)',width = 2)),

            colorbar = {'title':"CO Mean in ppm"}

            ) 

layout = dict(title = 'CO Mean Value in Winter by State',

              geo = dict(scope='usa',

                         showlakes = True,

                         lakecolor = 'rgb(85,173,240)')

             )

choromap = go.Figure(data = [data],layout = layout)

iplot(choromap)
data = dict(type='choropleth',

            locations = springdf['code'],

            z = springdf['CO Mean'],

            locationmode = 'USA-states',

            text = springdf['text'],

            marker = dict(line = dict(color = 'rgb(255,255,255)',width = 2)),

            colorbar = {'title':"CO Mean in ppm"}

            ) 

layout = dict(title = 'CO Mean Value in Spring by State',

              geo = dict(scope='usa',

                         showlakes = True,

                         lakecolor = 'rgb(85,173,240)')

             )

choromap = go.Figure(data = [data],layout = layout)

iplot(choromap)
data = dict(type='choropleth',

            locations = summerdf['code'],

            z = summerdf['CO Mean'],

            locationmode = 'USA-states',

            text = summerdf['text'],

            marker = dict(line = dict(color = 'rgb(255,255,255)',width = 2)),

            colorbar = {'title':"CO Mean in ppm"}

            ) 

layout = dict(title = 'CO Mean Value in Summer by State',

              geo = dict(scope='usa',

                         showlakes = True,

                         lakecolor = 'rgb(85,173,240)')

             )

choromap = go.Figure(data = [data],layout = layout)

iplot(choromap)
data = dict(type='choropleth',

            locations = autumndf['code'],

            z = autumndf['CO Mean'],

            locationmode = 'USA-states',

            text = autumndf['text'],

            marker = dict(line = dict(color = 'rgb(255,255,255)',width = 2)),

            colorbar = {'title':"CO Mean in ppm"}

            ) 

layout = dict(title = 'CO Mean Value in Autumn by State',

              geo = dict(scope='usa',

                         showlakes = True,

                         lakecolor = 'rgb(85,173,240)')

             )

choromap = go.Figure(data = [data],layout = layout)

iplot(choromap)
from sklearn.linear_model import LinearRegression

from sklearn.model_selection import train_test_split
chosenAddress = cal1[['Month','CO Mean']]

#aggregate them

chosenAddress = chosenAddress.groupby('Month').mean().reset_index().reset_index()

# we will use monthID on X-axis such that the first month in record will have monthID = 0

chosenAddress = chosenAddress.rename(columns = {'index':'monthID'})
start = cal1['Month'].min()

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

Y = chosenAddress[['CO Mean']] #label

# split train and test data

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=101)

# train our model

lm.fit(X_train,Y_train)
print("Intercept is "+str(lm.intercept_))
print("Coefficient is "+str(lm.coef_))
tempcal = pd.concat([X_train,Y_train],axis=1)

sns.lmplot(data=tempcal,x='monthID',y='CO Mean')
predictions = lm.predict(X_test)

sns.distplot((Y_test-predictions),bins=30)
inMonth = '2018-04'

print('In month '+inMonth+', predicted value of CO Mean is '+str(lm.predict([[toID(inMonth)]])[0][0]))
result_df = pd.DataFrame(index = list(range(len(Y_test))))

result_df['actual_result'] = Y_test.reset_index()['CO Mean']

result_df['predictions'] = predictions

result_df['difference'] = result_df['actual_result'] - result_df['predictions']#- predictions

result_df.to_csv("result_table.csv", index = False)

result_df.head()