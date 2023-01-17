# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
countyData = pd.read_csv('/kaggle/input/us-counties-covid-19-dataset/us-counties.csv')

countyData = countyData.sort_values(by=['state'],ascending=True).reset_index(drop=True)

# countyData
stateData = countyData.groupby(['state','county','date'])['deaths','cases'].apply(lambda x: x.sum())

stateData = stateData.reset_index()

stateData = stateData.sort_values(by='date',ascending=False)

stateData = stateData.reset_index(drop=True)



stateDate = stateData.groupby(['state','date'])['deaths','cases'].apply(lambda x: x.sum())

stateDate = stateDate.reset_index()

# stateDate



stateActive = stateDate.query('deaths >= 50')



stateCases = stateDate.groupby('state')['cases'].sum()

stateCases = stateCases.reset_index()

stateCases = stateCases.sort_values(by=['cases'],ascending=False)

# stateCases



import plotly.express as px

fig = px.scatter(stateActive,x='date',y='deaths',color='state')



fig.show()
# countyCases = stateData.query('cases >= 200').query('deaths >= 30').groupby(['date','state','county','cases'])['deaths'].max()

countyCases = stateData.query('cases >= 200').groupby(['date','state','county','cases'])['deaths'].max()

countyCases = countyCases.reset_index()

countyCases = countyCases.sort_values(by='deaths',ascending=False)

countyCases = countyCases.reset_index(drop=True)



countyCasesMax = countyCases.groupby(['state','county'])['deaths'].max()

countyCasesMax = countyCasesMax.reset_index()

countyCasesMax = countyCasesMax.sort_values(by='deaths',ascending=False)

countyCasesMax = countyCasesMax.reset_index(drop=True)

countyCasesMax
fig = px.scatter(countyCasesMax,x=countyCasesMax['county'],y=countyCasesMax['deaths'],size=countyCasesMax['deaths'],color=countyCasesMax['deaths'])

fig.show()
coCases = countyCases.query('state == "Colorado"')

coCases = coCases.groupby(['county','date'])['deaths','cases'].apply(lambda x: x.sum())

coCases = coCases.reset_index()

coCases
fig = px.line(coCases,x='date',y='cases',color='county')

fig.show()
caCases = countyCases.query('state == "California"')

caCases = caCases.groupby(['county','date'])['deaths','cases'].apply(lambda x: x.sum())

caCases = caCases.reset_index()

caCases
fig = px.line(caCases,x='date',y='cases',color='county')

fig.show()
miCases = countyCases.query('state == "Michigan"')

miCases = miCases.groupby(['county','date'])['deaths','cases'].apply(lambda x: x.sum())

miCases = miCases.reset_index()

miCases
fig = px.line(miCases,x='date',y='cases',color='county')

fig.show()
azCases = countyCases.query('state == "Arizona"')

azCases = azCases.groupby(['county','date'])['deaths','cases'].apply(lambda x: x.sum())

azCases = azCases.reset_index()

azCases
fig = px.line(azCases,x='date',y='cases',color='county')

fig.show()