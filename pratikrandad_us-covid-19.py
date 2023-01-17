import matplotlib.pyplot as plt

import seaborn as sns

import pandas as pd

import plotly.express as px



data=pd.read_csv('/kaggle/input/us-counties-covid-19-dataset/us-counties.csv')



data.head()
data.info()
data.describe()
res=data.groupby(['state','date'])['cases','deaths'].sum()

res=res.sort_values('date')

res=res.reset_index()

res=res.query('deaths > = 50')

res.head()



fig=px.scatter(res,x='date',y='cases',color='state')

fig.update_layout(

    title_text='Cases in US states'

)

fig.show()

import plotly.express as px

fig=px.scatter(res,x='date',y='deaths',color='state')

fig.update_layout(

    title_text='Deaths in US states'

)

fig.show()

maxCasesState=data.groupby(['state','date'])['state','cases','deaths'].max()

maxCasesState=maxCasesState.reset_index()

maxCasesState=maxCasesState.groupby('state').max()

maxCasesState=maxCasesState.sort_values('deaths',ascending=False)

maxCasesState.head()

countyCases=data.groupby(['state','county','date'])['state','county','cases','deaths'].sum()

countyCases=countyCases.reset_index()

countyCases=countyCases.sort_values('cases',ascending=False)

countyCases.head()

countyCasesMax=countyCases.query('state=="New York"')

countyCasesMax=countyCasesMax.groupby(['state','county']).max()

countyCasesMax=countyCasesMax.reset_index()

countyCasesMax=countyCasesMax.sort_values('deaths',ascending=False)

countyCasesMax.head()

countyCasesMax=countyCases.query('state=="New York"')

countyCasesMax=countyCasesMax.groupby(['county','date']).max()

countyCasesMax=countyCasesMax.reset_index()

countyCasesMax=countyCasesMax.sort_values('deaths',ascending=False)

countyCasesMax.head()



fig = px.line(countyCasesMax,x='date',y='cases',color='county')

fig.update_layout(

    title_text='Cases in New York State by Counties'

)

fig.show()

fig=px.scatter(countyCasesMax,x=countyCasesMax['county'],y=countyCasesMax['deaths'],size=countyCasesMax['deaths'],color=countyCasesMax['deaths'])

fig.update_layout(

    title_text='Deaths in New York State by Counties'

)

fig.show()

countyCasesMax=countyCases.query('state=="Michigan"')

countyCasesMax=countyCasesMax.groupby(['county','date']).max()

countyCasesMax=countyCasesMax.reset_index()

countyCasesMax=countyCasesMax.sort_values('deaths',ascending=False)

fig = px.line(countyCasesMax,x='date',y='cases',color='county')

fig.update_layout(

    title_text='Cases in Michigan State by Counties'

)

fig.show()

countyCasesMax=countyCases.query('state=="Illinois"')

countyCasesMax=countyCasesMax.groupby(['county','date']).max()

countyCasesMax=countyCasesMax.reset_index()

countyCasesMax=countyCasesMax.sort_values('deaths',ascending=False)

fig = px.line(countyCasesMax,x='date',y='cases',color='county')

fig.update_layout(

    title_text='Deaths in Michigan State by Counties'

)

fig.show()



mortalityRate=data.groupby(['state','county','date'])['county','cases','deaths'].sum()

mortalityRate=mortalityRate.reset_index()

mortalityRate=mortalityRate.groupby('county').max()

mortalityRate['Death Rate']=(mortalityRate['deaths']/ mortalityRate['cases'])* 100

mortalityRate=mortalityRate.query('deaths > 10').query('state=="New York"')

mortalityRate=mortalityRate.sort_values('Death Rate',ascending=False)

mortalityRate=mortalityRate.reset_index()

mortalityRate

fig = px.scatter(mortalityRate,x='Death Rate',y='cases',color='county',size='Death Rate')

fig.update_layout(

    title_text='Mortality rate in New York State by Counties'

)

fig.show()
newYork=data[data['state']=='New York']

newYork=newYork.groupby(['county','date']).sum()

newYork=newYork.reset_index()

newYork=newYork.set_index('date')

newYork.index = pd.to_datetime(newYork.index)

newYork=newYork.resample('W').max()

newYork
michigan=data[data['state']=='Michigan']

michigan=michigan.groupby(['county','date']).sum()

michigan=michigan.reset_index()

michigan=michigan.set_index('date')

michigan.index = pd.to_datetime(michigan.index)

michigan=michigan.resample('W').max()

michigan

res=pd.merge(newYork,michigan,how='left' ,left_index=True, right_index=True)

res=res.fillna(0)

res=res.reset_index()

res
from pylab import rcParams



rcParams['figure.figsize'] = 14, 8

res.plot(kind='line',y=['cases_x','cases_y'],x='date',label=['Cases in NY','Cases in Michigan'],linewidth=5)



plt.show()
res.plot(kind='line',y=['deaths_x','deaths_y'],x='date',label=['Deaths in NY','Deaths in Michigan'],linewidth=5)

plt.show()