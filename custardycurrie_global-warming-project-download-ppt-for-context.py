import seaborn as sns

import matplotlib.pyplot as plt

import numpy as np

import pandas as pd

%matplotlib inline

from plotly import __version__

from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

import cufflinks as cf

init_notebook_mode(connected=True)

cf.go_offline()



import plotly.plotly as py

import plotly.graph_objs as go

import plotly.tools as to

to.set_credentials_file(username='bbatv', api_key='qd2vtuCERZx7LkjuUMA5')

## Import packages ##
##Import Data, create Dataframes##
SeaTemp = pd.read_excel('Data Anlayst Task - Data.xlsx',sheet_name = 'Ave. Global Sea Surface Temp') 
SeaTemp.head()
SurfTemp = pd.read_excel('Data Anlayst Task - Data.xlsx',sheet_name = 'Gbl surf. Temp (land & sea)',skiprows=4) 
SurfTemp.head()
CO2 = pd.read_excel('Data Anlayst Task - Data.xlsx',sheet_name = 'Global CO2',skiprows=4) 
CO2

## This data is quite sparse so I will find another source##
Epica = pd.read_csv('Epica-tpt-co2.csv',skiprows=4)

## Ice Core data showing past 800k years of CO2 and Temperature.

## Came From http://www.climatedata.info/proxies/data-downloads/ 
## http://cait2.wri.org/historical/

CO2long = pd.read_csv('historical_emissions.csv')

CO2long
CO2short = CO2long.iloc[3]
CO2gas = CO2short[5:]

CO2gas.head()
# Sea Ice Data from http://nsidc.org/data/nsidc-0051.html 

SeaIce = pd.read_csv('seaice.csv')
CO21 = pd.read_csv('WorldResourcesInstituteCAIT.csv')

CO21.head()
#Visualising an unused data set that shows the decrease in area of Sea Ice since the 70s

SeaIce.groupby('Year').mean()['Extent'].plot()

SeaIce.groupby('Year').mean()['Extent'].head()
#Calculating Cumulative sum of emissions since 1850

CO2gas.sort_index(inplace=True)

Cumsum = np.cumsum(CO2gas)/22724

d = {'CO2 Emissions':CO2gas,'Cumulative sum of Emissions':Cumsum}

co2 = pd.DataFrame(data=d)
# Visualising the CumSum.

co2.index = list(map(int, co2.index))

fig = plt.Figure()

co2['Cumulative sum of Emissions'].plot(title='Cumulative Sum of CO2 Emissions',grid=True)

co2['Cumulative sum of Emissions'].set_xlabel = '% of Total Emissions'

# Visualising the change in annual emission rates using an interactive plot

co2['CO2 Emissions'].iplot(

                        kind='bar',

                        title='CO2 Emissions by year from 1850 to 2014',

                        xTitle='Year',

                        yTitle='CO2 Emissions (in Megatonnes)',

                        theme = 'pearl', filename = 'CO2 Emissions by year from 1850 to 2014',



    colorscale = 'polar', width = 3

)
sns.heatmap(SurfTemp.isnull(),yticklabels=False,cbar=False,cmap='viridis')

## Visualising the Null data in the Surface Temperature Dataset
## The UAH and RSS data begins in 1979 so I will create two separate data sets to work with from now on.
Surf79 = SurfTemp.dropna()

SurfT = SurfTemp.drop(axis = 1,labels = ['Lower troposphere (measured by satellite) (UAH)','Lower troposphere (measured by satellite) (RSS)'])
SurfT.set_index('Year',inplace=True)
# Following this are some simple visualisations to explore the data.
SurfT.plot(title='Earth Surface Temperature Plotted against Year')

#Here we can see an increase over the past 100+years with a particularly quick increase since 1960
Surf79.plot(y='Lower troposphere (measured by satellite) (UAH)', x='Year')
CO2.plot(x='Year',y='Energy')
# Below is a failed attempt at using machine learning to predict missing values for CO2 Levels
y_train = CO2['Energy']

x_train = CO2['Year'].values.reshape(-1,1)

x_pred = np.arange(1990,2010,1).reshape(-1,1)
from sklearn.linear_model import LinearRegression
lm = LinearRegression()
lm.fit(x_train,y_train)
y_pred = lm.predict(x_pred)
EnergyPred2  = list(y_pred.reshape(-1,1))
EnergyPred = pd.DataFrame(EnergyPred2)
EnergyPred.set_index(np.arange(1990,2010,1),inplace=True)
EnergyPred.set_axis(labels = 'Energy Predictions', axis=1)
CO2.set_index('Year')

EnergyPred.plot()
CO2.set_index('Year')

EnergyPred['Real'] = CO2['Energy']
## That didn't work, find better data and use that.##
pd.concat((CO2gas,SurfT),axis=1,verify_integrity=True)

##Attempt to perform outer Join ##
CO2gas.index

##Index is not an integer, which explains the error above ##
SurfT.index
CO2gas
CO2gas.index = list(map(int, CO2gas.index))
CO2gas.index
InnerCOSurf = pd.concat((CO2gas,SurfT),axis=1,join='inner')
SurfTemp.set_index('Year',inplace = True)

SeaTemp.set_index('Year',inplace = True)
## Now add other data sources to the master DataFrame.

InnerCOSurfSea = pd.concat((CO2gas,SurfT,SeaTemp),axis=1,join='inner')

InnerCOSurfSea.head()
InnerCOSurfSea.rename(columns={3:'CO2 Levels'},inplace=True)
droppedCO2 = InnerCOSurfSea.drop('CO2 Levels',axis=1)
# Exploration of correlation between CO2 and Temp measures

sns.pairplot(data=InnerCOSurfSea,y_vars='CO2 Levels',x_vars=droppedCO2.columns)
fig, ax1 = plt.subplots()

t = InnerCOSurfSea.index

s1 = InnerCOSurfSea['CO2 Levels']

ax1.plot(t, s1, 'b-')

ax1.set_xlabel('Year')

ax1.set_ylabel('CO2 Levels in Megatonnes', color='b')

ax1.tick_params('y', colors='b')



ax2 = ax1.twinx()

s2 = InnerCOSurfSea["Earth's surface (land and ocean - Fahreinheit)"]

ax2.plot(t, s2, 'r.')

ax2.set_ylabel('Earth Surface Temperature', color='r')

ax2.tick_params('y', colors='r')



fig.tight_layout()

plt.show()
fig, ax1 = plt.subplots()

t = InnerCOSurfSea[InnerCOSurfSea.index>1940].index

s1 = InnerCOSurfSea[InnerCOSurfSea.index>1940]['CO2 Levels']

ax1.plot(t, s1, 'b-')

ax1.set_xlabel('Year')

ax1.set_ylabel('CO2 Levels in Megatonnes', color='b')

ax1.tick_params('y', colors='b')



ax2 = ax1.twinx()

s2 = InnerCOSurfSea[InnerCOSurfSea.index>1940]["Earth's surface (land and ocean - Fahreinheit)"]

ax2.plot(t, s2, 'r.')

ax2.set_ylabel('Earth Surface Temperature', color='r')

ax2.tick_params('y', colors='r')



fig.tight_layout()

plt.show()
InnerCOSurfSea[InnerCOSurfSea.index>1940].index
InnerCOSurfSea['CO2 Levels + 10 years'] = InnerCOSurfSea['CO2 Levels']

for a in np.arange(1911,2015,1):

    InnerCOSurfSea['CO2 Levels + 10 years'][a] = InnerCOSurfSea['CO2 Levels'][a - 10]
#Using CumSum

from textwrap import wrap



fig, ax1 = plt.subplots()

t = InnerCOSurfSea[InnerCOSurfSea.index>1940].index

s1 = InnerCOSurfSea[InnerCOSurfSea.index>1940].cumsum()['CO2 Levels']

ax1.plot(t, s1, 'b-')

ax1.set_xlabel('Year')

ax1.set_ylabel('CO2 Levels in Megatonnes', color='b')

ax1.tick_params('y', colors='b')

ax1.set_title("\n".join(wrap('Correlation Between Cumulative CO2 Emissions and Earth Surface Temperature'

              ' in the 20th and 21st Century')))



ax2 = ax1.twinx()

s2 = InnerCOSurfSea[InnerCOSurfSea.index>1940]["Earth's surface (land and ocean - Fahreinheit)"]

ax2.plot(t, s2, 'r.')

ax2.set_ylabel('Earth Surface Temperature', color='r')

ax2.tick_params('y', colors='r')



plt.savefig("Cumsum CO2 Vs Temp in 2021st.png",dpi = 200)

plt.show()


InnerCOSurfSea['CO2 Levels + 10 years'] = InnerCOSurfSea['CO2 Levels']

for a in np.arange(1911,2015,1):

    InnerCOSurfSea['CO2 Levels + 10 years'][a] = InnerCOSurfSea['CO2 Levels'][a - 10]
InnerCOSurfSea['CO2 Levels + 5 years'] = InnerCOSurfSea['CO2 Levels']

for a in np.arange(1906,2015,1):

    InnerCOSurfSea['CO2 Levels + 5 years'][a] = InnerCOSurfSea['CO2 Levels'][a - 5]
InnerCOSurfSea['CO2 Levels + 15 years'] = InnerCOSurfSea['CO2 Levels']

for a in np.arange(1916,2015,1):

    InnerCOSurfSea['CO2 Levels + 15 years'][a] = InnerCOSurfSea['CO2 Levels'][a - 15]
fig, ax1 = plt.subplots(figsize = (9,6))

t = InnerCOSurfSea[InnerCOSurfSea.index>1940].index

s1 = InnerCOSurfSea[InnerCOSurfSea.index>1940]['CO2 Levels + 5 years']

s11 = InnerCOSurfSea[InnerCOSurfSea.index>1940]['CO2 Levels + 10 years']

s12 = InnerCOSurfSea[InnerCOSurfSea.index>1940]['CO2 Levels + 15 years']

ax1.plot(t, s1, 'b-')

ax1.set_xlabel('Year')

ax1.set_ylabel('CO2 Levels in Megatonnes + 10 years Wait', color='b')

ax1.tick_params('y', colors='b')



ax1.plot(t, s11, 'r-')

ax1.set_xlabel('Year')

ax1.set_ylabel('CO2 Levels + 20 year Delay', color='b')

ax1.tick_params('y', colors='black')



ax1.plot(t, s12, 'g-')

ax1.set_xlabel('Year')

ax1.set_ylabel('CO2 Levels in Megatonnes', color='b')

ax1.tick_params('y', colors='b')



ax2 = ax1.twinx()

s2 = InnerCOSurfSea[InnerCOSurfSea.index>1940]["Earth's surface (land and ocean - Fahreinheit)"]

ax2.plot(t, s2, 'r.')

ax2.set_ylabel('Earth Surface Temperature', color='r')

ax2.tick_params('y', colors='r')



fig.legend(loc=(0.1,0.8),labels=('5 Year Delay','10 Year Delay','30 Year Delay','Surface Temperature'))

fig.tight_layout()

fig.savefig('CO2 Levels + Delay.png')

plt.show()



#Comparison of 0,+5,10,15 year delays to Surface Temperature. There seems to be a stronger correlation with the 

# 30 year delay than the others. 
# Calculating ROC for Temperature

Epica['Surface Temp Change'] = Epica['Temperature'].diff()

Epica['Surface Temp Change'].describe()
RateOfCurrentChange = (1.25*100/44)

Morethancurrent = Epica[Epica['Surface Temp Change']>RateOfCurrentChange].count()['BP']

ninety5th = Epica[Epica['Surface Temp Change']>(RateOfCurrentChange*0.95)].count()['BP']

print('Count of ROCs greater than current:',Morethancurrent)

print('Count of ROCs within 5%:',ninety5th)
Epica['Surface Temp Change'].iplot(kind='hist',title='Histogram of Surface Temperature Rate of Change over the past 800k Years',theme='polar'

                                   ,xTitle='ROC of Temperature (in Fahreinheit)',yTitle='Count',filename='Hist for 800k ROC.png',)
InnerCOSurfSea[InnerCOSurfSea.index>1941].corr()


sns.pairplot(InnerCOSurfSea)
# Issues Creating error bars#

plt.figure()

plt.errorbar(x = SeaTemp.index, y = SeaTemp['Annual anomaly (fahrenheit)'], yerr=[SeaTemp['Upper 95% confidence interval'],SeaTemp['Lower 95% confidence interval']],uplims=True,lolims=True,ecolor='orange')

plt.title("Simplest errorbars, 0.2 in x, 0.4 in y")
SeaTemp['Error'] = SeaTemp['Upper 95% confidence interval'] - SeaTemp['Lower 95% confidence interval']


usethis = [

    go.Scatter(

        title = 'Average Sea Surface Temperature Since 1880',

        x=SeaTemp[SeaTemp.index>1850].index,

        y=SeaTemp[SeaTemp.index>1850]['Annual anomaly (fahrenheit)'],

       error_y=dict(

            type='data',

            symmetric=True,

            array=SeaTemp['Error'],

           thickness = 0.8,

           width = 2,

           color = 'orange', opacity = 0.8

            

        ), 

        line = dict(color = 'blue', width = 1, shape = 'spline',smoothing = 1),

        xaxis = dict(text = 'Year')

    )

]

py.iplot(usethis, filename='SeaTemps With Error Bars')
#Calculating Average for Lower Troposphere Temp Reading#

Surf79['Average Lower Troposphere'] = (Surf79['Lower troposphere (measured by satellite) (RSS)']

                                        +Surf79['Lower troposphere (measured by satellite) (UAH)'])/2
co2.index
After79 = Surf79.set_index('Year')

After79['Sea Temperature'] = SeaTemp[SeaTemp.index>1978]['Annual anomaly (fahrenheit)']

After79['CO2 Levels'] = co2[co2.index>1978]['CO2 Emissions']

After79['Cumulative CO2 Emissions'] = co2[co2.index>1978]['Cumulative sum of Emissions']

After79.head()
bins = np.arange(1850,2030,10)
co2.astype('float64').head()
sns.set_style(style='whitegrid')

sns.color_palette(palette='pastel')

sns.barplot(data = co2, x=co2.index,y=binned)
fig, ax1 = plt.subplots(figsize=(12,6))

t = Epica['BP']

s1 = Epica['CO2']

ax1.plot(t, s1, 'b-')

ax1.set_xlabel('Years Before Present')

ax1.set_ylabel('CO2 Levels in Megatonnes', color='black')

ax1.tick_params('y', colors='black')



ax2 = ax1.twinx()

s2 = Epica['Temperature']

ax2.plot(t, s2, 'r-', alpha = 0.3)

ax2.set_ylabel('Earth Surface Temperature', color='black')

ax2.tick_params('y', colors='black')





plt.show()


trace1 = go.Scatter(

    x=Epica['BP'],

    y=Epica['CO2'],

    name='CO2 Levels',

    line = dict(width = 1,shape = 'spline', smoothing = 1)

)

trace2 = go.Scatter(

    x=Epica['BP'],

    y=Epica['Temperature'],

    name='Temperature',

    yaxis='y2',

    line = dict(width = 0.5, shape = 'spline', smoothing = 1)

)

data = [trace1, trace2]

layout = go.Layout(

    title='Relationship Between CO2 Levels and Global Temperature',

    yaxis=dict(

        title='CO2 Levels'

    ),

    yaxis2=dict(

        title='Temperature',

        overlaying='y',

        side='right'

    )

)

fig = go.Figure(data=data, layout=layout) 

py.iplot(fig, filename='CO2 Levels Vs Global Temperature')


X = InnerCOSurfSea.drop(columns = ["Earth's surface (land and ocean - Fahreinheit)",'Annual anomaly (fahrenheit)', 

                    'Lower 95% confidence interval','Upper 95% confidence interval'])

y = InnerCOSurfSea["Earth's surface (land and ocean - Fahreinheit)"]
from sklearn.model_selection import train_test_split



X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.25, random_state=101)

from sklearn.linear_model import LinearRegression



lm = LinearRegression()



lm.fit(X_train,y_train)
# The coefficients

print('Coefficients: \n', lm.coef_)
predictions = lm.predict( X_test)
plt.scatter(y_test,predictions)

plt.xlabel('Y Test')

plt.ylabel('Predicted Y')
from sklearn import metrics



print('MAE:', metrics.mean_absolute_error(y_test, predictions))

print('MSE:', metrics.mean_squared_error(y_test, predictions))

print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))
sns.distplot((y_test-predictions),bins=50);
coeffecients = pd.DataFrame(lm.coef_,X.columns)

coeffecients.columns = ['Coeffecient']

coeffecients
# Not very useful but was fun to try!