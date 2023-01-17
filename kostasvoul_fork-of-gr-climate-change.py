#Modules import

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import datetime

from bq_helper import BigQueryHelper

import warnings

warnings.filterwarnings('ignore')

years =range(1949,2018)
helper = BigQueryHelper('bigquery-public-data', 'noaa_gsod')



sql = '''

SELECT

  year, mo, da, temp, min, max, prcp, stn, b.lat,b.lon

FROM

    `bigquery-public-data.noaa_gsod.gsod{}` a



INNER JOIN

(SELECT usaf,country,lat,lon FROM `bigquery-public-data.noaa_gsod.stations`) b ON a.stn = b.usaf



WHERE 

    b.country = 'GR'

 '''
weather = [ helper.query_to_pandas(sql.format(i)) for i in years ]
weather = pd.concat(weather)



weather.head()
#Getting rid of missing values



#weather['temp'] = weather['temp'].replace({ 9999.9 : np.nan })

weather['min'] = weather['min'].replace({ 9999.9 : np.nan })

weather['max'] = weather['max'].replace({ 9999.9 : np.nan })

weather['prcp'] = weather['prcp'].replace( { 99.99 : 0 })



# turning temp into celsius degrees



def far_to_cel(temp_f):

    temp_c = (temp_f - 32) * 5/9

    return round(temp_c, 2)



for col in ['temp','min','max']:

    weather[col] = weather[col].apply(far_to_cel)





#and inches to mm

weather['prcp'] = round(weather['prcp'] * 25.4 , 2) 



#create our time index

weather['date'] = weather.apply(lambda x: 

                                datetime.datetime(int(x.year), int(x.mo), int(x.da)), 

                                axis=1)



weather.set_index('date', inplace = True)



weather.head()
%matplotlib inline

plt.style.use('fivethirtyeight')
grouped = weather.groupby('year').mean()





fig, ax = plt.subplots(figsize = (30,20))





ax.plot(grouped.index,

        grouped.temp ,

        linestyle = 'None', 

        marker = 'v', 

        color = 'blue', 

        label ='Average Temperature')



ax.plot(grouped.index,

        grouped[['min']] ,

        linestyle = '--',

        marker = 'o',

        color = 'red',

        label = 'Minimum Temperature')



ax.plot(grouped.index,

        grouped[['max']] ,

        linestyle = '--',

        marker = 'o',

        color = 'red',

        label = 'Maximum Temperature')





ax.tick_params('x', rotation= 90)

plt.title('Minimum, Average and Maximum Temperature per year')

plt.legend()



plt.show()
monthly_grouped = weather.groupby(['year','mo'])["temp"].mean()



monthly_grouped = pd.DataFrame.unstack(monthly_grouped)



monthly_grouped.columns = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']



fig, ax = plt.subplots(6,2,figsize = (35,25))





for i,f in enumerate(fig.axes):



    #Observations as markers

    

    f.plot(monthly_grouped.index.to_series().apply(lambda x: int(x)) ,

                     monthly_grouped.iloc[:,i],

                     linestyle = 'None',

                     marker = 'v')

    

    #moving average as line

    

    f.plot(monthly_grouped.index.to_series().apply(lambda x: int(x)) ,

                     monthly_grouped.iloc[:,i].rolling(5).mean(),

                     linestyle = '--',

                     color = 'orange')    

    

    

    f.set_xticks([*range(int(weather.year.min()),int(weather.year.max()),10)]) 

              #str([*range(int(weather.year.min()),int(weather.year.max()),5)]))

    

    #f.tick_params('x', rotation = 90)

    

    f.set_title(monthly_grouped.columns[i])

    

    

fig.suptitle('Temperature by Month As Points \n Moving Average of 5 years As Line', fontsize = 30)



plt.show()
#create a decade column



from math import floor



weather['decade'] = (((weather.year.apply(float))/10).apply(floor))*10



by_decade = weather.reset_index()[['temp','decade','mo']]



by_decade['mo'] = by_decade['mo'].astype('int32')
fig, axes = plt.subplots(6,2,figsize = (45,35))



for i,f in enumerate(fig.axes):

    

    sns.boxplot(x = 'decade', y = 'temp', ax = f , data = (by_decade[ by_decade['mo'] == i+1 ][['temp', 'decade']]), palette='deep') 

    

    f.set_title(monthly_grouped.columns[i])

    

    f.set_xlabel('')



    fig.suptitle('Distribution of temperatures by decade, faceted by month')

    

plt.show()
fig, axes = plt.subplots(6,2,figsize = (45,35))



for i,f in enumerate(fig.axes):

    

    sns.violinplot(x = 'decade', y = 'temp', ax = f , data = (by_decade[ by_decade['mo'] == i+1 ][['temp', 'decade']]), palette='pastel') 

    

    f.set_title(monthly_grouped.columns[i])

    

    f.set_xlabel('')



    fig.suptitle('Distribution of temperatures by decade, faceted by month')



    

plt.show()
by_decade.head()
#colors = ['blue','purple','green','orange','yellow','red']



#fig, ax = plt.subplots(1,1,figsize = (30,20))



#for i,f in enumerate(fig.axes):



g = sns.FacetGrid(by_decade, row = 'decade', height = 2, aspect = 7)



g.map(sns.distplot, "temp")



#plt.show()
(weather.index.unique())