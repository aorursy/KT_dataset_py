import numpy as np 

import pandas as pd 



import os

# Disable warnings 

import warnings

warnings.filterwarnings('ignore')

# Reading the coronvirus dataset

data= pd.read_csv("../input/novel-corona-virus-2019-dataset/covid_19_data.csv")



data.head()
temp_all=pd.read_csv('../input/climate-change-earth-surface-temperature-data/GlobalLandTemperaturesByCountry.csv')

temp_all['Year']=temp_all['dt'].apply(lambda x:int(x.split('-')[0]))

temp_all['Month']=temp_all['dt'].apply(lambda x:int(x.split('-')[1]))

print('Available Year Range  Min -',temp_all['Year'].min(),' Max -',temp_all['Year'].max())

temp_all=temp_all[temp_all['Year']>2010]

avg_temp=temp_all.groupby(['Country','Month']).agg('mean')

avg_temp=avg_temp.reset_index()

feb_temp=avg_temp[avg_temp['Month']==2] #slicing

feb_temp_avg=feb_temp[['Country','AverageTemperature']]

feb_temp_avg.columns=['Country/Region','FebAverageTemperature']

feb_temp_avg.reset_index(drop=True)

mar_temp=avg_temp[avg_temp['Month']==3] #slicing

mar_temp_avg=feb_temp[['Country','AverageTemperature']]

mar_temp_avg.columns=['Country/Region','MarAverageTemperature']

mar_temp_avg.reset_index(drop=True)



 #replacing proper country names

data['Country/Region']=data['Country/Region'].apply(lambda x:x.replace('UK','United Kingdom'))

data['Country/Region']=data['Country/Region'].apply(lambda x:x.replace('US','United States'))

data['Country/Region']=data['Country/Region'].apply(lambda x:x.replace('Mainland China','China'))

data=data[data['Country/Region']!='Asia']

data['NumDaysSinceFirstCase']=data.groupby('Country/Region').agg('cumcount')



#daywise aggregation at country level since we have data at province/state level

data_daywise=data.groupby(['Country/Region','ObservationDate']).agg('sum').reset_index()

data_cumcount=data_daywise.groupby('Country/Region').agg('cumcount')

data_daywise['NumDaysSinceFirstCase']=data_cumcount

del data_daywise['SNo']



#Finding number of days since the first case at each country is reported in the dataset.

data_daywise.groupby(['Country/Region'])['NumDaysSinceFirstCase'].max().reset_index().sort_values('NumDaysSinceFirstCase',ascending=False).head(15)

#merge feb temperature data

data_daywise=data_daywise.merge(feb_temp_avg,how='left',on='Country/Region')
##55 days since the first infection in each country

n=55

day_n=data_daywise[data_daywise['NumDaysSinceFirstCase']==n]

day_n.sort_values('Confirmed',ascending=False).reset_index(drop=True)

import numpy as np



from bokeh.layouts import column, row

from bokeh.models import CustomJS, Slider, HoverTool

from bokeh.plotting import ColumnDataSource, figure, output_file, show,output_notebook

day_n.columns=['Country', 'ObservationDate', 'Confirmed', 'Deaths', 'Recovered', 'NumDaysSinceFirstCase', 'FebAverageTemperature']

output_notebook()

source = ColumnDataSource(day_n)

hover = HoverTool(tooltips=[

    ("Country", "@Country"),

    ("Confirmed", "@Confirmed"),

    ('Feb Avg. Temp', '@FebAverageTemperature'),

])

p = figure(tools=[hover])

p.vbar(x='FebAverageTemperature', top='Confirmed', source=source,width=1)

show(p)
del day_n['NumDaysSinceFirstCase']

day_n.corr()
from scipy.spatial.distance import pdist, squareform

import numpy as np



#thanks to @satra for distance correlation script since there is no standard script in scipy or any other standard python packages 

#https://gist.github.com/satra/aa3d19a12b74e9ab7941



def distcorr(X, Y):

    """ Compute the distance correlation function

    

    >>> a = [1,2,3,4,5]

    >>> b = np.array([1,2,9,4,4])

    >>> distcorr(a, b)

    0.762676242417

    """

    X = np.atleast_1d(X)

    Y = np.atleast_1d(Y)

    if np.prod(X.shape) == len(X):

        X = X[:, None]

    if np.prod(Y.shape) == len(Y):

        Y = Y[:, None]

    X = np.atleast_2d(X)

    Y = np.atleast_2d(Y)

    n = X.shape[0]

    if Y.shape[0] != X.shape[0]:

        raise ValueError('Number of samples must match')

    a = squareform(pdist(X))

    b = squareform(pdist(Y))

    A = a - a.mean(axis=0)[None, :] - a.mean(axis=1)[:, None] + a.mean()

    B = b - b.mean(axis=0)[None, :] - b.mean(axis=1)[:, None] + b.mean()

    

    dcov2_xy = (A * B).sum()/float(n * n)

    dcov2_xx = (A * A).sum()/float(n * n)

    dcov2_yy = (B * B).sum()/float(n * n)

    dcor = np.sqrt(dcov2_xy)/np.sqrt(np.sqrt(dcov2_xx) * np.sqrt(dcov2_yy))

    return dcor

day_n=day_n[~day_n['FebAverageTemperature'].isna()]

print('distance correlation',distcorr(day_n['FebAverageTemperature'],day_n['Confirmed']))
##Correlation between temp and confirmed cases over time

dist_correl=[]

num_countries=[]

for i in range(1,max(data_daywise['NumDaysSinceFirstCase'])):

    day_n=data_daywise[data_daywise['NumDaysSinceFirstCase']==i]

    day_n=day_n[~day_n['FebAverageTemperature'].isna()]

    dist_correl.append(distcorr(day_n['FebAverageTemperature'],day_n['Confirmed']))

    num_countries.append(len(day_n))

corr_df=pd.DataFrame(dist_correl,columns=['dist_correl'])

corr_df['Days']= range(1,max(data_daywise['NumDaysSinceFirstCase']))

corr_df['NumCountries']=num_countries

p = figure()

source = ColumnDataSource(corr_df)

p.vbar(x='Days', top='dist_correl', source=source,width=1)

show(p)

mar_temp_avg=mar_temp_avg[mar_temp_avg['Country/Region']!='Asia']

#Tbe list of high vulnerable countries are which lie in temp. range of -5 to 10 degree C during march

list(mar_temp_avg[(mar_temp_avg['MarAverageTemperature']>-5) & (mar_temp_avg['MarAverageTemperature']<10)]['Country/Region'])