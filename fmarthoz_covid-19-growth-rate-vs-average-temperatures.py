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
from datetime import datetime



df_temp=pd.read_csv('/kaggle/input/corona-virus-report/covid_19_clean_complete.csv')

temps=pd.read_csv('/kaggle/input/avg-temps-feb20/temps.csv')

arrivals = pd.read_csv("/kaggle/input/popstats/arrivals.csv")

pop_density = pd.read_csv("/kaggle/input/popstats/pop_density.csv")

ctr_fr=["China","Korea, Rep.","United States","Hong Kong SAR, China","United Kingdom","Russian Federation","Iran, Islamic Rep."]

ctr_to=["Mainland China","South Korea","US","Hong Kong","UK","Russia","Iran"]

pop_density=pop_density.replace(ctr_fr,ctr_to)

arrivals=arrivals.replace(ctr_fr,ctr_to)



df=pd.merge(df_temp, temps, how='inner', left_on='Country/Region', right_on='Country')

df=pd.merge(df, pop_density, how='inner', left_on='Country/Region', right_on='Country')

df=pd.merge(df, arrivals, how='inner', left_on='Country/Region', right_on='Country')

# Lets list all countries in the document

countries=df["Country/Region"].unique()

df['Date'] = pd.to_datetime(df['Date'], format='%m/%d/%y')

df.arrivals_2018=df.arrivals_2018/df.arrivals_2018.max()

import matplotlib.pyplot as plt

from pylab import *

from scipy.optimize import curve_fit

def func(x, a, b):

    return a*np.exp(b*x)

countries=['Mainland China' ,'Thailand', 'Japan', 'South Korea' ,'US',

 'Hong Kong' ,'Singapore' ,'Vietnam','France' , 'Malaysia' ,'Canada',

 'Australia' ,'Germany', 'Finland' ,'Philippines', 'Italy', 'UK', 'Russia',

 'Sweden', 'Spain', 'Belgium', 'Iran' ,'Lebanon', 'Algeria' ,'Croatia','Switzerland',

 'Austria' ,'Israel', 'Brazil','Greece', 'Norway', 'Romania', 'Denmark', 'Estonia', 'Netherlands', 'New Zealand',

  'Ireland' , 'Indonesia']

df['growth_rate']=0

df['nber_days']=0

lt=0

for ctr in countries:

    lt=lt+1

    country=(df[(df["Country/Region"]==ctr) & (df["Confirmed"]>1)]).groupby("Date").Confirmed.sum()

    n=country.shape[0]

    a=np.arange(n)

    popt, pcov = curve_fit(func, a, country, [1,-1])

    #plt.plot(a,func(a,*popt),color="red",alpha=0.9)

    #print("Growth rate for "+ctr+"="+str(np.exp(popt[1])))

    df.loc[df["Country/Region"] == ctr, 'growth_rate'] = np.exp(popt[1])

    df.loc[df["Country/Region"] == ctr, 'nber_days'] = n

    plt.subplot(10,4,lt)



    plt.plot(a,country)

    plt.title(ctr)

    plt.xticks(rotation='vertical')

    plt.plot(a,func(a,*popt),color="red",alpha=0.9)

plt.subplots_adjust(bottom=0.1)



plt.show()    
ctr="Mainland China"

country=(df[(df["Country/Region"]==ctr) & (df["Confirmed"]>1)]).groupby("Date").Confirmed.sum()

n=country.shape[0]

a=np.arange(n)

popt, pcov = curve_fit(func, a, country, [1,1])

print("Growth rate for "+ctr+"="+str(np.exp(popt[1])))

df.loc[df["Country/Region"] == ctr, 'growth_rate'] = np.exp(popt[1])

df.loc[df["Country/Region"] == ctr, 'nber_days'] = n

plt.plot(a,country)

plt.title(ctr)

plt.xticks(rotation='vertical')

plt.plot(a,func(a,*popt),color="red",alpha=0.9)



plt.show()    
df2=df[(df.growth_rate>0) & (df.nber_days>10)].groupby('Country/Region', as_index=False).mean()



import statsmodels.api as sm

results = sm.OLS(df2.growth_rate,sm.add_constant(df2.temp_feb)).fit()

print(results.summary())

fig, axes= plt.subplots(nrows=1, ncols=1,figsize=(16,13))

plt.scatter(df2.temp_feb,df2.growth_rate)

plt.plot(df2.temp_feb,results.params[0]+results.params[1]*df2.temp_feb)

plt.title("Temperatures vs Growth Rate")

for i in range(0,df2.shape[0]):

    plt.annotate(df2.loc[i,"Country/Region"],[df2.loc[i,"temp_feb"],df2.loc[i,"growth_rate"]])

plt.show()
fig, axes= plt.subplots(nrows=1, ncols=1,figsize=(15,15))

plt.scatter(df2.arrivals_2018,df2.growth_rate)

results = sm.OLS(df2.growth_rate,sm.add_constant(df2.arrivals_2018)).fit()

plt.plot(df2.arrivals_2018,results.params[0]+results.params[1]*df2.arrivals_2018)

plt.title("Tourism influx (in 2018) vs Growth Rate")

for i in range(0,df2.shape[0]):

    plt.annotate(df2.loc[i,"Country/Region"],[df2.loc[i,"arrivals_2018"],df2.loc[i,"growth_rate"]])

plt.show()





fig, axes= plt.subplots(nrows=1, ncols=1,figsize=(15,15))

plt.scatter(df2.pop_density_2018,df2.growth_rate)

results = sm.OLS(df2.growth_rate,sm.add_constant(df2.pop_density_2018)).fit()

plt.plot(df2.pop_density_2018,results.params[0]+results.params[1]*df2.pop_density_2018)

plt.title("Population density (in 2018) vs Growth Rate")

for i in range(0,df2.shape[0]):

    plt.annotate(df2.loc[i,"Country/Region"],[df2.loc[i,"pop_density_2018"],df2.loc[i,"growth_rate"]])

plt.show()





df2=df[(df.growth_rate>0) & (df.nber_days>10)].groupby('Country/Region', as_index=False).mean()

import statsmodels.api as sm

results = sm.OLS(df2.growth_rate,sm.add_constant(df2[['temp_feb','arrivals_2018']])).fit()

fig, axes= plt.subplots(nrows=1, ncols=1,figsize=(15,15))

plt.scatter(y=df2.growth_rate,x=results.predict())

plt.plot([0.9,1.6],[0.9,1.6],color="red")

plt.title("Predicted vs. observed")

plt.xlabel("Model Predictions")

plt.ylabel("Actuals")

for i in range(0,df2.shape[0]):

    plt.annotate(df2.loc[i,"Country/Region"],[results.predict()[i],df2.loc[i,"growth_rate"]])

plt.show()

print(results.summary())