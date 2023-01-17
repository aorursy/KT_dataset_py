#importing

import numpy as np

import pandas as pd 

import matplotlib.pyplot as plt 

import scipy.stats as stat

import os
#Loading in the NYC county level data from new york times

us_data=pd.read_csv('../input/uncover/UNCOVER/New_York_Times/covid-19-county-level-data.csv')



#seperation of inital growth data in smaller sub time steps

def exponential_growth(panda):

    days = len(panda.values)

    collection = np.zeros((36,2))



    k = 0

    for i in range(days - 1): 

        inital = panda.values[i]

        data = np.log(panda.iloc[i+1:].values/inital)

        x = np.arange(8-i)+1

        collection[k:k+8-i,0] = x

        collection[k:k+8-i,1] = data

        k += 8-i

    return collection





#determination of exponential growth constant and plots relative change in cases vs time

def New_york(county, data):

    #takes in county name and the data set to pull it from

    

    growth_bq_raw_data = data[(data['county'] == county) & (data['state'] =='New York') & (data['date'] > '2020-03-18') & (data['date'] < '2020-03-28')]['cases']

    growth_bq_exp = exponential_growth(growth_bq_raw_data)

    growth_aq_raw_data = data[(data['county'] == county) & (data['state'] =='New York') & (data['date'] > '2020-04-01') & (data['date'] < '2020-04-11')]['cases']

    growth_aq_exp = exponential_growth(growth_aq_raw_data)



    #scipy linregress to find the optimal growth constant

    bq = stat.linregress(growth_bq_exp[:,0],growth_bq_exp[:,1])

    aq = stat.linregress(growth_aq_exp[:,0],growth_aq_exp[:,1])

    

    #plotting

    x = np.arange(8)+1

    plt.plot(growth_bq_exp[:,0],growth_bq_exp[:,1],'o',label = 'before quarantine (Slope: %0.3f, std: %0.3f)'%(bq.slope,bq.stderr))

    plt.plot(x ,bq.intercept + bq.slope*x)

    plt.plot(growth_aq_exp[:,0],growth_aq_exp[:,1],'^',label = 'after quarantine (Slope %0.3f, std: %0.3f)'%(aq.slope,aq.stderr))

    plt.plot(x ,aq.intercept + aq.slope*x)

    plt.xlabel("T (days)")

    plt.ylabel('$ln(C_{t_{i}}/C_{t_{o}})$')

    plt.legend()

plt.figure(1,figsize=(12,12))

plt.subplot(221)

plt.title('New York City')

New_york('New York City',us_data)

plt.subplot(222)

plt.title('Westchester')

New_york('Westchester',us_data)

plt.subplot(223)

plt.title('Rockland')

New_york('Rockland',us_data)

plt.subplot(224)

plt.title('Nassau')

New_york('Nassau',us_data)

plt.show()
New_york('Ontario',us_data)

plt.title('Ontario')

plt.show()
def exponential_growth_sim(k,time,C ):

    collection = np.zeros((time))

    for i in np.arange(time):

        collection[i] = C* np.exp(k*i)

    return collection





#Data Prep from previous section

Real_data = us_data[(us_data['county'] == 'New York City') & (us_data['state'] =='New York') & (us_data['date'] > '2020-03-18')& (us_data['date'] < '2020-04-15')]['cases']

N = len(Real_data)

nyc_bq = 0.286

nyc_aq = 0.072

dates = us_data[(us_data['county'] == 'New York City') & (us_data['state'] =='New York') & (us_data['date'] > '2020-03-18')& (us_data['date'] < '2020-04-15')]['date'].values





#Simulation of Delayed quarantine

Sim_data=np.zeros((N))

Sim_data[:16] = exponential_growth_sim(nyc_bq, 16,Real_data.values[0])

Sim_data[15:] = exponential_growth_sim(nyc_aq, N-15,Sim_data[15])



#Simulation of Normal quarantine

Sim_data2=np.zeros((N))

Sim_data2[:10] = exponential_growth_sim(nyc_bq, 10,Real_data.values[0])

Sim_data2[9:] = exponential_growth_sim(nyc_aq, N-9,Sim_data2[9])





#plotting the figure

plt.figure(1,figsize=(8,8))

plt.plot(np.arange(N),Real_data.values,'o-',label='Actual NYC')

plt.plot(np.arange(N),Sim_data,'o-',label='Simulated NYC (Delayed Quarantine)')

plt.plot(np.arange(N),Sim_data2,'o-',label='Simulated NYC')

plt.vlines(2,0,400000,'r','--',label='Quarantine Start Date')

plt.vlines(9,0,400000,'b','--',label='Delayed Quarantine Start')

plt.title('Delayed Quarantine')

plt.ylabel('Cases')



plt.legend()

plt.xticks(np.arange(N),dates,rotation=70,fontsize=8)

plt.show()
