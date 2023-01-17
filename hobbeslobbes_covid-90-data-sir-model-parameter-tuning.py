# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from datetime import datetime

from scipy.integrate import odeint

import warnings

warnings.filterwarnings('ignore')





# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

        df = pd.read_csv(os.path.join(dirname, filename))



        print('Sampling of Raw Data')        

        print(df.tail())

# Any results you write to the current directory are saved as output.
# Consolidating Countries



def state_to_federal(df, Country='US') :



    for id, gdf in df.groupby('Country/Region'):

        gdf = gdf.reset_index()

        gdf['Total'] = gdf.apply(lambda x: x['Confirmed']+x['Deaths']+x['Recovered'], axis=1)



        total=gdf['Total'].to_list()

        N_pop = total[-1]        

        if (id == Country) :

            fed_gdf=gdf.drop(['index'], axis=1) 



            # Construct Aggregate        

            fed_gdf = fed_gdf.drop(['Province/State','Country/Region', 'Lat', 'Long'], axis=1)

            fed_gdf = fed_gdf.groupby('Date').aggregate({'Confirmed':'sum',

                                                         'Deaths':'sum',

                                                         'Recovered':'sum',

                                                         'Total':'sum' }).reset_index()

            fed_gdf['DateTime'] = fed_gdf['Date'].apply(lambda x: datetime.strptime(x, '%m/%d/%y'))



            fed_gdf.sort_values('DateTime', inplace=True)

            

            break



    return fed_gdf            



# China: 

china_fed_df = state_to_federal(df, 'China')

us_fed_df = state_to_federal(df, 'US')

df.tail()
print('*** Summary: Countries with Total Fatalities > 50 ***')

print('  * This Display:*  ')

      

country_df = []

for id, gdf in df.groupby('Country/Region'):

#     print(id)

    gdf = state_to_federal(gdf, id)

    gdf = gdf.drop(['DateTime', 'Total'], axis=1)

    gdf_desc=gdf.describe()



    max_deaths = gdf['Deaths'].max()

    if ( max_deaths > 50 ) :

#         print(gdf)

        #max_deaths = gdf_desc.loc['max','Deaths']

#         print(f'Total Fatalities: {id}, {max_deaths}')

#         print('Total fatalities:')

        total_df = gdf[gdf['Deaths']==max_deaths].reset_index().drop('index', axis=1)

        if total_df.shape[0] > 1 :

            total_df['Datetime'] = total_df['Date'].apply(lambda x: datetime.strptime(x, '%m/%d/%y'))

            total_df.head()

            total_df = total_df[total_df['Datetime']==total_df['Datetime'].max()]

            total_df = total_df.reset_index().drop(['index', 'Datetime'], axis=1)

            

#             total_df = gdf[gdf['Confirmed']==gdf_desc.loc['max','Confirmed']]

        total_df['Country'] = id

#         print( total_df )

        country_df.append(total_df)



country_df = pd.concat(country_df)      

country_df = country_df.reset_index().drop(['index'], axis=1)

country_df.set_index('Country', inplace=True)

country_df.sort_values('Confirmed', inplace=True)

print(country_df.head())
import matplotlib.pyplot as plt

plt.rcParams['font.serif'] = ['Times']

AXES_SIZE = 14

SIZE=16



axes = country_df.plot.bar(rot=0)

plt.title('Countries > 50 fatalities', fontsize=20)

plt.xticks(fontsize=14, rotation=75)

plt.yscale('log')

# axes.set_xlabel(fontsize=18)



fig = plt.gcf()

fig.set_size_inches(10, 6)



plt.rc('font', size=SIZE)

plt.rc('axes', titlesize=AXES_SIZE)



# China - Sampling

fig = plt.gcf()

fig.set_size_inches(20, 10)



plt.rc('font', size=SIZE)

plt.rc('axes', titlesize=AXES_SIZE)

plt.scatter(china_fed_df['Date'], china_fed_df['Confirmed'], alpha=0.5, label='China: Whole')



for id, gdf in df.groupby('Country/Region'):

    if (id == 'China') :

        #gdf = gdf.sort_values(by=['Confirmed'])

        china_gdf=gdf

        plt.scatter(gdf['Date'], gdf['Confirmed'], alpha=0.5, label=gdf['Province/State'])

        #break

plt.xticks(fontsize=14, rotation=45)

# plt.yscale('log')    

plt.legend()

plt.grid('on');
fig = plt.gcf()

fig.set_size_inches(20, 10)



plt.rc('font', size=SIZE)

plt.rc('axes', titlesize=AXES_SIZE)



for id, gdf in df.groupby('Country/Region'):

    gdf = gdf.reset_index()

    gdf['Total'] = gdf.apply(lambda x: x['Confirmed']+x['Deaths']+x['Recovered'], axis=1)



    total=gdf['Confirmed'].to_list()

    N_pop = total[-1]

    



    if (id == 'India') :

        plt.scatter(gdf['Date'], gdf['Confirmed'], alpha=0.5, label=id)

        india_df = gdf.drop(['index'], axis=1)

        Nind_pop = N_pop*500

    if (id == 'Italy') :

        plt.scatter(gdf['Date'], gdf['Confirmed'], alpha=0.5, label=id)

        italy_df = gdf.drop(['index'], axis=1)

        Nit_pop = N_pop*5 #*30

    if (id == 'United Kingdom') :

        uk_df = gdf[gdf['Province/State']=='United Kingdom']

        uk_df=uk_df.drop(['index'], axis=1)   

        uk_df = uk_df.reset_index().drop(['index'], axis=1)        

        plt.scatter(gdf['Date'], gdf['Confirmed'], alpha=0.95, label=id)

        Nuk_pop = N_pop*30

        



plt.scatter(us_fed_df['Date'], us_fed_df['Confirmed'], alpha=0.95, label='US: Federal')

plt.xticks(fontsize=14, rotation=45)

# plt.yscale('log')    

plt.legend()

plt.grid('on');

plt.yscale('log')





# Baseline pops to Italy

# print(Nit_pop)

Nit_pop = np.round(Nit_pop,-4) #1.25

Nind_pop = Nit_pop

Nuk_pop = Nit_pop

Nus_pop = Nit_pop

# Shaping Data for SIR model



def modf_df(df, N_pop):

    df['dconf'] = df['Confirmed'].fillna(0).apply(lambda x: x/N_pop)

    df['ddeaths'] = df['Deaths'].fillna(0).apply(lambda x: x/N_pop)

    df['drec'] = df['Recovered'].fillna(0).apply(lambda x: x/N_pop)

    df['dinf'] = df.apply(lambda x: x['dconf']-x['drec'], axis=1)

    df['Inf'] = df.apply(lambda x: x['Confirmed']-x['Recovered'], axis=1)

    

    # 

    df['days_fromstart'] = df['Date'].apply(lambda x: datetime.strptime(x, '%m/%d/%y'))

    df['days_fromstart'] = df['days_fromstart']  - df['days_fromstart'][0]

    df['days_fromstart'] = df['days_fromstart'].apply(lambda x: float(x.days) )

    

    return df

    
# Managing China and strategic provinces (Hubei and Hongkong)

china_gdf['Total'] = china_gdf.apply(lambda x: x['Confirmed']+x['Deaths']+x['Recovered'], axis=1)

for id, c_gdf in china_gdf.groupby('Province/State'):

    total=c_gdf['Confirmed'].to_list()

    N_pop = max(10000, np.round(total[-1], -2))

    

    if (id == 'Hubei') or (id == 'Hong Kong') :        

        c_gdf = c_gdf.reset_index()

        c_df = modf_df(c_gdf, N_pop)

    

    if id == 'Hubei' :

        china_hub_gdf = c_gdf

        Nchina_hub_pop = N_pop

    if id == 'Hong Kong' :

        china_hk_gdf = c_gdf

        Nchina_hk_pop = N_pop

        

# china_fed_df = modf_df(china_gdf, Nus_pop)
# Shaping strategic country data



india_df = modf_df(india_df, Nind_pop)

italy_df = modf_df(italy_df, Nit_pop)

uk_df = modf_df(uk_df, Nuk_pop)

us_fed_df = modf_df(us_fed_df, Nus_pop)



# Population size list - for reporting

Npop_list = [Nchina_hub_pop, Nit_pop, Nchina_hk_pop, Nuk_pop, Nus_pop, Nind_pop ]

Ncountry_list = ['China_Hubei', 'Italy', 'China_HK', 'UK', 'USA', 'India'] 



Npop_dict = dict(zip(Ncountry_list, Npop_list))

# Npop_dict

fig, axes = plt.subplots(nrows=1,ncols=1,figsize=(15,10),facecolor='w')

axes.plot(china_hub_gdf['days_fromstart'], china_hub_gdf['Inf'], 'ro', label='China: Hubei')

axes.plot(china_hk_gdf['days_fromstart'], china_hk_gdf['Inf'], 'yo', alpha=0.95, label='China: HK')



axes.plot(italy_df['days_fromstart'], italy_df['Inf'], 'bo', label='Italy')

axes.plot(uk_df['days_fromstart'], uk_df['Inf'], 'co', label='UK')

axes.plot(us_fed_df['days_fromstart'], us_fed_df['Inf'], 'mo', label='US')

#axes.plot( china_hub_gdf['days_fromstart'], china_hub_gdf['drec'], 'bv',alpha=0.25,  label='China: Hubei, Recovered')

axes.plot(india_df['days_fromstart'], india_df['Inf'], 'ko', label='India')

#axes.plot( india_df['days_fromstart'], india_df['drec'], 'yo', label='India, Recovered')



axes.set_ylabel('Numbers of New Infections Reported')

axes.set_xlabel('Days since 22 Jan 2020')

axes.grid('on')

legend = axes.legend(bbox_to_anchor=(1.1, 1.05))

axes.set_yscale('log')

import matplotlib.pyplot as plt



# Incubation Period

Tinc=15



# The SIR model differential equations.



def deriv(y, t, N, beta, gamma):

    S, I, R = y

    dSdt = -beta * S * I / N

    dIdt = beta * S * I / N - gamma * I

    dRdt = gamma * I

    return dSdt, dIdt, dRdt



def sir_model(N=1000, beta=0.35, gamma=1/21):

    # Total population, N.

    N = N/50

    # Initial number of infected and recovered individuals, I0 and R0.

    I0, R0 = 1.001, 0.0001

    # Everyone else, S0, is susceptible to infection initially.



    S0 = N - I0 - R0

    # A grid of time points (in days)

    t = np.linspace(0, 160, 160)





    # Initial conditions vector

    y0 = S0, I0, R0

    # Integrate the SIR equations over the time grid, t.

    ret = odeint(deriv, y0, t, args=(N, beta, gamma))

    S, I, R = ret.T

    

    return S, I, R





Tstart = 0

# Contact rate, beta, and mean recovery rate, gamma, (in 1/days).

beta, gamma = 0.35, 1./(21)



t = np.linspace(0, 160, 160)

# Plot the data on three separate curves for S(t), I(t) and R(t)

fig, axes = plt.subplots(nrows=2,ncols=3,figsize=(25,15),facecolor='w', sharex=True, sharey=True)

i = -1

for i1, axx in enumerate(axes):

    for j1, ax in enumerate(axx):

        i += 1

        ax = ax #fig.add_subplot(111, axis_bgcolor='#dddddd', axisbelow=True)

        

        Npop = Npop_list[i]

        if i== 2:

            Npop=1200

        S, I, R = sir_model(N=Npop, beta=beta, gamma=gamma)

        

        if i == 0 :

            Tstart = 0 #-30

            ax.plot(china_hub_gdf['days_fromstart'], china_hub_gdf['dinf'], 'ro', alpha=0.25,  label='China: Hubei, Infected')

            ax.plot(china_hub_gdf['days_fromstart'], china_hub_gdf['drec'], 'go',  alpha=0.25, label='China: Hubei, Recovered')

        elif i==2 :

            Tstart = Tinc

            ax.plot(china_hk_gdf['days_fromstart'], china_hk_gdf['dinf'], 'ro', alpha=0.25,  label='China: HK, Infected')

            ax.plot(china_hk_gdf['days_fromstart'], china_hk_gdf['drec'], 'go',  alpha=0.25, label='China: HK, Recovered')



        elif i==1 :

            Tstart = 25+Tinc

            ax.plot(italy_df['days_fromstart'], italy_df['dinf'], 'ro', alpha=0.25,  label='Italy, Infected')

            ax.plot(italy_df['days_fromstart'], italy_df['drec'], 'go',  alpha=0.25, label='Italy, Recovered')  

        elif i==3 :

            Tstart = 35+Tinc

            ax.plot(uk_df['days_fromstart'], uk_df['dinf'], 'ro', alpha=0.25,  label='UK, Infected')

            ax.plot(uk_df['days_fromstart'], uk_df['drec'], 'go',  alpha=0.25, label='UK, Recovered')  

        elif i==4 :

            Tstart = 30+Tinc

            ax.plot(us_fed_df['days_fromstart'], us_fed_df['dinf'], 'ro', alpha=0.25,  label='US, Infected')

            ax.plot(us_fed_df['days_fromstart'], us_fed_df['drec'], 'go',  alpha=0.25, label='US, Recovered')  

        elif i==5 :

            Tstart = 45+Tinc

            ax.plot(india_df['days_fromstart'], india_df['dinf'], 'ro', alpha=0.25,  label='India, Infected')

            ax.plot(india_df['days_fromstart'], india_df['drec'], 'go',  alpha=0.25, label='India, Recovered')  



    #     ax.plot(Tstart+t, S/1000, 'b', alpha=0.5, lw=2, label='Susceptible')

        ax.plot(Tstart+t, I/1000, 'r', alpha=0.5, lw=2, label=f'Infected: (beta={beta})')

        ax.plot(Tstart+t, R/1000, 'g', alpha=0.5, lw=2, label=f'Recovered (1/{int(1/gamma)}days)')



        

        ax.set_ylabel(f'Scaled Numbers / {Npop}')

        ax.grid('on')

        legend = ax.legend(bbox_to_anchor=(0.5, 0.45))

        ax.set_yscale('log')



        if i1 >= 1:

            ax.set_xlabel('Time /days')

            



# ax.set_ylim(0,1.2)



ax.yaxis.set_tick_params(length=0)

ax.xaxis.set_tick_params(length=0)



plt.show()