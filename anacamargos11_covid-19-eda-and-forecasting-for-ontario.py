from IPython.display import Image

Image("../input/covidpic/covid2.jpg",width=800)
%matplotlib inline 

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

from matplotlib import cm

import os
brazil_df = pd.read_csv("/kaggle/input/covid19-global-and-regional/brazil_province_wise.csv", parse_dates=['Date'])

canadian_provinces_df = pd.read_csv("/kaggle/input/covid19-global-and-regional/canada_province_wise.csv", parse_dates=['Date'])

china_df = pd.read_csv("/kaggle/input/covid19-global-and-regional/china_province_wise.csv", parse_dates=['Date'])

italy_df = pd.read_csv("/kaggle/input/covid19-global-and-regional/italy_province_wise.csv", parse_dates=['Date'])

global_df = pd.read_csv("/kaggle/input/covid19-global-and-regional/covid_19_clean_complete.csv", parse_dates=['Date'])



brazil_df.set_index(brazil_df['Date'],drop=True,inplace=True)

canadian_provinces_df.set_index(canadian_provinces_df['Date'],drop=True,inplace=True)

china_df.set_index(china_df['Date'],drop=True,inplace=True)

italy_df.set_index(italy_df['Date'],drop=True,inplace=True)



brazil_df.drop(['Date'],axis=1, inplace=True)

canadian_provinces_df.drop(['Date'],axis=1, inplace=True)

china_df.drop(['Date'],axis=1, inplace=True)

italy_df.drop(['Date'],axis=1, inplace=True)



canada_df = canadian_provinces_df.groupby(['Date']).sum()
global_df.info()
canadian_provinces_df.tail()
plt.rcParams.update({'font.size': 15})

with plt.style.context('seaborn-white'):

    fig, ax = plt.subplots(2,2, figsize=(16,11))

    

brazil_df[['Confirmed','Deaths','Recovered']].plot(ax=ax[1,0],linestyle='--', linewidth=2.5)

canada_df[['Confirmed','Deaths']].plot(ax=ax[0,0], sharex=ax[0,0],linestyle='--', linewidth=2.5)

china_df.groupby(['Date']).sum()[['Confirmed','Deaths','Recovered']].plot(ax=ax[0,1],linestyle='--', linewidth=2.5)

italy_df[['Confirmed','Deaths','Recovered']].plot(ax=ax[1,1], sharex=ax[0,1],linestyle='--', linewidth=2.5)



ax[0,0].set_title('Canada')

ax[1,0].set_title('Brazil')

ax[0,1].set_title('China')

ax[1,1].set_title('Italy')



def make_yticklabel(tick_value, pos): 

    return "{}K".format(tick_value / 1000)



from matplotlib.ticker import FuncFormatter 

ax[0,0].yaxis.set_major_formatter(FuncFormatter(make_yticklabel))

ax[1,0].yaxis.set_major_formatter(FuncFormatter(make_yticklabel))

ax[0,1].yaxis.set_major_formatter(FuncFormatter(make_yticklabel))

ax[1,1].yaxis.set_major_formatter(FuncFormatter(make_yticklabel))



plt.tight_layout()
plt.rcParams.update({'font.size': 22})

df0 = global_df[global_df['Date']=='2020-04-24'].groupby(['Country/Region'])['Confirmed'].sum().sort_values(ascending=False)[1:15]

df0 = df0.sort_values(ascending=True)

df1 = global_df[global_df['Date']=='2020-04-24'].groupby(['Country/Region'])['Confirmed'].sum().sort_values(ascending=False)[0:15]

df1 = df1.sort_values(ascending=True)

with plt.style.context('seaborn-darkgrid'):

    fig, ax = plt.subplots(1,2)

    df0.plot.barh(ax=ax[0],title="COVID-19 Confirmed Cases - US excluded",figsize=(38,20),color=['black','orange','red','black','green','navy', 'red', 'green', 'red', 'navy','black','blue','green','goldenrod'])

    df1.plot.barh(ax=ax[1],title="COVID-19 Confirmed Cases including US",figsize=(38,20),color=['black','orange','red','black','green','navy', 'red', 'green', 'red', 'navy','black','blue','green','goldenrod','blue'])

    ax[0].set_ylabel(None)

    ax[1].set_ylabel(None)
df = global_df[global_df['Date']=='2020-04-24'].groupby(['Country/Region'])[['Recovered','Deaths']].sum().sort_values(by='Deaths',ascending=False)[0:11]

df = df.sort_values(by='Deaths',ascending=True)

df.drop(['Netherlands'], axis=0,inplace=True)

with plt.style.context('seaborn-poster'):

    df.plot.barh(title="COVID-19 Deaths and Recoveries",figsize=(14,6))
df = global_df[global_df['Date']=='2020-04-24'].groupby(['Country/Region'])[['Confirmed','Recovered','Deaths']].sum()

df['Deaths/Confirmed'] = df['Deaths']/df['Confirmed']

df = df.sort_values(by='Deaths/Confirmed',ascending=False)[0:45]

df = df.sort_values(by='Deaths/Confirmed',ascending=True)

with plt.style.context('seaborn-poster'):

    df['Deaths/Confirmed'].plot.barh(title="COVID-19 - Deaths/Confirmed ",figsize=(20,20))

plt.xticks(rotation=30,ha='right')

plt.show()
ax = canadian_provinces_df.loc['2020-04-24'][['Province/State','Confirmed']].sort_values(by='Confirmed',ascending=False)[0:9]

with plt.style.context('seaborn-poster'):

    ax.plot.bar(x='Province/State',title="COVID-19 Confirmed Cases in Canada",figsize=(15,6),color='navy')

plt.xticks(rotation=30,ha='right')

plt.show()
Image("../input/covid19sir/sir.png",width=800)
from scipy.integrate import odeint

from scipy.optimize import curve_fit

from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_squared_error
# population

N_ontario = 14446515 
# ontario recoveries table

on_recov_df = pd.read_csv('/kaggle/input/ontario-recovered/ontario_recovered.csv', parse_dates=['date_recovered'])

on_recov_df = on_recov_df[on_recov_df['province']=='Ontario']

on_recov_df.drop(['province'],axis=1,inplace=True)

on_recov_df.sort_values(by='date_recovered',inplace=True)

on_recov_df.rename(columns={'date_recovered': 'Date'},inplace=True)

on_recov_df.reset_index(drop=True,inplace=True)



# joining the ontario recoveries table to the general canadian provinces table

ontario_df = canadian_provinces_df[canadian_provinces_df['Province/State']=='Ontario'].iloc[0:,:]

ontario_df = ontario_df.merge(on_recov_df,how='inner',on='Date') 

ontario_df.drop(['Recovered'],axis=1,inplace=True)

ontario_df.rename(columns={"cumulative_recovered": "Recovered"},inplace=True)



# treating missing values on the resulting dataframe, and including day count data

ontario_df.fillna(0,inplace=True)

ontario_df['day_count'] = list(range(1,len(ontario_df)+1))
# defining the variables for the SIR model

ontario_df['Rec_immune'] = ontario_df['Deaths'] + ontario_df['Recovered']

ontario_df['Infected'] = ontario_df['Confirmed'] - ontario_df['Rec_immune']

ontario_df['Susceptible'] = N_ontario - ontario_df['Rec_immune'] - ontario_df['Infected']



# we need arrays for odeint

sus = np.array(ontario_df['Susceptible'],dtype=float)

infec = np.array(ontario_df['Infected'],dtype=float)

rec = np.array(ontario_df['Rec_immune'],dtype=float)



# splitting the data for validation purposes

x_train, x_test, y_train, y_test = train_test_split(ontario_df['day_count'],\

    ontario_df[['Susceptible','Infected','Rec_immune']],test_size=0.25,shuffle=False)

xtrain = np.array(x_train.iloc[0:],dtype=float)

ytrain = np.array(y_train.iloc[0:],dtype=float)

xtest = np.array(x_test.iloc[0:],dtype=float)

ytest = np.array(y_test.iloc[0:],dtype=float)



# forecasting data

tdata = np.array(ontario_df.day_count,dtype=float)

xcast = np.linspace(0,120,121)

ycast = np.array(ontario_df[['Susceptible','Infected','Rec_immune']],dtype=float)
# create model

def sir_model(z,t,beta,gamma):  

    dSdt = -(beta*z[1]*z[0])/N_ontario

    dIdt = (beta*z[1]*z[0])/N_ontario - gamma*z[1]

    dRdt = gamma*z[1]

    dzdt = [dSdt, dIdt, dRdt]

    return dzdt
# fit model to train set

z0 = [ytrain[0,0],ytrain[0,1],ytrain[0,2]]



def fit_odeint(t,beta,gamma):

    return odeint(sir_model,z0,t,args=(beta,gamma,))[:,1]



popt, pcov = curve_fit(fit_odeint,xtrain,ytrain[:,1],p0=[1,1])

fitted = fit_odeint(xtrain, *popt)



print("Optimal parameters: beta =", popt[0], " and gamma = ", popt[1])
# predict

xcast = np.linspace(0,120,121)

predicted = odeint(sir_model,ytrain[0,:],xcast,args=(popt[0],popt[1],))[:,1] 



poptcast, pcovcast = curve_fit(fit_odeint,tdata,ycast[:,1],p0=[1,1])

forecast = fit_odeint(xcast,*poptcast)
# visualization

with plt.style.context('seaborn-white'):

    fig, ax = plt.subplots(1,1, figsize=(16,11))

plt.plot(xtrain, ytrain[:,1], 'o',label="Feb-12 to Apr-07",color='lightcoral')

plt.plot(xtest, ytest[:,1], 'o',label="Apr-07 to Apr-28",color='cornflowerblue')

plt.plot(xcast, predicted,label="Original curve",color='lightcoral')

plt.plot(xcast, forecast,label="Flattened curve",color='cornflowerblue')

plt.axvline(x=xtest[-1], ls='--',color='gray')

plt.legend(bbox_to_anchor=(0.7, 1), loc='upper left', borderaxespad=0.)

plt.title("Fit of SIR model: COVID-19 infections in Ontario")

plt.ylabel("Population infected")

plt.xlabel("Days")

plt.show()