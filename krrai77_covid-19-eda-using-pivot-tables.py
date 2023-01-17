#COVID19 is declared as pandemic by WHO
#import necessary libraries

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

from scipy.stats import norm

from sklearn.preprocessing import StandardScaler

from scipy import stats

import datetime 

%matplotlib inline

pd.set_option('display.max_rows', 500)



#load the data

covid_19 = pd.read_csv("../input/covid19/covid_19_clean_complete.csv")

covid_19.Date=pd.to_datetime(covid_19.Date)

covid_19.rename(columns={'Country/Region':'Country','Province/State':'State'},inplace=True)
covid_19['Year']=pd.DatetimeIndex(covid_19['Date']).year

covid_19['Month']=pd.DatetimeIndex(covid_19['Date']).month

covid_19['Day']=pd.DatetimeIndex(covid_19['Date']).day

covid_19["Month"].replace({1:'Jan',2:'Feb',3:'March'}, inplace=True)
covid = pd.pivot_table(data=covid_19, index=['Country','State','Month'], aggfunc=np.sum, values=['Confirmed', 'Deaths','Recovered'])
country=covid.index.get_level_values(0).unique()

for count in country:



    split=covid.xs(count)

    Confirm=split["Confirmed"]

    Death=split["Deaths"]

    Recover=split['Recovered']



    fig = plt.figure()

    ax1 = Confirm.plot(kind="bar")

    ax2 = ax1.twinx()

    ax2.plot(ax1.get_xticks(),Recover,linestyle='-',color="r")

    ax2.set_ylim((-5, 50.))

    Death.plot(color='green').legend()

    ax1.set_ylabel('Confirm', color='blue')

    ax2.set_ylabel('Recover', color='red')

    ax1.set_xlabel('State/Months')

    plt.title(count)




