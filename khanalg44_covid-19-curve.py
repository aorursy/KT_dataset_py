# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

! wget https://covid.ourworldindata.org/data/ecdc/total_deaths.csv
! wget https://covid.ourworldindata.org/data/ecdc/total_cases.csv

#dfd = pd.read_csv('./data/total_deaths.csv', parse_dates=True, index_col=0, skipfooter=1, engine='python')
#dfc = pd.read_csv('./data/total_cases.csv', parse_dates=True, index_col=0, skipfooter=1, engine='python')

dfd = pd.read_csv('./total_deaths.csv', parse_dates=True, index_col=0)
dfc = pd.read_csv('./total_cases.csv', parse_dates=True, index_col=0)

dfd = dfd.fillna(0)
dfc = dfc.fillna(0)
dfd.head()
dfc.head()
# To check all the countries in the data set
#for col in dfd.columns: 
#    print(col)

countries = ['World', 'China', 'France', 'Germany', 'Italy', 'Spain', 'United States']
N_countries = len(countries)
data_d = dfd[countries]
data_c = dfc[countries]

print ('total deaths')
data_d.tail()
print ('total cases')
data_c.tail()
import matplotlib.pyplot as plt
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()
plt.figure(figsize=(22,8))
for i in range(1, N_countries):
    plt.subplot(230+i)
    plt.title(countries[i], fontsize=20, color='red')
    plt.plot(data_c.index, data_c[countries[i]], label='Total Cases', color='blue')
    plt.legend(loc='upper left', frameon=False, fontsize=14)
    if i in [1,4] :
        plt.ylabel('Total', color='blue', fontsize=22)

    plt.twinx()
    plt.plot(data_d.index, data_d[countries[i]], label='Total Deaths', color='maroon')    
    plt.legend(loc='center left', frameon=False, fontsize=14)
    if i in [3, 6]:
        plt.ylabel('Death', color='maroon', fontsize=22)
def get_indices(N):
    index_N={}
    for i in range(1, N_countries):
        cases=0
        for j in range(len(data_d['China'])):
            cases += data_c[countries[i]][j]
            if cases >= N:
                index_N[countries[i]] = j
                break
    return index_N

Ncases = [100, 1000, 10000, 20000, 40000, 50000]

for i, N in enumerate(Ncases):
    idx_N=get_indices(N)
    print ("Days to reach", N, 'case:', idx_N)
def Plot_Milestones():
    Nmilestones = len(Ncases)
    Milestones=np.zeros((Nmilestones, N_countries-1), dtype=int)

    for i, Nm in enumerate(Ncases):
        idx_N=get_indices(Nm)
        Milestones[i,:] = list(idx_N.values())
        
    for i in range(N_countries-1):
        xx=range(Nmilestones)
        plt.plot(xx, Milestones[:,i], '-o', label=countries[i+1])
        plt.legend(loc='lower right', frameon=False, fontsize=12)
        plt.xticks(xx, Ncases)
        plt.xlabel("# Cases", fontsize=16)
        plt.ylabel("# Days", fontsize=16)
Plot_Milestones()
def Shifted_Origin(N):
    index_N = get_indices(N)
    rL = [25, 13, 85, 8, 11, 40]
    plt.figure(figsize=(20,10))
    for i in range(1, N_countries):
        plt.subplot(230+i)
        plt.title(countries[i], fontsize=30, color='red')
        indx=index_N[countries[i]]
        xx= range(len(data_c.index))

        plt.plot(xx[indx:], (1./rL[i-1])*data_c[countries[i]][indx:], label='Total Cases (x'+str(rL[i-1])+')', color='blue')
        plt.legend(loc='upper left', frameon=False, fontsize=16)
        if i in [1,4] :
            plt.ylabel('Total Cases', color='blue', fontsize=20)

        plt.twinx()
        plt.plot(xx[indx:], data_d[countries[i]][indx:], label='Total Deaths', color='maroon')

        plt.legend(loc='center left', frameon=False, fontsize=16)
        if i in [3,6] :
            plt.ylabel('Total Deaths', color='maroon', fontsize=20)

        xtL = [ data_c.index[ind].date().strftime("%m:%d") for ind in 
               [indx, indx+10, indx+20, indx+30, indx+40, indx+50, indx+60, indx+70, indx+80]
               if len(data_c.index) > ind ]
        xt = [indx+i*10 for i in range(len(xtL))]
        plt.xticks(xt, xtL, fontsize=20)
        plt.grid(axis='y')
Shifted_Origin(100)
Shifted_Origin(1000)
from scipy.optimize import curve_fit

def sigmoid(x, L ,x0, k):
    return  L / (1 + np.exp(-k*(x-x0)))

countries= ['World', 'China', 'Spain','United States' ]; N_countries = len(countries)
rr=[1., 1., 1.5, 1.5, 1.5, 1.5]

def FitTotal(dat, labs):
    for i in range(1, N_countries):    
        plt.subplot(140+i)
        plt.title(countries[i]+':'+labs, fontsize=20, color='red')
        
        yy = dat[countries[i]]
        xx = np.array(range(len(yy)))
    
        p0= [max(yy), rr[i]*np.mean(xx), 1.]
        popt, pcov =curve_fit(sigmoid, xx ,yy, p0)#, method='dogbox', maxfev=1000)        
        xnew = np.array(range(int( rr[i]*len(yy) )))
        
        plt.plot(xx, yy, 'o', color='blue', label='data')
        plt.plot(xnew, sigmoid(xnew, *popt), color='maroon', label='Fit')
        plt.legend(loc='upper left', frameon=False, fontsize=16)
        plt.xlabel('Days', fontsize=16)
        plt.grid(axis='y')

labs=[' Total Cases', ' Total Deaths']

for idd, dat in enumerate([data_c, data_d]):
    print 
    plt.figure(idd+1, figsize=(20, 3))
    FitTotal(dat, labs[idd])
def Gauss(x, mu, sigma, C):
    return C*1./(sigma * np.sqrt(2 * np.pi)) * np.exp( - (x - mu)**2 / (2 * sigma**2) )

countries= ['World', 'China', 'France', 'Germany', 'Italy', 'Spain', 'United States']
N_countries = len(countries)

data_c_day = data_c.diff().fillna(data_c)
data_d_day = data_d.diff().fillna(data_c)

def PlotPerDay():
    plt.figure(1, figsize=(14, 6))
    for i in range(1, N_countries):    
        plt.subplot(230+i)
        plt.title(countries[i], fontsize=20, color='red')
        xx = np.array(range(len(data_c_day[countries[i]].index)) )
        yy = data_c_day[countries[i]]
        plt.plot(xx, yy, 'o')

        mean = sum(xx * yy) / sum(yy)
        sigma = np.sqrt(sum(yy * (xx - mean)**2) / sum(yy))

        #print (mean, sigma)
        popt, pcov =curve_fit(Gauss, xx ,yy, p0=[mean, sigma, max(yy)])
        #print (popt)
        plt.plot(xx, Gauss(xx, *popt), 'r-')

PlotPerDay()

