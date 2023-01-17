# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



from scipy.optimize import curve_fit



import matplotlib.pyplot as plt

import matplotlib.dates as mdates

import matplotlib.ticker as mtick

from matplotlib.ticker import FuncFormatter



import seaborn as sns



import datetime

from datetime import timedelta  



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
train = pd.read_csv("../input/covid19-global-forecasting-week-2/train.csv")

train['Date'] = train['Date'].apply(lambda x: (datetime.datetime.strptime(x, '%Y-%m-%d')))



train['NewFatalities'] = train['Fatalities'].diff(1)/1

train['NewCases'] = train['ConfirmedCases'].diff(1)/1



#display(train.head(5))



print("Count of Country_Region: ", train['Country_Region'].nunique())



print("Countries with Province/State: ", train[train['Province_State'].isna()==False]['Country_Region'].unique())



print("Date range: ", min(train['Date']), " - ", max(train['Date']))

#create two new columns: 'Region' and 'State' to bring European countries into a single region. 

#All other Country_Regions with Province_State are also captured in these two columns Region, State



Europe=[

    'Austria', 

    'Belgium', 

    'Denmark', 

    'Finland', 

    'France', 

    'Germany', 

    'Greece', 

    'Iceland', 

    'Ireland', 

    'Italy', 

    'Netherlands',

    'Norway', 

    'Portugal', 

    'Spain', 

    'Sweden', 

    'Switzerland', 

    'United Kingdom'

]





train['State'] = train['Province_State']

train['Region'] = train['Country_Region']



train.loc[train['Country_Region'].isin(Europe),'Region']='EU'

train.loc[train['Country_Region'].isin(Europe),'State']=train.loc[train['Country_Region'].isin(Europe),'Country_Region']
REGION = 'US'

CUTOFF = 25



c = train[train['Region']==REGION]

c = c.groupby(['Region','State','Date']).sum().reset_index()



#find the list of States with fatalities above cutoff 

states = c[c['Fatalities']>CUTOFF]['State'].unique()

display(states)



#find the first date when the fatalities cutoff was reached by a country in the region

s = c[c['State'].isin(states)]

minDate = s[s['Fatalities']>CUTOFF]['Date'].min()

print(CUTOFF, " deaths reached for a country on ", minDate)



s = s[s['Date']>minDate]

s['Days'] = (s['Date'] - minDate) / np.timedelta64(1, 'D')



display(s.head())



fig,axs = plt.subplots(nrows=2, ncols=2,figsize=[16,16])

plt.tight_layout()



g = sns.lineplot(data=s,x='Date',y='Fatalities',hue='State',ax=axs[0,0])

g.set(yscale='log')



g = sns.lineplot(data=s,x='Date',y='NewFatalities',hue='State',ax=axs[0,1])



g = sns.lineplot(data=s,x='Date',y='ConfirmedCases',hue='State',ax=axs[1,0])

#g.set(yscale='log')



g = sns.lineplot(data=s,x='Date',y='NewCases',hue='State',ax=axs[1,1])



#locator = mdates.DayLocator(interval=5)

#axs[0,0].xaxis.set_major_locator(locator)

#axs[0,1].xaxis.set_major_locator(locator)

#axs[1,0].xaxis.set_major_locator(locator)

#axs[1,1].xaxis.set_major_locator(locator)

fig.autofmt_xdate()



plt.show()
def expfunc(x, a, b, c):

    return a * np.exp(b * x) #+ c



#cubic supports an initial acceleration followed by deccelarion

def polyfunc(x, a, b, c, d):

    return a * x**3 + b * x**2 + c * x + d



def linfunc(x, a, b):

    return a * x + b



#STATE = 'France'

#STATE = 'Italy'

#STATE = 'California'

STATE = 'New York'

#STATE = 'Hubei'



CUTOFF = 10



#filter the data and keep the given STATE only

c = train[train['State']==STATE]

c = c.groupby(['Date']).sum().reset_index()



#find the first date when the fatalities cutoff was reached by this STATE

minDate = c[c['Fatalities']>CUTOFF]['Date'].min()

print(CUTOFF, " deaths reached on ", minDate)



s1 = c[c['Date']>minDate].copy()



#calculate the number of days since the first day fatalities exceeded the cutoff

s1['Days'] = (s1['Date'] - minDate) / np.timedelta64(1, 'D')

x = s1['Days']



#Smooth the daily NewFatalities by calculating the moving average of 3 days.

#for i in range(1,s1.shape[0]-1):

#    s1.loc[s1.index[i],'smoothed'] = (s1['NewFatalities'].iloc[i-1]+ s1['NewFatalities'].iloc[i] +s1['NewFatalities'].iloc[i+1]) / 3

#s1.loc[s1.index[0],'smoothed'] = s1['NewFatalities'].iloc[0]

#s1.loc[s1.index[-1],'smoothed'] = s1['NewFatalities'].iloc[-1]



#fit the cumulative FATALITIES curve with an EXPONENTIAL model and differentiate to obtain the daily fatalities

popt, pcov = curve_fit(expfunc, x, s1['Fatalities'])

s1['fit Fatalities (exp)'] = expfunc(x, *popt)

s1['fit NewFatalities (exp)'] = s1['fit Fatalities (exp)'].diff()



#fit the DAILY FATALITIES curve with a POLYNOMIAL model and integrate to obtain the cumulative cumulative fatalities

popt, pcov = curve_fit(polyfunc, x, s1['NewFatalities'])

s1['fit NewFatalities (poly)'] = polyfunc(x, *popt)

s1['fit Fatalities (poly)'] = s1['fit NewFatalities (poly)'].cumsum() + s1['Fatalities'].iloc[0]



#fit the CONFIRMED CASES withg an EXPONENTIAL model and differentiate to obtain the daily new cases

popt, pcov = curve_fit(expfunc, x, s1['ConfirmedCases'])

s1['fit Cases (exp)'] = expfunc(x, *popt)

s1['fit NewCases (exp)'] = s1['fit Cases (exp)'].diff()



#fit the DAILY NEW CASES curve with a POLYNOMIAL model and integrate to obtain the cumulative confirmed cases

popt, pcov = curve_fit(polyfunc, x, s1['NewCases'])

s1['fit NewCases (poly)'] = polyfunc(x, *popt)

s1['fit Cases (poly)'] = s1['fit NewCases (poly)'].cumsum() + s1['ConfirmedCases'].iloc[0]



display(s1.sort_values(by='Date',ascending=False))



fig,axs = plt.subplots(nrows=2, ncols=2,figsize=[16,16])



plt.subplot(221)

plt.title(STATE + ' Fatalities')

plt.plot(x, s1['Fatalities'],'ko-',label='Actual')

plt.plot(x, s1['fit Fatalities (exp)'],'r-',label='Exp')

plt.plot(x, s1['fit Fatalities (poly)'],'b-',label='Poly')

plt.legend()

plt.grid()

plt.yscale('log')



plt.subplot(222)

plt.title(STATE + ' New Fatalities')

plt.plot(s1['Date'], s1['NewFatalities'],'ko-',label='Actual')

#plt.plot(x, s1['smoothed'],'-',label='Smoothed')

plt.plot(s1['Date'], s1['fit NewFatalities (exp)'],'r-',label='Exp')

plt.plot(s1['Date'], s1['fit NewFatalities (poly)'],'b-',label='Poly')

plt.legend()

plt.grid()

#plt.yscale('log')



plt.subplot(223)

plt.title(STATE + ' Cases')

plt.plot(x, s1['ConfirmedCases'],'ko-',label='Actual')

plt.plot(x, s1['fit Cases (exp)'],'r-',label='Exp')

plt.plot(x, s1['fit Cases (poly)'],'b-',label='Poly')

plt.legend()

plt.grid()

plt.yscale('log')



plt.subplot(224)

plt.title(STATE + ' New Cases')

plt.plot(x, s1['NewCases'],'ko-',label='Actual')

plt.plot(x, s1['fit NewCases (exp)'],'r-',label='Exp')

plt.plot(x, s1['fit NewCases (poly)'],'b-',label='Poly')

plt.legend()

plt.grid()

#plt.yscale('log')





plt.show()



#check the relative rate of increase of fatalities

#it looks too noisy to calibrate on...

fig,ax = plt.subplots(figsize=[8,8])

plt.plot(x[1:], s1['Fatalities'].diff()[1:] / s1['Fatalities'][1:])

#-------------------------------------------------------

#params:

# array of number of days since inception (not used except to size output)

# i0 : initial percentage of infected population

# beta : daily rate of transmission by infected people to susceptible people, prior R0=2.7=beta/gamma

#death_rate prior=0.01/21.0  ; death rate of infected people (1% die about 3 weeks after infection)

#gamma prior=1.0/21  ; it takes three weeks to stop being infectious (either fully recovered, or dead)

#-------------------------------------------------------



def SIR(x, i0, beta, gamma, death_rate):

    

    y = np.empty((x.size,5))



    for i in range(0,x.size):

        

        if i==0:

            #initial conditions

            infected = i0

            susceptible = 1.0 - i0

            recovered = 0.0

            fatalities = 0.0       

            positives = i0

          

        else:

            #compute variations

            new_fatalities = death_rate * infected

            new_recovered = (gamma - death_rate) * infected

            new_positives = beta * susceptible * infected

            new_infected = beta * susceptible * infected - gamma * infected 

            new_susceptible = - beta * susceptible * infected

            

            #integrate and store in result array

            susceptible += new_susceptible

            positives += new_positives

            infected += new_infected

            recovered += new_recovered

            fatalities += new_fatalities

            

        y[i,0] = susceptible

        y[i,1] = infected

        y[i,2] = recovered

        y[i,3] = fatalities

        y[i,4] = positives  #cumul of infected, does not come down on recovery

            

    return y





x = np.arange(300)

y = SIR(x, i0=1e-6, beta=2.5/21, gamma=1.0/21, death_rate=0.01/21)



fig,axs = plt.subplots(nrows=1,ncols=2, figsize=[16,8])

plt.legend()

import matplotlib.ticker as mtick



lns1 = axs[0].plot(x, y[:,0],'g-',label='susceptible (lhs)')

lns2 = axs[0].plot(x, y[:,2],'b-',label='recovered (lhs)')

lns21 = axs[0].plot(x, y[:,4],'m-',label='positives (lhs)')

ax2 = axs[0].twinx() #instantiate second y axis, share same x axis

lns3 = ax2.plot(x, y[:,1],'r-',label='infected (rhs)')

lns4 = ax2.plot(x, y[:,3],'c-',label='fatalities (rhs)')

axs[0].yaxis.set_major_formatter(mtick.PercentFormatter(1.0))

ax2.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))



lns = lns1+lns2+lns21+lns3+lns4

labs = [l.get_label() for l in lns]

axs[0].legend(lns, labs, loc=0)

axs[0].grid()





lns1 = axs[1].plot(np.diff(y[:,1]),'r-',label='new infected (lhs)')

ax2 = axs[1].twinx() #instantiate second y axis, share same x axis

lns2 = ax2.plot(np.diff(y[:,3]),'c-',label='new fatalities (rhs)')

axs[1].yaxis.set_major_formatter(mtick.PercentFormatter(1.0))

ax2.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))



lns = lns1+lns2

labs = [l.get_label() for l in lns]

axs[1].legend(lns, labs, loc=0)

axs[1].grid()





#display(y)



#explore the dynamics of the model



fig,axs = plt.subplots(nrows=2,ncols=2, figsize=[16,8])

plt.legend()



i0 = 1e-6

beta = 2.5/14

gamma =1/21

deathrate = gamma*0.01



x = np.arange(365)



plt.subplot(221)

plt.title('Infection for I0 (per million)')

for ri0 in [1e-6, 5e-6, 10e-6, 100e-6]:

    y = SIR(x,ri0,beta,gamma,deathrate)

    plt.plot(x, y[:,1],'-',label="{:.1f}".format(ri0/1e-6))

plt.legend()

plt.grid()



plt.subplot(222)

plt.title('Fatalities dependency on Beta \n(label is R0 for gamma=1/21)')

for rbeta in [1.5, 1.75, 2, 2.5]:

    y = SIR(x,i0,rbeta*gamma,gamma,deathrate)

    plt.plot(x, y[:,3],'-',label="{:.2f}".format(rbeta*14))

plt.legend()

plt.grid()



plt.subplot(223)

plt.title('Infection for Gamma \n(label in days)')

for rgamma in [7,14,21,28]:

    y = SIR(x,i0,beta,1/rgamma,deathrate)

    plt.plot(x, y[:,1],'-',label="{:.0f}".format(rgamma))

plt.legend()

plt.grid()



plt.subplot(224)

plt.title('Fatalities for Death Rate')

for rdr in [0.5, 1, 1.5, 2]:

    y = SIR(x,i0,beta,gamma,rdr/100)

    plt.plot(x, y[:,3],'-',label="{:.1f}".format(rdr))

plt.legend()

plt.grid()
'''

OBSERVATIONS



The main effect of I0 is horizontal translation, it does not change the shape 



The main effect of Gamma is on the height of the peak of infection and final level of fatalities.

THIS IS NOT A GOOD PARAMETER TO USE WHEN CALIBRATING WITH ONLY EARLY DATA

It also has a material effect on the timing and width of the peak



The main effect of Beta at constant Gamma  is the curvature (acceleration of fatalities at the onset, and decceleration at the peak)



Death Rate is linear on severity



'''
##############################

####### CALIBRATION TO fatalities

##############################





Population = {

    'New York':20e6,

    'California':40e6,

    'France':67e6,

    'Italy':60e6,

    'Hubei':11e6 #wuhan=11, hubei=59 59e6

}



#STATE = 'Italy'

#STATE = 'France'

STATE = 'New York'

#STATE = 'California'

#STATE = 'Hubei'



POPULATION = Population[STATE]

CUTOFF = 1



#calibrate the model parameters to match actual fatality rates (assumption is that number of fatalities are best quality data)

def SIR_fatalities(x, i0, beta, gamma, death_rate):

    y = SIR(x, i0, beta=beta, gamma=gamma, death_rate=death_rate*gamma)

    return y[:,3] * POPULATION



'''

#calibration succeeds for NY,CA,IT,FR with CUTOFF=10 and data as of march 27, when using this set of bounds, but calibration is on boundaries 

#so, not very convincing...

I0_min = 1e-6

I0_max = 1000e-6

Gamma_min = 1/35

Gamma_max = 1/14

Beta_min = 1.1 * Gamma_min

Beta_max = 8 * Gamma_max

DeathRate_min = 0.5e-2

DeathRate_max = 2e-2

'''



#translational

I0_min = 1e-6

I0_max = 1000e-6



#use as narrow a range for gamma as can be established, as main effect is on total number of infected people and this cannot be calibrated with early data

Gamma_min = 1/24

Gamma_max = 1/14

Beta_min = 1.1 * Gamma_min

Beta_max = 6 * Gamma_max

DeathRate_min = 0.5e-2

DeathRate_max = 2e-2



bounds = ((I0_min, Beta_min, Gamma_min, DeathRate_min),(I0_max, Beta_max, Gamma_max, DeathRate_max))

initial_guess = [(I0_max+I0_min)/2, 2.5/21, 1/21,(DeathRate_max+DeathRate_min)/2]



#formatting functions for charts

def millions(x, pos):

    'The two args are the value and tick position'

    return '$%1.1fM' % (x * 1e-6)



#formatting functions for charts

def thousands(x, pos):

    'The two args are the value and tick position'

    return '$%1.0fT' % (x * 1e-3)



#filter the data and keep the given STATE only

c = train[train['State']==STATE]

c = c.groupby(['Date']).sum().reset_index()



#find the first date when the fatalities cutoff was reached by this STATE

minDate = c[c['Fatalities']>CUTOFF]['Date'].min()

print(CUTOFF, " deaths reached on ", minDate)



s1 = c[c['Date']>minDate].copy()



#calculate the number of days since the first day fatalities exceeded the cutoff

s1['Days'] = (s1['Date'] - minDate) / np.timedelta64(1, 'D')

x = s1['Days']



popt, pcov = curve_fit(SIR_fatalities, x, s1['Fatalities'], bounds=bounds,p0=initial_guess)



calib_I0 = popt[0]

calib_Beta = popt[1]

calib_Gamma = popt[2]

calib_DeathRate = popt[3]



s1['fit Fatalities (SIR)'] = SIR_fatalities(x, *popt)

s1['fit NewFatalities (SIR)'] = s1['fit Fatalities (SIR)'].diff()



y = POPULATION * SIR(x,i0=calib_I0, beta=calib_Beta, gamma=calib_Gamma, death_rate=calib_Gamma*calib_DeathRate)



s1['fit Cases (SIR)'] = y[:,4]  #reported stats are about new cases, they do not seem to account for people having recovered

s1['fit NewCases (SIR)'] = s1['fit Cases (SIR)'].diff()



print("SIR model fit")

print("I0 = {:,.0f} per million, or {:,.0f} persons initially infected".format(calib_I0 * 1e6, calib_I0*POPULATION))

print("BETA = {:.3f}".format(calib_Beta))

print("GAMMA = {:.3f}, or {:.1f} days to recover".format(calib_Gamma, 1/calib_Gamma))

print("DEATH RATE = {:.3%} infected people die".format(calib_DeathRate))

print("RHO = {:.2f}".format(calib_Beta/calib_Gamma))

#display(popt)



#display(s1.sort_values(by='Date',ascending=False))



fig,axs = plt.subplots(nrows=3, ncols=2,figsize=[16,16])



plt.subplot(321)

plt.title(STATE + ' Fatalities')

plt.plot(x, s1['Fatalities'],'ko-',label='Actual')

plt.plot(x, s1['fit Fatalities (SIR)'],'b-',label='SIR')

plt.legend()

plt.grid()

#plt.yscale('log')



ax = plt.subplot(322)

plt.title(STATE + ' New Fatalities')

plt.plot(x, s1['NewFatalities'],'ko-',label='Actual')

plt.plot(x, s1['fit NewFatalities (SIR)'],'b-',label='SIR')

plt.legend()

plt.grid()

#plt.yscale('log')



ax = plt.subplot(323)

plt.title(STATE + ' Confirmed Cases')

plt.plot(x, s1['ConfirmedCases'],'ko-',label='Actual')

plt.plot(x, s1['fit Cases (SIR)'],'b-',label='SIR')

plt.legend()

plt.grid()

plt.yscale('log')

ax.yaxis.set_major_formatter(FuncFormatter(thousands))



ax = plt.subplot(324)

plt.title(STATE + ' New Cases')

plt.plot(x, s1['NewCases'],'ko-',label='Actual')

plt.plot(x, s1['fit NewCases (SIR)'],'b-',label='SIR')

plt.legend()

plt.grid()

plt.yscale('log')

ax.yaxis.set_major_formatter(FuncFormatter(thousands))



xx = np.arange(180)

y = POPULATION * SIR(xx, i0=calib_I0, beta=calib_Beta, gamma=calib_Gamma, death_rate=calib_Gamma*calib_DeathRate) 



idx = np.argmax(y[:,1]).item()

print("Confirmed Cases would peak day {:,} after the cutoff, on {:%Y-%m-%d}".format(idx,minDate+timedelta(days=idx)))



ax = plt.subplot(325)

plt.title(STATE + ' Forecast')

plt.plot(xx, y[:,0],'g-',label='Susceptible')

plt.plot(xx, y[:,1],'r-',label='Infected')

plt.plot(xx, y[:,2],'b-',label='Recovered')

plt.plot(xx[1:], np.diff(y[:,1]),'m-',label='Daily New Infections')

ax.yaxis.set_major_formatter(FuncFormatter(millions))

plt.legend()

plt.grid()

#plt.yscale('log')



ax = plt.subplot(326)

plt.title(STATE + ' Forecast')

lns2 = plt.plot(xx, y[:,3],'c-',label='Fatalities (lhs)')

ax.yaxis.set_major_formatter(FuncFormatter(thousands))



ax2 = ax.twinx() #instantiate second y axis, share same x axis

lns3 = plt.plot(xx[1:], np.diff(y[:,3]),'m-',label='Daily Fatalities (rhs)')

ax2.yaxis.set_major_formatter(FuncFormatter(thousands))



lns = lns2+lns3

labs = [l.get_label() for l in lns]

ax.legend(lns, labs, loc=0)

ax.grid()

#plt.yscale('log')



plt.show()
'''

OBSERVATIONS



the model does not fit Hubei data, need stupid parameters to match the fatalities curve

china confirmed cases do not account for recoveries, it is the cumul of positive tests



fitting on Fatalities results in forecast of Confirmed Cases one or two orders of magnitude above reported numbers.

dynamics also 

'''
####################

## CALIBRATION TO BOTH fatalities and confirmedcases

####################



#STATE = 'Italy'

#STATE = 'France'

STATE = 'New York'

#STATE = 'California'

#STATE = 'Hubei'



POPULATION = Population[STATE]

CUTOFF = 1



#calibrate the model parameters to match actual fatality rates (assumption is that number of fatalities are best quality data)

#in real life, stats about confirmed cases do not appear to take recovered cases into account

def SIR_confirmed_fatalities(x, i0, beta, gamma, death_rate):  

    y = SIR(x, i0, beta=beta, gamma=gamma, death_rate=death_rate*gamma)

    return POPULATION * np.append(y[:,4], y[:,3])



I0_min = 1e-6

I0_max = 10000e-6

Gamma_min = 1/24

Gamma_max = 1/7

Beta_min = 1.1 * Gamma_min

Beta_max = 6 * Gamma_max

DeathRate_min = 0.5e-2

DeathRate_max = 10e-2



bounds = ((I0_min, Beta_min, Gamma_min, DeathRate_min),(I0_max, Beta_max, Gamma_max, DeathRate_max))

initial_guess = [2000e-6, 2.5/21, 1/21,1e-2]



#calibrate on both ConfirmedCases adn Fatalities curves

z = s1['ConfirmedCases']

z = z.append(s1['Fatalities'])

popt, pcov = curve_fit(SIR_confirmed_fatalities, x, z, bounds=bounds,p0=initial_guess)



calib_I0 = popt[0]

calib_Beta = popt[1]

calib_Gamma = popt[2]

calib_DeathRate = popt[3]



y = POPULATION * SIR(x,i0=calib_I0, beta=calib_Beta, gamma=calib_Gamma, death_rate=calib_Gamma*calib_DeathRate)



s1['fit Fatalities (SIR)'] = y[:,3]

s1['fit NewFatalities (SIR)'] = s1['fit Fatalities (SIR)'].diff()



s1['fit Cases (SIR)'] = y[:,4]  #reported stats are about new cases, they do not seem to account for people having recovered

s1['fit NewCases (SIR)'] = s1['fit Cases (SIR)'].diff()



print("SIR model fit")

print("I0 = {:,.0f} per million, or {:,.0f} persons initially infected".format(calib_I0 * 1e6, calib_I0*POPULATION))

print("BETA = {:.3f}".format(calib_Beta))

print("GAMMA = {:.3f}, or {:.1f} days to recover".format(calib_Gamma, 1/calib_Gamma))

print("DEATH RATE = {:.3%} infected people die".format(calib_DeathRate))

print("RHO = {:.2f}".format(calib_Beta/calib_Gamma))

#display(popt)



#display(s1.sort_values(by='Date',ascending=False))



fig,axs = plt.subplots(nrows=3, ncols=2,figsize=[16,16])



plt.subplot(321)

plt.title(STATE + ' Fatalities')

plt.plot(x, s1['Fatalities'],'ko-',label='Actual')

plt.plot(x, s1['fit Fatalities (SIR)'],'b-',label='SIR')

plt.legend()

plt.grid()

#plt.yscale('log')



ax = plt.subplot(322)

plt.title(STATE + ' New Fatalities')

plt.plot(x, s1['NewFatalities'],'ko-',label='Actual')

plt.plot(x, s1['fit NewFatalities (SIR)'],'b-',label='SIR')

plt.legend()

plt.grid()

#plt.yscale('log')



ax = plt.subplot(323)

plt.title(STATE + ' Confirmed Cases')

plt.plot(x, s1['ConfirmedCases'],'ko-',label='Actual')

plt.plot(x, s1['fit Cases (SIR)'],'b-',label='SIR')

plt.legend()

plt.grid()

plt.yscale('log')

ax.yaxis.set_major_formatter(FuncFormatter(thousands))



ax = plt.subplot(324)

plt.title(STATE + ' New Cases')

plt.plot(x, s1['NewCases'],'ko-',label='Actual')

plt.plot(x, s1['fit NewCases (SIR)'],'b-',label='SIR')

plt.legend()

plt.grid()

plt.yscale('log')

ax.yaxis.set_major_formatter(FuncFormatter(thousands))



xx = np.arange(180)

y = POPULATION * SIR(xx, i0=calib_I0, beta=calib_Beta, gamma=calib_Gamma, death_rate=calib_Gamma*calib_DeathRate) 



idx = np.argmax(y[:,1]).item()

print("Confirmed Cases would peak day {:,} after the cutoff, on {:%Y-%m-%d}".format(idx,minDate+timedelta(days=idx)))



ax = plt.subplot(325)

plt.title(STATE + ' Forecast')

plt.plot(xx, y[:,0],'g-',label='Susceptible')

plt.plot(xx, y[:,1],'r-',label='Infected')

plt.plot(xx, y[:,2],'b-',label='Recovered')

plt.plot(xx[1:], np.diff(y[:,1]),'m-',label='Daily New Infections')

ax.yaxis.set_major_formatter(FuncFormatter(millions))

plt.legend()

plt.grid()

#plt.yscale('log')



ax = plt.subplot(326)

plt.title(STATE + ' Forecast')

lns2 = plt.plot(xx, y[:,3],'c-',label='Fatalities (lhs)')

ax.yaxis.set_major_formatter(FuncFormatter(thousands))



ax2 = ax.twinx() #instantiate second y axis, share same x axis

lns3 = plt.plot(xx[1:], np.diff(y[:,3]),'m-',label='Daily Fatalities (rhs)')

ax2.yaxis.set_major_formatter(FuncFormatter(thousands))



lns = lns2+lns3

labs = [l.get_label() for l in lns]

ax.legend(lns, labs, loc=0)

ax.grid()

#plt.yscale('log')



plt.show()
'''

OBSERVATION

while it is possible to fit both Fatalities and Confirmed Cases, the calibrated resuls do not appear credible (death rate is too high, time to recovery is too low)

this could support the hypothesis that observed confirmed cases are underestimated by a large margin

'''
####################################

##CALIBRATION TO FATALITIES

##POPULATION AS A FREE PARAMETER

####################################



#STATE = 'Italy'

#STATE = 'France'

STATE = 'New York'

#STATE = 'California'

#STATE = 'Hubei'



CUTOFF = 10



#calibrate the model parameters to match actual fatality rates (assumption is that number of fatalities are best quality data)

#the model also calibrate the population size

def SIR_fatalities(x, i0, beta, gamma, death_rate,population):

    y = SIR(x, i0, beta=beta, gamma=gamma, death_rate=death_rate*gamma)

    return y[:,3] * population



'''

#calibration succeeds for NY,CA,IT,FR with CUTOFF=10 and data as of march 27, when using this set of bounds, but calibration is on boundaries 

#so, not very convincing...

I0_min = 1e-6

I0_max = 1000e-6

Gamma_min = 1/35

Gamma_max = 1/14

Beta_min = 1.1 * Gamma_min

Beta_max = 8 * Gamma_max

DeathRate_min = 0.5e-2

DeathRate_max = 2e-2

'''



I0_min = 1e-6

I0_max = 1000e-6

#Gamma_min = 1/42

#Gamma_max = 1/7

Gamma_min = 1/42

Gamma_max = 1/7



Beta_min = 1.1 * Gamma_min

Beta_max = 6 * Gamma_max

DeathRate_min = 0.5e-2

DeathRate_max = 2e-2

Population_min = 1e6

Population_max = 100e6



bounds = ((I0_min, Beta_min, Gamma_min, DeathRate_min, Population_min),(I0_max, Beta_max, Gamma_max, DeathRate_max, Population_max))

initial_guess = [(I0_max+I0_min)/2, 2.5/21, 1/21,(DeathRate_max+DeathRate_min)/2,(Population_max+Population_min)/2]



#formatting functions for charts

def millions(x, pos):

    'The two args are the value and tick position'

    return '$%1.1fM' % (x * 1e-6)



#formatting functions for charts

def thousands(x, pos):

    'The two args are the value and tick position'

    return '$%1.0fT' % (x * 1e-3)



#filter the data and keep the given STATE only

c = train[train['State']==STATE]

c = c.groupby(['Date']).sum().reset_index()



#find the first date when the fatalities cutoff was reached by this STATE

minDate = c[c['Fatalities']>CUTOFF]['Date'].min()

print(CUTOFF, " deaths reached on ", minDate)



s1 = c[c['Date']>minDate].copy()



#calculate the number of days since the first day fatalities exceeded the cutoff

s1['Days'] = (s1['Date'] - minDate) / np.timedelta64(1, 'D')

x = s1['Days']



popt, pcov = curve_fit(SIR_fatalities, x, s1['Fatalities'], bounds=bounds,p0=initial_guess)



calib_I0 = popt[0]

calib_Beta = popt[1]

calib_Gamma = popt[2]

calib_DeathRate = popt[3]

calib_Population = popt[4]



s1['fit Fatalities (SIR)'] = SIR_fatalities(x, *popt)

s1['fit NewFatalities (SIR)'] = s1['fit Fatalities (SIR)'].diff()



y = calib_Population * SIR(x,i0=calib_I0, beta=calib_Beta, gamma=calib_Gamma, death_rate=calib_Gamma*calib_DeathRate)



s1['fit Cases (SIR)'] = y[:,4]  #reported stats are about new cases, they do not seem to account for people having recovered

s1['fit NewCases (SIR)'] = s1['fit Cases (SIR)'].diff()



print("SIR model fit")

print("POPULATION = {:,.0f} million".format(calib_Population / 1e6))

print("I0 = {:,.0f} per million, or {:,.0f} persons initially infected".format(calib_I0 * 1e6, calib_I0*calib_Population))

print("BETA = {:.3f}".format(calib_Beta))

print("GAMMA = {:.3f}, or {:.1f} days to recover".format(calib_Gamma, 1/calib_Gamma))

print("DEATH RATE = {:.3%} infected people die".format(calib_DeathRate))

print("RHO = {:.2f}".format(calib_Beta/calib_Gamma))

#display(popt)



#display(s1.sort_values(by='Date',ascending=False))



fig,axs = plt.subplots(nrows=3, ncols=2,figsize=[16,16])



plt.subplot(321)

plt.title(STATE + ' Fatalities')

plt.plot(x, s1['Fatalities'],'ko-',label='Actual')

plt.plot(x, s1['fit Fatalities (SIR)'],'b-',label='SIR')

plt.legend()

plt.grid()

#plt.yscale('log')



ax = plt.subplot(322)

plt.title(STATE + ' New Fatalities')

plt.plot(x, s1['NewFatalities'],'ko-',label='Actual')

plt.plot(x, s1['fit NewFatalities (SIR)'],'b-',label='SIR')

plt.legend()

plt.grid()

#plt.yscale('log')



ax = plt.subplot(323)

plt.title(STATE + ' Confirmed Cases')

plt.plot(x, s1['ConfirmedCases'],'ko-',label='Actual')

plt.plot(x, s1['fit Cases (SIR)'],'b-',label='SIR')

plt.legend()

plt.grid()

plt.yscale('log')

ax.yaxis.set_major_formatter(FuncFormatter(thousands))



ax = plt.subplot(324)

plt.title(STATE + ' New Cases')

plt.plot(x, s1['NewCases'],'ko-',label='Actual')

plt.plot(x, s1['fit NewCases (SIR)'],'b-',label='SIR')

plt.legend()

plt.grid()

plt.yscale('log')

ax.yaxis.set_major_formatter(FuncFormatter(thousands))



xx = np.arange(180)

y = calib_Population * SIR(xx, i0=calib_I0, beta=calib_Beta, gamma=calib_Gamma, death_rate=calib_Gamma*calib_DeathRate) 



idx = np.argmax(y[:,1]).item()

print("Confirmed Cases would peak day {:,} after the cutoff, on {:%Y-%m-%d}".format(idx,minDate+timedelta(days=idx)))



ax = plt.subplot(325)

plt.title(STATE + ' Forecast')

plt.plot(xx, y[:,0],'g-',label='Susceptible')

plt.plot(xx, y[:,1],'r-',label='Infected')

plt.plot(xx, y[:,4],'b-',label='Positives')

plt.plot(xx[1:], np.diff(y[:,1]),'m-',label='Daily New Infections')

ax.yaxis.set_major_formatter(FuncFormatter(millions))

plt.legend()

plt.grid()

#plt.yscale('log')



ax = plt.subplot(326)

plt.title(STATE + ' Forecast')

lns2 = plt.plot(xx, y[:,3],'c-',label='Fatalities (lhs)')

ax.yaxis.set_major_formatter(FuncFormatter(thousands))



ax2 = ax.twinx() #instantiate second y axis, share same x axis

lns3 = plt.plot(xx[1:], np.diff(y[:,3]),'m-',label='Daily Fatalities (rhs)')

ax2.yaxis.set_major_formatter(FuncFormatter(thousands))



lns = lns2+lns3

labs = [l.get_label() for l in lns]

ax.legend(lns, labs, loc=0)

ax.grid()

#plt.yscale('log')



plt.show()
'''

-------------------------------------------------------

DELAYED SIR MODEL



    compartments are introduced to capture the progression from incubating to recovery over time.

    this could be useful to better capture the delay between reported fatalities and infection time

    all buckets have the same lengh of time, it should  be 5 to 7 days; this param is used when sampling the data before calibration



    susceptible

        -> incub susceptible * (symptomatic+moderate1+moderate2)

            -> symptomatic 

                20% -> hospital1

                        -> hospital2

                            10% -> death

                            90% -> resolved

                80% -> moderate1

                        -> moderate2

                            -> resolved



-------------------------------------------------------

'''



#columns of the array returned by SIR2()

cSu  = 0  #suceptible compartment

cI   = 1  #incubation period, not contagious (from cSu * cSy, cM1, cM2)

cSy  = 2  #firs period of contagion / symptom (from cI)

cM1  = 3  #second period of contagion / stay home (80% from cSy)

cM2  = 4  #last period of contagion / stay home (from cM1)

cH1  = 5  #first period of hospitalization (20% from cSy)

cH2  = 6  #second period of hospitalization (from cH1)

cR   = 7  #recovery (90% from cH2, all from cM2)

cF   = 8  #fatalities (10% from cH2)

cC   = 9  #cumulative number of confirmed cases

columnTitles=["Su","I","Sy","M1","M2","H1","H2","R","F","C"]       



def SIR2(x, incubating0, symptomaticInfectionRate, moderateInfectionRate, criticalRate, deathRate):

    

    '''

    '''         

    def propagate(state):



        y = np.zeros(10)

        

        new_fatalities = deathRate * state[cH2]  #about 10% of hospitalized people die after the second week in hospital

        new_recovered = (1-deathRate) * state[cH2] + state[cM2]  #survivors are discharged from hospital after two weeks, and moderate cases are recovered after two week 

        new_hospitalized2 = state[cH1] #hospitalized people in their first week move to their second week

        new_moderate2 = state[cM1] #moderate cases in their first week move to their second week

        new_hospitalized1 = criticalRate * state[cSy] #about 10% of people need critical care at the end of the first week after symptom onset

        new_moderate1 = (1-criticalRate) * state[cSy] #people who have moderate to no symptoms after first week continue to stay at home

        new_symptomatic = state[cI]  #onset of symptoms and become contagious after period of incubation

        new_incubating = state[cSu] * (symptomaticInfectionRate*state[cSy] + moderateInfectionRate*(state[cM1]+state[cM2]))  #new infections from contagious sources (assume no contagion at hospital, not enough data to calibrate this param)...



        y[cF] = state[cF] + new_fatalities

        y[cR] = new_recovered

        y[cM2] = new_moderate2

        y[cM1] = new_moderate1

        y[cH2] = new_hospitalized2

        y[cH1] = new_hospitalized1

        y[cSy] = new_symptomatic

        y[cI] = new_incubating

        y[cC] = state[cC] + new_symptomatic  #confirmed cases: assume all symptomatic people are tested; recovered people are not reported in official stats

        y[cSu]  = max(0, state[cSu] - new_incubating)



        return y

    

    y = np.zeros((x.size,10))



    for i in range(0,x.size):

        

        if i==0:

            

            #introduce the initially incubating population and run the algo a few times to remove the early oscillations

            y[0,cI] = 4*incubating0 

            y[0,cSy] = 2*incubating0 

            y[0,cM1] = incubating0 

            y[0,cSu] = 1-7*incubating0 

            y[0] = propagate(y[0])

            y[0] = propagate(y[0])

            y[0] = propagate(y[0])

            y[0] = propagate(y[0])

            y[0] = propagate(y[0])

            y[0] = propagate(y[0])

            y[0] = propagate(y[0])

            

        else:

            y[i] = propagate(y[i-1])



    return y







def plotSIR2(x,y):

    fig,axs = plt.subplots(nrows=1,ncols=2, figsize=[16,8])

    plt.legend()



    lns1 = axs[0].plot(x, y[:,cSu],'g-',label='susceptible (lhs)')

    lns2 = axs[0].plot(x, y[:,cR],'b-',label='recovered (lhs)')

    lns21 = axs[0].plot(x, y[:,cC],'m-',label='positives (lhs)')

    ax2 = axs[0].twinx() #instantiate second y axis, share same x axis

    lns3 = ax2.plot(x, y[:,cI],'r-',label='incubating (rhs)')

    lns4 = ax2.plot(x, y[:,cM1] + y[:,cM2],'y-',label='moderate1+2 (rhs)')

    lns5 = ax2.plot(x, y[:,cH1] + y[:,cH2],'k-',label='hospitalized1+2 (rhs)')

    lns6 = ax2.plot(x, y[:,cF],'c-',label='fatalities (rhs)')

    axs[0].yaxis.set_major_formatter(mtick.PercentFormatter(1.0))

    ax2.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))



    lns = lns1+lns2+lns21+lns3+lns4+lns5+lns6

    labs = [l.get_label() for l in lns]

    axs[0].legend(lns, labs, loc=0)

    axs[0].grid()





    lns1 = axs[1].plot(np.diff(y[:,cI]),'r-',label='new incubating (lhs)')

    ax2 = axs[1].twinx() #instantiate second y axis, share same x axis

    lns2 = ax2.plot(np.diff(y[:,cF]),'c-',label='new fatalities (rhs)')

    axs[1].yaxis.set_major_formatter(mtick.PercentFormatter(1.0))

    ax2.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))



    lns = lns1+lns2

    labs = [l.get_label() for l in lns]

    axs[1].legend(lns, labs, loc=0)

    axs[1].grid()

    

    plt.show()



    

x = np.arange(20)

y = SIR2(x, incubating0=0.001e-6, symptomaticInfectionRate=2, moderateInfectionRate=1, criticalRate=0.2, deathRate=0.1 )

    

res = pd.DataFrame(y*1e6, columns=columnTitles)

res["dI"] = res['I'].diff()

display(res.style.format("{:.4f}"))



plotSIR2(x,y)



print("Zoom on first 5 periods")

plotSIR2(x[:5], y[:5,:]) 

#############################

#DELAYED SIR

#############################



#STATE = 'Italy'

STATE = 'France'

#STATE = 'New York'

#STATE = 'California'

#STATE = 'Hubei'



CUTOFF = 1



sampleFreq = 5



Population = {

    'New York':20e6,

    'California':40e6,

    'France':67e6,

    'Italy':60e6,

    'Hubei':11e6 #wuhan=11, hubei=59 59e6

}

calib_population = Population[STATE]



calib_criticalRate =0.15  #about 15% of infected people will require hospitalization

calib_deathRate = 0.1  #about 10% of hospitalized people will die



#calibrate the model parameters to match actual fatality rates (assumption is that number of fatalities are best quality data)

#the model also calibrate the population size

def SIR2_fatalities(x, incubating0, infectionRate):

    

    y = SIR2(x, 

             incubating0=incubating0, 

             symptomaticInfectionRate=infectionRate, 

             moderateInfectionRate=infectionRate, 

             criticalRate=calib_criticalRate, 

             deathRate=calib_deathRate)

    

    return y[:,cF] * calib_population



'''

'''



incubating0_min = 0

incubating0_max = 10000e-6



infectionRate_min = 0.1

infectionRate_max = 10



bounds = ((incubating0_min, infectionRate_min),(incubating0_max, infectionRate_max))



#formatting functions for charts

def millions(x, pos):

    'The two args are the value and tick position'

    return '$%1.1fM' % (x * 1e-6)



#formatting functions for charts

def thousands(x, pos):

    'The two args are the value and tick position'

    return '$%1.0fT' % (x * 1e-3)



#filter the data and keep the given STATE only

c = train[train['State']==STATE]

c = c.groupby(['Date']).sum().reset_index()



#find the first date when the fatalities cutoff was reached by this STATE

minDate = c[c['Fatalities']>CUTOFF]['Date'].min()

print(CUTOFF, " deaths reached on ", minDate)



s1 = c[c['Date']>=minDate]

s1 = s1.iloc[0::sampleFreq, :].copy()  #weekly sample, starting on the cutoff

s1['NewFatalities'] = s1['Fatalities'].diff()  #replace original daily values after sampling

s1['NewConfirmedCases'] = s1['ConfirmedCases'].diff()





#calculate the number of days since the first day fatalities exceeded the cutoff

s1['Days'] = (s1['Date'] - minDate) / np.timedelta64(1, 'D')

x = s1['Days']



#display(s1[['Days','Fatalities','ConfirmedCases']])



popt, pcov = curve_fit(SIR2_fatalities, x, s1['Fatalities'], bounds=bounds) #,p0=initial_guess)



calib_incubating0 = popt[0]

calib_infectionRate = popt[1]





#calib_Population = 40e6

#calib_symptomaticInfectionRate = 4

#calib_moderateInfectionRate = 4

           

           

y = calib_population * SIR2(x, 

                            incubating0=calib_incubating0, 

                            symptomaticInfectionRate = calib_infectionRate, 

                            moderateInfectionRate = calib_infectionRate, 

                            criticalRate=calib_criticalRate, 

                            deathRate=calib_deathRate)



s1['fit Fatalities (SIR)'] = y[:,cF]  

s1['fit NewFatalities (SIR)'] = s1['fit Fatalities (SIR)'].diff()



s1['fit Cases (SIR)'] = y[:,cC]  #reported stats are about new cases, they do not seem to account for people having recovered

s1['fit NewCases (SIR)'] = s1['fit Cases (SIR)'].diff()



print("SIR2 model fit")



print("Assumption Population = {:,.0f} millions".format(calib_population / 1e6))

print("Assumption Hospitalization Rate = {:.1%}".format(calib_criticalRate))

print("Assumption Death Rate after Hospitalization = {:.1%}".format(calib_deathRate))



print("Calibrated Incubating0 = {:.1f} per million susceptible".format(calib_incubating0*1e6))

print("Calibrated Infection Rate = {:.1f}".format(calib_infectionRate))



display(s1.sort_values(by='Date',ascending=False))



fig,axs = plt.subplots(nrows=3, ncols=2,figsize=[16,16])



ax = plt.subplot(321)

plt.title(STATE + ' Fatalities')

plt.plot(x, s1['Fatalities'],'ko-',label='Actual')

plt.plot(x, s1['fit Fatalities (SIR)'],'b*-',label='SIR')

plt.legend()

plt.grid()

plt.xlabel('days since report of {:.0f} fatalities'.format(CUTOFF))

#plt.yscale('log')



ax = plt.subplot(322)

plt.title(STATE + ' New Fatalities')

plt.plot(x, s1['NewFatalities'],'ko-',label='Actual')

plt.plot(x, s1['fit NewFatalities (SIR)'],'b*-',label='SIR')

plt.legend()

plt.grid()

#plt.yscale('log')



ax = plt.subplot(323)

plt.title(STATE + ' Confirmed Cases')

plt.plot(x, s1['ConfirmedCases'],'ko-',label='Actual')

plt.plot(x, s1['fit Cases (SIR)'],'b*-',label='SIR')

plt.legend()

plt.grid()

plt.yscale('log')

ax.yaxis.set_major_formatter(FuncFormatter(thousands))



ax = plt.subplot(324)

plt.title(STATE + ' New Cases')

plt.plot(x, s1['NewCases'],'ko-',label='Actual')

plt.plot(x, s1['fit NewCases (SIR)'],'b*-',label='SIR')

plt.legend()

plt.grid()

plt.yscale('log')

ax.yaxis.set_major_formatter(FuncFormatter(thousands))



#forecast for 20 periods

xx = np.arange(20)*sampleFreq

y = calib_population * SIR2(xx, 

                            incubating0=calib_incubating0, 

                            symptomaticInfectionRate = calib_infectionRate, 

                            moderateInfectionRate = calib_infectionRate, 

                            criticalRate=calib_criticalRate, 

                            deathRate=calib_deathRate)



idx = np.argmax(y[:,cSy]).item()

print("New Confirmed Cases would peak period {:,} after the cutoff, on {:%Y-%m-%d}".format(idx,minDate+timedelta(days=sampleFreq*idx)))



ax = plt.subplot(325)

plt.title(STATE + ' Forecast')

plt.plot(xx, y[:,cSu],'g-',label='Susceptible')

plt.plot(xx, y[:,cI],'r-',label='Incubating')

plt.plot(xx, y[:,cC],'b-',label='Positives')

plt.plot(xx[1:], np.diff(y[:,cI]),'m-',label='New Infections')

ax.yaxis.set_major_formatter(FuncFormatter(millions))

plt.legend()

plt.grid()

#plt.yscale('log')



ax = plt.subplot(326)

plt.title(STATE + ' Forecast')

lns2 = plt.plot(xx, y[:,cF],'c-',label='Fatalities (lhs)')

ax.yaxis.set_major_formatter(FuncFormatter(thousands))



ax2 = ax.twinx() #instantiate second y axis, share same x axis

lns3 = plt.plot(xx[1:], np.diff(y[:,cF]),'m-',label='New Fatalities (rhs)')

ax2.yaxis.set_major_formatter(FuncFormatter(thousands))



lns = lns2+lns3

labs = [l.get_label() for l in lns]

ax.legend(lns, labs, loc=0)

ax.grid()

#plt.yscale('log')



plt.show()
#TO-DO : calibrate the model on different starting days, to use all the data



#############################

#DELAYED SIR

#############################



#STATE = 'Italy'

#STATE = 'France'

#STATE = 'New York'

#STATE = 'California'

STATE = 'Hubei'



CUTOFF = 1

sampleFreq = 5



Population = {

    'New York':20e6,

    'California':40e6,

    'France':67e6,

    'Italy':60e6,

    'Hubei':11e6 #wuhan=11, hubei=59 59e6

}

calib_population = Population[STATE]



calib_criticalRate =0.20  # 20% of symptomatic people will require hospitalization : high number because of biais in testing strategy

calib_deathRate = 0.05  # 5% of hospitalized people will die



#calibrate the model parameters to match actual fatality rates (assumption is that number of fatalities are best quality data)

#the model also calibrate the population size

def SIR2_multifatalities(data_days, incubating0, infectionRate, samples):



    #run the model with current parameters on each of the samples

    res = np.zeros(0)

    

    for x in samples:

        

        y = SIR2(np.array(x), 

                 incubating0=incubating0, 

                 symptomaticInfectionRate=infectionRate, 

                 moderateInfectionRate=infectionRate, 

                 criticalRate=calib_criticalRate, 

                 deathRate=calib_deathRate)

    

        res = np.append(res, y[:,cF].flatten())

    

    res *= calib_population

    return res



'''

good for EU, US

incubating0_min = 1e-6

incubating0_max = 10000e-6

infectionRate_min = 0.1

infectionRate_max = 10



initial_guess =(10e-6, 2)

'''



incubating0_min = 1000e-6

incubating0_max = 100000e-6

infectionRate_min = 0.1

infectionRate_max = 10

initial_guess =(10000e-6, 2)



bounds = ((incubating0_min, infectionRate_min),(incubating0_max, infectionRate_max))





#formatting functions for charts

def millions(x, pos):

    'The two args are the value and tick position'

    return '$%1.1fM' % (x * 1e-6)



#formatting functions for charts

def thousands(x, pos):

    'The two args are the value and tick position'

    return '$%1.0fT' % (x * 1e-3)



#filter the data and keep the given STATE only

c = train[train['State']==STATE]

c = c.groupby(['Date']).sum().reset_index()



#find the first date when the fatalities cutoff was reached by this STATE

minDate = c[c['Fatalities']>CUTOFF]['Date'].min()

print(CUTOFF, " deaths reached on ", minDate)



s1 = c[c['Date']>=minDate]

s1 = s1.iloc[0::sampleFreq, :].copy()  #weekly sample, starting on the cutoff

s1['NewFatalities'] = s1['Fatalities'].diff()  #replace original daily values after sampling

s1['NewConfirmedCases'] = s1['ConfirmedCases'].diff()





#calculate the number of days since the first day fatalities exceeded the cutoff

s1['Days'] = (s1['Date'] - minDate) / np.timedelta64(1, 'D')

x = s1['Days']



#display(s1[['Days','Fatalities','ConfirmedCases']])



#sample the actual data at the given frequency. create one sample for each possible offset day 

calib_data = []

calib_data_days = []

calib_data_fatalities = []

samples = []

max_sample =0 



for shift in range(sampleFreq):



    s1 = c[c['Date']>=minDate - np.timedelta64(1, 'D')]  #start 3 days earlier to adjust for the biais in having the samples start with one week difference

    s1 = s1.iloc[shift::sampleFreq, :].copy()  #weekly sample with a shift to start



    #calculate the number of days since the first day fatalities exceeded the cutoff

    s1['Days'] = (s1['Date'] - minDate) / np.timedelta64(1, 'D')



    sample = pd.DataFrame()

    sample['Days'] = s1['Days']

    sample['Fatalities'] = s1['Fatalities']

    days = []

    

    for index,row in sample.iterrows():

        days.append(row['Days'])

        calib_data_days.append(row['Days'])

        calib_data_fatalities.append(row['Fatalities'])

    samples.append(days)

    if len(days)>max_sample:

        max_sample=len(days)



#display(samples)

#display(calib_data_days)

#display(calib_data_fatalities)



popt, pcov = curve_fit(lambda x, incubating0, infectionRate : SIR2_multifatalities(x, incubating0, infectionRate, samples=samples),

                       calib_data_days,

                       calib_data_fatalities,

                       bounds=bounds, 

                       p0=initial_guess)



calib_incubating0 = popt[0]

calib_infectionRate = popt[1]



calib_incubating0 = 1e-6

calib_infectionRate = 2.5





x = np.arange(max_sample)*sampleFreq

           

y = calib_population * SIR2(x, 

                            incubating0=calib_incubating0, 

                            symptomaticInfectionRate = calib_infectionRate, 

                            moderateInfectionRate = calib_infectionRate, 

                            criticalRate=calib_criticalRate, 

                            deathRate=calib_deathRate)

s2 = pd.DataFrame()

s2['Days'] = x

s2['fit Fatalities (SIR)'] = y[:,cF]  

s2['fit NewFatalities (SIR)'] = s2['fit Fatalities (SIR)'].diff()



s2['fit Cases (SIR)'] = y[:,cC]  #reported stats are about new cases, they do not seem to account for people having recovered

s2['fit NewCases (SIR)'] = s2['fit Cases (SIR)'].diff()



print("SIR2 model fit")



print("Assumption Population = {:,.0f} millions".format(calib_population / 1e6))

print("Assumption Hospitalization Rate = {:.1%}".format(calib_criticalRate))

print("Assumption Death Rate after Hospitalization = {:.1%}".format(calib_deathRate))



print("Calibrated Incubating0 = {:.1f} per million susceptible".format(calib_incubating0*1e6))

print("Calibrated Infection Rate = {:.1f}".format(calib_infectionRate))



display(s2.sort_values(by='Days',ascending=False))



fig,axs = plt.subplots(nrows=3, ncols=2,figsize=[16,16])



for shift in range(sampleFreq):



    s1 = c[c['Date']>=minDate]

    s1 = s1.iloc[shift::sampleFreq, :].copy()  #weekly sample with a shift to start

    s1['NewFatalities'] = s1['Fatalities'].diff()  #replace original daily values after sampling

    s1['NewConfirmedCases'] = s1['ConfirmedCases'].diff()



    #calculate the number of days since the first day fatalities exceeded the cutoff

    s1['Days'] = (s1['Date'] - minDate) / np.timedelta64(1, 'D')

    x = s1['Days']  

    

    ax = plt.subplot(321)

    plt.title(STATE + ' Fatalities')

    plt.plot(x, s1['Fatalities'],'ko-',label='Actual')

    plt.plot(s2['Days'], s2['fit Fatalities (SIR)'],'b*-',label='SIR')

    plt.legend()

    plt.grid()

    plt.xlabel('days since report of {:.0f} fatalities'.format(CUTOFF))

    #plt.yscale('log')



    ax = plt.subplot(322)

    plt.title(STATE + ' New Fatalities')

    plt.plot(x, s1['NewFatalities'],'ko-',label='Actual')

    plt.plot(s2['Days'], s2['fit NewFatalities (SIR)'],'b*-',label='SIR')

    plt.legend()

    plt.grid()

    #plt.yscale('log')



    ax = plt.subplot(323)

    plt.title(STATE + ' Confirmed Cases')

    plt.plot(x, s1['ConfirmedCases'],'ko-',label='Actual')

    plt.plot(s2['Days'], s2['fit Cases (SIR)'],'b*-',label='SIR')

    plt.legend()

    plt.grid()

    plt.yscale('log')

    ax.yaxis.set_major_formatter(FuncFormatter(thousands))



    ax = plt.subplot(324)

    plt.title(STATE + ' New Cases')

    plt.plot(x, s1['NewCases'],'ko-',label='Actual')

    plt.plot(s2['Days'], s2['fit NewCases (SIR)'],'b*-',label='SIR')

    plt.legend()

    plt.grid()

    plt.yscale('log')

    ax.yaxis.set_major_formatter(FuncFormatter(thousands))



#forecast for 20 periods

xx = np.arange(20)*sampleFreq

y = calib_population * SIR2(xx, 

                            incubating0=calib_incubating0, 

                            symptomaticInfectionRate = calib_infectionRate, 

                            moderateInfectionRate = calib_infectionRate, 

                            criticalRate=calib_criticalRate, 

                            deathRate=calib_deathRate)



idx = np.argmax(y[:,cSy]).item()

print("New Confirmed Cases would peak period {:,} after the cutoff, on {:%Y-%m-%d}".format(idx,minDate+timedelta(days=sampleFreq*idx)))



ax = plt.subplot(325)

plt.title(STATE + ' Forecast')

plt.plot(xx, y[:,cSu],'g-',label='Susceptible')

plt.plot(xx, y[:,cI],'r-',label='Incubating')

plt.plot(xx, y[:,cC],'b-',label='Positives')

plt.plot(xx[1:], np.diff(y[:,cI]),'m-',label='New Infections')

ax.yaxis.set_major_formatter(FuncFormatter(millions))

plt.legend()

plt.grid()

plt.xlabel('days since report of {:.0f} fatalities'.format(CUTOFF))

#plt.yscale('log')



ax = plt.subplot(326)

plt.title(STATE + ' Forecast')

lns2 = plt.plot(xx, y[:,cF],'c-',label='Fatalities (lhs)')

ax.yaxis.set_major_formatter(FuncFormatter(thousands))



ax2 = ax.twinx() #instantiate second y axis, share same x axis

lns3 = plt.plot(xx[1:], np.diff(y[:,cF]),'m-',label='New Fatalities (rhs)')

ax2.yaxis.set_major_formatter(FuncFormatter(thousands))



lns = lns2+lns3

labs = [l.get_label() for l in lns]

ax.legend(lns, labs, loc=0)

ax.grid()

#plt.yscale('log')



plt.show()

def intervention(day, day0, lag=1, effect=0.0):

    if day>day0+lag:

        return 1.0 - effect

    if day>day0:

        return 1.0 - effect * (day-day0)/lag

    return 1.0



days = np.arange(300)

effects = np.zeros(300)

for d in days:

    effects[d] = intervention(d, 200, 3, 0.75)

plt.plot(days, effects)

plt.show()



#-------------------------------------------------------

#params:

# array of number of days since inception (not used except to size output)

# i0 : initial percentage of infected population

# beta : daily rate of transmission by infected people to susceptible people, prior R0=2.7=beta/gamma

#death_rate prior=0.01/21.0  ; death rate of infected people (1% die about 3 weeks after infection)

#gamma prior=1.0/21  ; it takes three weeks to stop being infectious (either fully recovered, or dead)

#-------------------------------------------------------



def SIR3(x, i0, beta, gamma, death_rate, intervention_day, intervention_lag, intervention_effect):

    

    y = np.empty((x.size,5))



    for i in range(0,x.size):

        

        if i==0:

            #initial conditions

            infected = i0

            susceptible = 1.0 - i0

            recovered = 0.0

            fatalities = 0.0       

            positives = i0

          

        else:

            #compute variations

            

            rate = beta * intervention(i, intervention_day, intervention_lag, intervention_effect)

            

            new_fatalities = death_rate * infected

            new_recovered = (gamma - death_rate) * infected

            new_positives = rate * susceptible * infected

            new_infected = new_positives - gamma * infected 

            new_susceptible = - new_positives

            

            #integrate and store in result array

            susceptible += new_susceptible

            positives += new_positives

            infected += new_infected

            recovered += new_recovered

            fatalities += new_fatalities

            

        y[i,0] = susceptible

        y[i,1] = infected

        y[i,2] = recovered

        y[i,3] = fatalities

        y[i,4] = positives  #cumul of infected, does not come down on recovery

            

    return y





x = np.arange(300)



y0 = SIR(x, i0=1e-6, beta=2.5/21, gamma=1.0/21, death_rate=0.01/21)



y = SIR3(x, i0=1e-6, beta=2.5/21, gamma=1.0/21, death_rate=0.01/21, 

         intervention_day = 100, intervention_lag = 3, intervention_effect = 0.1)



fig,axs = plt.subplots(nrows=1,ncols=2, figsize=[16,8])

plt.legend()

import matplotlib.ticker as mtick



lns1 = axs[0].plot(x, y[:,0],'g-',label='susceptible (lhs)')

lns2 = axs[0].plot(x, y[:,2],'b-',label='recovered (lhs)')

lns21 = axs[0].plot(x, y[:,4],'m-',label='positives (lhs)')

ax2 = axs[0].twinx() #instantiate second y axis, share same x axis

lns3 = ax2.plot(x, y[:,1],'r-',label='infected (rhs)')

lns31 = ax2.plot(x, y0[:,1],'k-',label='infected baseline (rhs)')

lns4 = ax2.plot(x, y[:,3],'c-',label='fatalities (rhs)')

axs[0].yaxis.set_major_formatter(mtick.PercentFormatter(1.0))

ax2.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))



lns = lns1+lns2+lns21+lns3+lns31+lns4

labs = [l.get_label() for l in lns]

axs[0].legend(lns, labs, loc=0)

axs[0].grid()





lns1 = axs[1].plot(np.diff(y[:,1]),'r-',label='new infected (lhs)')

ax2 = axs[1].twinx() #instantiate second y axis, share same x axis

lns2 = ax2.plot(np.diff(y[:,3]),'c-',label='new fatalities (rhs)')

lns21 = ax2.plot(np.diff(y0[:,3]),'k-',label='new fatalities baseline(rhs)')

axs[1].yaxis.set_major_formatter(mtick.PercentFormatter(1.0))

ax2.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))



lns = lns1+lns2+lns21

labs = [l.get_label() for l in lns]

axs[1].legend(lns, labs, loc=0)

axs[1].grid()





#display(y)



##############################

####### CALIBRATION TO fatalities

####### with INTERVENTION

##############################





Population = {

    'New York':20e6,

    'California':40e6,

    'France':67e6,

    'Italy':60e6,

    'North Dakota': 0.7e6,

    'Florida': 21e6,

    'Hubei':59e6 #wuhan=11, hubei=59 59e6

}



#STATE = 'Italy'

STATE = 'France'

#STATE = 'New York'

#STATE = 'California'

#STATE = 'Hubei'

#STATE = 'North Dakota'

#STATE = 'Florida'



POPULATION = Population[STATE]

CUTOFF = 0



#progressive implementation of the intervention to flatten the curve, in days

#calib_intervention_lag = 7



#calibrate on cumulative fatalities or daily fatalities?

#0: calibrate on Fatalities; 

#1: calibrate on daily fatalities

CALIB_DIFF = 1



#calibrate the model parameters to match actual fatality rates (assumption is that number of fatalities are best quality data)

def SIR3_fatalities(x, i0, beta, gamma, death_rate,intervention_day, intervention_lag, intervention_effect):

    y = SIR3(x, i0, beta=beta, gamma=gamma, death_rate=death_rate*gamma,

            intervention_day = intervention_day, intervention_lag = intervention_lag, intervention_effect=intervention_effect)

    

    if CALIB_DIFF==1:

        y = np.diff(y[:,3])

        y = np.insert(y,0,0)

        return y * POPULATION

    else:

        return y[:,3] * POPULATION



'''

'''



I0_min = 1e-6

I0_max = 1000e-6

Gamma_min = 1/24

Gamma_max = 1/14

Beta_min = 1.1 * Gamma_min

Beta_max = 6 * Gamma_max

DeathRate_min = 0.5e-2

DeathRate_max = 2e-2

InterventionDay_min = 1

InterventionDay_max = 25

InterventionLag_min = 1

InterventionLag_max = 100

InterventionEffect_min = 0

InterventionEffect_max = 0.95 #85% reduction of initial transmission rate



initial_guess = [(I0_max+I0_min)/2, 

                 2.5/21, 

                 1/21,

                 (DeathRate_max+DeathRate_min)/2, 

                 5,  #intervention day

                 1,   #lag

                 0.2]



bounds = ((I0_min, Beta_min, Gamma_min, DeathRate_min, InterventionLag_min, InterventionDay_min, InterventionEffect_min),

          (I0_max, Beta_max, Gamma_max, DeathRate_max, InterventionLag_max, InterventionDay_max, InterventionEffect_max))



#formatting functions for charts

def millions(x, pos):

    'The two args are the value and tick position'

    return '$%1.1fM' % (x * 1e-6)



#formatting functions for charts

def thousands(x, pos):

    'The two args are the value and tick position'

    return '$%1.0fT' % (x * 1e-3)



#filter the data and keep the given STATE only

c = train[train['State']==STATE]

c = c.groupby(['Date']).sum().reset_index()



#find the first date when the fatalities cutoff was reached by this STATE

minDate = c[c['Fatalities']>CUTOFF]['Date'].min()

print(CUTOFF, " deaths reached on ", minDate)



s1 = c[c['Date']>minDate].copy()



#calculate the number of days since the first day fatalities exceeded the cutoff

s1['Days'] = (s1['Date'] - minDate) / np.timedelta64(1, 'D')

x = s1['Days']



if CALIB_DIFF==1:

    #calibrate on daily fatality numbers

    z = s1['Fatalities'].diff()

    z.fillna(0,inplace=True)

else:

    z = s1['Fatalities']

    

popt, pcov = curve_fit(SIR3_fatalities, x, z, bounds=bounds,p0=initial_guess)



calib_I0 = popt[0]

calib_Beta = popt[1]

calib_Gamma = popt[2]

calib_DeathRate = popt[3]

calib_InterventionDay = popt[4]

calib_InterventionLag = popt[5]

calib_InterventionEffect = popt[6]



y = POPULATION * SIR3(x,i0=calib_I0, beta=calib_Beta, gamma=calib_Gamma, death_rate=calib_Gamma*calib_DeathRate,

                     intervention_day = calib_InterventionDay, intervention_lag=calib_InterventionLag, intervention_effect = calib_InterventionEffect)



s1['fit Fatalities (SIR)'] = y[:,3]  #reported stats are about new cases, they do not seem to account for people having recovered

s1['fit NewFatalities (SIR)'] = s1['fit Fatalities (SIR)'].diff()



s1['fit Cases (SIR)'] = y[:,4]  #reported stats are about new cases, they do not seem to account for people having recovered

s1['fit NewCases (SIR)'] = s1['fit Cases (SIR)'].diff()



print("SIR model fit")

print("I0 = {:,.0f} per million, or {:,.0f} persons initially infected".format(calib_I0 * 1e6, calib_I0*POPULATION))

print("BETA = {:.3f}".format(calib_Beta))

print("GAMMA = {:.3f}, or {:.1f} days to recover".format(calib_Gamma, 1/calib_Gamma))

print("DEATH RATE = {:.3%} infected people die".format(calib_DeathRate))

print("RHO = {:.2f}".format(calib_Beta/calib_Gamma))

print("Intervention Day = {:.0f} days after the cutoff on {:%Y-%m-%d}".format(calib_InterventionDay, minDate+timedelta(days=calib_InterventionDay)))

print("Intervention Lag = {:.0f} days for full intervention effect".format(calib_InterventionLag))

print("Intervention Effect = {:.0%} reduction of initial transmission rate".format(calib_InterventionEffect))

#display(popt)



#display(s1.sort_values(by='Date',ascending=False))



fig,axs = plt.subplots(nrows=3, ncols=2,figsize=[16,16])



plt.subplot(321)

plt.title(STATE + ' Fatalities')

plt.plot(x, s1['Fatalities'],'ko-',label='Actual')

plt.plot(x, s1['fit Fatalities (SIR)'],'b-',label='SIR')

plt.legend()

plt.grid()

#plt.yscale('log')



ax = plt.subplot(322)

plt.title(STATE + ' New Fatalities')

plt.plot(x, s1['NewFatalities'],'ko-',label='Actual')

plt.plot(x, s1['fit NewFatalities (SIR)'],'b-',label='SIR')

plt.legend()

plt.grid()

#plt.yscale('log')



ax = plt.subplot(323)

plt.title(STATE + ' Confirmed Cases')

plt.plot(x, s1['ConfirmedCases'],'ko-',label='Actual')

plt.plot(x, s1['fit Cases (SIR)'],'b-',label='SIR')

plt.legend()

plt.grid()

plt.yscale('log')

ax.yaxis.set_major_formatter(FuncFormatter(thousands))



ax = plt.subplot(324)

plt.title(STATE + ' New Cases')

plt.plot(x, s1['NewCases'],'ko-',label='Actual')

plt.plot(x, s1['fit NewCases (SIR)'],'b-',label='SIR')

plt.legend()

plt.grid()

plt.yscale('log')

ax.yaxis.set_major_formatter(FuncFormatter(thousands))



xx = np.arange(250)

y = POPULATION * SIR3(xx, i0=calib_I0, beta=calib_Beta, gamma=calib_Gamma, death_rate=calib_Gamma*calib_DeathRate,

                      intervention_day = calib_InterventionDay, intervention_lag=calib_InterventionLag, intervention_effect = calib_InterventionEffect)





idx = np.argmax(y[:,1]).item()

print("Confirmed Cases would peak day {:,}, on {:%Y-%m-%d}".format(idx,minDate+timedelta(days=idx)))



ax = plt.subplot(325)

plt.title(STATE + ' Forecast')

plt.plot(xx, y[:,0],'g-',label='Susceptible')

plt.plot(xx, y[:,1],'r-',label='Infected')

plt.plot(xx, y[:,2],'b-',label='Recovered')

plt.plot(xx[1:], np.diff(y[:,1]),'m-',label='Daily New Infections')

ax.yaxis.set_major_formatter(FuncFormatter(millions))

plt.legend()

plt.grid()

#plt.yscale('log')



ax = plt.subplot(326)

plt.title(STATE + ' Forecast')

lns2 = plt.plot(xx, y[:,3],'c-',label='Fatalities (lhs)')

ax.yaxis.set_major_formatter(FuncFormatter(thousands))



ax2 = ax.twinx() #instantiate second y axis, share same x axis

lns3 = plt.plot(xx[1:], np.diff(y[:,3]),'m-',label='Daily Fatalities (rhs)')

ax2.yaxis.set_major_formatter(FuncFormatter(thousands))



lns = lns2+lns3

labs = [l.get_label() for l in lns]

ax.legend(lns, labs, loc=0)

ax.grid()

#plt.yscale('log')



plt.show()