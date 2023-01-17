import numpy as np

import matplotlib.pyplot as plt

from scipy import integrate, optimize

import pandas as pd

df= pd.read_csv("../input/covidprocesseddata/confirmed-data-top-affected-countries.csv")

df["Country"].values
df.head()
df_pop=pd.read_csv("../input/covidprocesseddata/population_by_country_2020.csv")

df_pop.head()

def seir(y, x, R0, Tr, Tl):

    gamma=1./Tr

    beta=R0*gamma

    sigma=1./Tl

    sus = -beta*y[0]*y[2]/N

    expo = (beta*y[0]*y[2]/N)-sigma*y[1]

    infected = sigma*y[1] - gamma*y[2]

    reco = gamma*y[2]

    return sus, expo, infected, reco

## 5.5 tl, 2.5 tr

def fit_ode(x,  R0, Tr, Tl):

    y= integrate.odeint(seir, (sus0, exp0, inf0, rec0), x, args=( R0, Tr, Tl))

    return y[:,2] + y[:,3]

def fit_country(country,population,bounds=([3,1,1], [5,12,12])):

    ydata= df[df["Country"]==country].values[0,1:]

    ydata = np.array(ydata, dtype=float)

    #xdata = np.array(xdata, dtype=float)

    idx=np.where(ydata>50)[0][0]

    ydata=ydata[idx:]

    xdata= np.arange(len(ydata))  #+ idx

#    xdata=xdata[idx:]

    global N, inf0, sus0,rec0,exp0

    #print(xdata,ydata)

    N = population

    inf0 = ydata[0]

    sus0 = N - inf0 -inf0*5

    rec0 = 0.0

    exp0= inf0*5

    popt, pcov = optimize.curve_fit(fit_ode, xdata, ydata,bounds=bounds)

    perr = np.sqrt(np.diag(pcov))

    fitted = fit_ode(xdata, *popt)

    err=np.linalg.norm(fitted-ydata)



    la=country, " R0 =  %3.1f " %popt[0], " Tr = %3.1f  "%popt[1], " Tl = %1.1f" %popt[2], " err = %1.1f" %err



    plt.plot(xdata, ydata, 'o', mec = 'r',label=country )

    plt.plot(xdata, fitted,label=la)

    

    plt.legend()

    plt.yscale("log")

    plt.ylabel("Confirmed Cases with fit(I+R)")

    plt.xlabel("days")

    #print(country, " R0 =  %3.1f \t" %popt[0], ", Tr = %3.1f \t "%popt[1], ", Tl = %1.1f" %popt[2])



   # print(pcov)

   # plt.title(country)



plt.figure(figsize=(10,10))

country='Italy'

population=df_pop[df_pop["Country (or dependency)"]=='Italy']["Population (2020)"].values[0] #'South Korea' #United States

fit_country(country,population)#bounds=([1,0,0],[15,30,15]))

country='Germany'

population=df_pop[df_pop["Country (or dependency)"]=='Germany']["Population (2020)"].values[0] #'South Korea' #United States

fit_country(country,population)#bounds=([1,0,0],[15,30,15]))

#country='China'

#population=df_pop[df_pop["Country (or dependency)"]=='China']["Population (2020)"].values[0] #'South Korea' #United States

#fit_country(country,population)#bounds=([1,0,0],[15,30,15]))

#country='China'

#population=df_pop[df_pop["Country (or dependency)"]=='China']["Population (2020)"].values[0] #'South Korea' #United States

#fit_country(country,population,bounds=([0,0,0],[30,14,14]))

#country='Korea, South'

#population=df_pop[df_pop["Country (or dependency)"]=='South Korea']["Population (2020)"].values[0] #'South Korea' #United States

#fit_country(country,population)#bounds=([1,0,0],[15,30,15]))

#plt.savefig("fit_1.png")

#plt.figure(figsize=(10,10))

#population=df_pop[df_pop["Country (or dependency)"]=='South Korea']["Population (2020)"].values[0] #'South Korea' #United States

#fit_country(country,population,bounds=([0,0,0],[30,14,14]))

#country='US'

#population=df_pop[df_pop["Country (or dependency)"]=='United States']["Population (2020)"].values[0] #'South Korea' #United States

#fit_country(country,population,bounds=([1,0,0],[15,30,15]))

#country='Iran'

#population=df_pop[df_pop["Country (or dependency)"]=='Iran']["Population (2020)"].values[0] #'South Korea' #United Statesfit_country(country,population,bounds=([0,0,0],[30,14,14]))

#fit_country(country,population,bounds=([0,1,1],[30,14,14]))

#plt.savefig("fit_2.png")



#plt.figure(figsize=(10,10))



country='Spain'

population=df_pop[df_pop["Country (or dependency)"]=='Spain']["Population (2020)"].values[0] #'South Korea' #United States

fit_country(country,population)#bounds=([1,0,0],[15,30,15]))

country='United Kingdom'

population=df_pop[df_pop["Country (or dependency)"]=='United Kingdom']["Population (2020)"].values[0] #'South Korea' #United States

fit_country(country,population)#bounds=([1,0,0],[15,30,15]))

country='France'

population=df_pop[df_pop["Country (or dependency)"]=='France']["Population (2020)"].values[0] #'South Korea' #United States

fit_country(country,population)#bounds=([1,0,0],[15,30,15]))

#plt.savefig("fit_3.png")



#plt.savefig("fit_EU_1.png")
def seir_norm(y,x, R0, Tr, Tl):

    gamma=1./Tr

    beta=R0*gamma

    sigma=1./Tl

    sus = -beta*y[0]*y[2]

    expo = (beta*y[0]*y[2])-sigma*y[1]

    infected = sigma*y[1] - gamma*y[2]

    reco = gamma*y[2]

    return sus, expo, infected, reco

def fit_ode_norm(x,  R0, Tr, Tl):

    return integrate.odeint(seir_norm, (sus0, exp0, inf0, rec0), x, args=( R0, Tr, Tl),atol=1e-16)[:,2]

#integrate.solve_ivp(seir_norm, [sus0, exp0, inf0, rec0], args=( R0, Tr, Tl),method='RK45').y[:,2]

def fit_country_norm(country,population,bounds=([3,1,1], [5,12,12])):

    ydata= df[df["Country"]==country].values[0,1:]

    ydata = np.array(ydata, dtype=float)

    #xdata = np.array(xdata, dtype=float)

    idx=np.where(ydata>100)[0][0]

    ydata=ydata[idx:]

    xdata= np.arange(len(ydata))  #+ idx

#    xdata=xdata[idx:]

    global N, inf0, sus0,rec0,exp0



    #print(xdata,ydata)

    N = population

    inf0 = ydata[0]/N

    exp0= inf0*10

    sus0 = 1 - rec0 - exp0

    rec0 = inf0*.1

    

    popt, pcov = optimize.curve_fit(fit_ode_norm, xdata, ydata/N,bounds=bounds)

    fitted = fit_ode_norm(xdata, *popt)

    err=np.linalg.norm(fitted-ydata)

    la=country, " R0 =  %3.1f " %popt[0], " Tr = %3.1f  "%popt[1], " Tl = %1.1f" %popt[2], " err = %1.1f" %err

    plt.plot(xdata, ydata/N, 'o', mec = 'r',label=country)

    plt.plot(xdata, fitted,label=la)

    plt.legend()

    plt.yscale("log")

  #  print(country, " R0 =  %3.3f \t" %popt[0], ", Tr = %3.3f \t "%popt[1], ", Tl = %3.3E" %popt[2])

   # print(pcov)

   # plt.title(country)



plt.figure(figsize=(10,10))

country='Italy'

population=df_pop[df_pop["Country (or dependency)"]=='Italy']["Population (2020)"].values[0] #'South Korea' #United States

fit_country_norm(country,population)#bounds=([1,0,0],[15,30,15]))

country='Germany'

population=df_pop[df_pop["Country (or dependency)"]=='Germany']["Population (2020)"].values[0] #'South Korea' #United States

fit_country_norm(country,population)#bounds=([1,0,0],[15,30,15]))

#country='China'

#population=df_pop[df_pop["Country (or dependency)"]=='China']["Population (2020)"].values[0] #'South Korea' #United States

#fit_country_norm(country,population)#bounds=([1,0,0],[15,30,15]))

#country='China'

#population=df_pop[df_pop["Country (or dependency)"]=='China']["Population (2020)"].values[0] #'South Korea' #United States

#fit_country_norm(country,population,bounds=([1,1,1],[30,14,14]))

#country='Korea, South'

#population=df_pop[df_pop["Country (or dependency)"]=='South Korea']["Population (2020)"].values[0] #'South Korea' #United States

#fit_country_norm(country,population)#bounds=([1,0,0],[15,30,15]))

#plt.savefig("fit_1.png")

#plt.figure(figsize=(10,10))

#population=df_pop[df_pop["Country (or dependency)"]=='South Korea']["Population (2020)"].values[0] #'South Korea' #United States

#fit_country_norm(country,population,bounds=([0,0,0],[30,14,14]))

#country='US'

#population=df_pop[df_pop["Country (or dependency)"]=='United States']["Population (2020)"].values[0] #'South Korea' #United States

#fit_country_norm(country,population,bounds=([1,0,0],[15,30,15]))

#country='Iran'

#population=df_pop[df_pop["Country (or dependency)"]=='Iran']["Population (2020)"].values[0] #'South Korea' #United Statesfit_country(country,population,bounds=([0,0,0],[30,14,14]))

#fit_country_norm(country,population,bounds=([0,1,1],[30,14,14]))

#plt.savefig("fit_2.png")



#plt.figure(figsize=(10,10))



country='Spain'

population=df_pop[df_pop["Country (or dependency)"]=='Spain']["Population (2020)"].values[0] #'South Korea' #United States

fit_country_norm(country,population)#bounds=([1,0,0],[15,30,15]))

country='United Kingdom'

population=df_pop[df_pop["Country (or dependency)"]=='United Kingdom']["Population (2020)"].values[0] #'South Korea' #United States

fit_country_norm(country,population)#bounds=([1,0,0],[15,30,15]))

country='France'

population=df_pop[df_pop["Country (or dependency)"]=='France']["Population (2020)"].values[0] #'South Korea' #United States

fit_country_norm(country,population)#bounds=([1,0,0],[15,30,15]))

#plt.savefig("fit_3.png")



plt.savefig("fit_EU_normalised.png")