# provide a country and its population here

country = 'India'

country_population = 1333000000
import numpy as np 

import pandas as pd 

import matplotlib.pyplot as plt

from scipy import integrate, optimize



submission_example = pd.read_csv("../input/covid19-global-forecasting-week-4/submission.csv")

test = pd.read_csv("../input/covid19-global-forecasting-week-4/test.csv")

train = pd.read_csv("../input/covid19-global-forecasting-week-4/train.csv")

india=train[train["Country_Region"]=="India"]

india.head(50)
# train=train[train["Country_Region"]=="India"]

# y_last=train.iloc[-1,:]

# def sir(y,t,N,beta,gamma):

#     S,I,R=y

#     dsdt=-beta*S*I/N

#     didt=beta*S*I/N-gamma*I

#     drdt=gamma*I

#     return dsdt,didt,drdt



# N=country_population

# beta=0.15337209

# gamma=0.01919359

# I0=18539

# R0=3500

# S0=country_population-I0-R0

# t=np.linspace(0,100,100)

# y0=S0,I0,R0

# ret=integrate.odeint(sir,y0,t,args=(N,beta,gamma))

# S,I,R=ret.T



# def plotsir(t, S, I, R):

#   f, ax = plt.subplots(1,1,figsize=(10,4))

#   ax.plot(t, S, 'b', alpha=0.7, linewidth=2, label='Susceptible')

#   ax.plot(t, I, 'y', alpha=0.7, linewidth=2, label='Infected')

#   ax.plot(t, R, 'g', alpha=0.7, linewidth=2, label='Recovered')



#   ax.set_xlabel('Time (days)')



#   ax.yaxis.set_tick_params(length=0)

#   ax.xaxis.set_tick_params(length=0)

#   ax.grid(b=True, which='major', c='w', lw=2, ls='-')

#   legend = ax.legend()

#   legend.get_frame().set_alpha(0.5)

# plotsir(t,S,I,R)
# def sir_model(y, x, beta, gamma):

#     sus = -beta * y[0] * y[1] / N

#     rec = gamma * y[1]

#     inf = -(sus + rec)

#     return sus, inf, rec



# def fit_odeint(x, beta, gamma):

#     return integrate.odeint(sir_model, (sus0, inf0, rec0), x, args=(beta, gamma))[:,1]
# confirmed_total_date_country = train[train['Country_Region']==country].groupby(['Date']).agg({'ConfirmedCases':['sum']})

# fatalities_total_date_country = train[train['Country_Region']==country].groupby(['Date']).agg({'Fatalities':['sum']})

# total_date_country = confirmed_total_date_country.join(fatalities_total_date_country)

# country_df = total_date_country[(80+1):]
# country_df
# confirmed_total_date_country = train[train['Country_Region']==country].groupby(['Date']).agg({'ConfirmedCases':['sum']})

# fatalities_total_date_country = train[train['Country_Region']==country].groupby(['Date']).agg({'Fatalities':['sum']})

# total_date_country = confirmed_total_date_country.join(fatalities_total_date_country)

# country_df = total_date_country[(8+1):]

# country_df['day_count'] = list(range(1,len(country_df)+1))

# ydata = [i for i in country_df.ConfirmedCases['sum'].values]

# xdata = country_df.day_count

# ydata = np.array(ydata, dtype=float)

# xdata = np.array(xdata, dtype=float)

# N = country_population

# inf0 = ydata[0]

# sus0 = N - inf0

# rec0 = 0.0



# def sir_model(y, x, beta, gamma):

#         #y[0] is susceptible

#         #y[1] is infected

#         #y1 is recovered

#         sus = -beta * y[0] * y[1] / N

#         rec = gamma * y[1]

# #         inf = -(sus + rec)

#         inf=beta*y[0]*y[1]/N-(gamma*y[1])

#         return sus, inf, rec



# def fit_odeint(x, beta, gamma):

#         return integrate.odeint(sir_model, (sus0, inf0, rec0), x, args=(beta, gamma))

def fit_sir_country(country, country_pop, initial_date, additional_sim_days):

    population = float(country_pop)

    confirmed_total_date_country = train[train['Country_Region']==country].groupby(['Date']).agg({'ConfirmedCases':['sum']})

    fatalities_total_date_country = train[train['Country_Region']==country].groupby(['Date']).agg({'Fatalities':['sum']})

    total_date_country = confirmed_total_date_country.join(fatalities_total_date_country)

    country_df = total_date_country[(initial_date+1):]

    country_df['day_count'] = list(range(1,len(country_df)+1))



    ydata = [i for i in country_df.ConfirmedCases['sum'].values]

    xdata = country_df.day_count

    ydata = np.array(ydata, dtype=float)

    xdata = np.array(xdata, dtype=float)



    N = population

    inf0 = ydata[0]

    sus0 = N - inf0

    rec0 = 50.0

    

    def sir_model(y, x, beta, gamma):

        #y[0] is susceptible

        #y[1] is infected

        #y1 is recovered

        sus = -beta * y[0] * y[1] / N

        rec = gamma * y[1]

#         inf = -(sus + rec)

        inf=beta*y[0]*y[1]/N-(gamma*y[1])

        return sus, inf, rec



    def fit_odeint(x, beta, gamma):

        return integrate.odeint(sir_model, (sus0, inf0, rec0), x, args=(beta, gamma))[:,1]



    sim_length = len(xdata) + additional_sim_days # Length of simulation

    xdata2 = np.arange(1,sim_length)

    popt, pcov = optimize.curve_fit(fit_odeint, xdata, ydata)

    fitted = fit_odeint(xdata2, *popt)

#     print("betaaaaaaa",popt)

    print("Initial Start day : ", initial_date, " Optimal parameters: beta =", popt[0], " and gamma = ", popt[1])

    remaining_to_peak = np.argmax(fitted) - len(xdata)

    print("   Remaining days to reach global peak infected cases : ", remaining_to_peak)

    return remaining_to_peak, ydata, fitted, xdata, xdata2
# Fit SIR to the corresponding country and for the initial simulation start date

remaining_to_peak, ydata, fitted, xdata, xdata2 = fit_sir_country(country, country_population,52, 120)

plt.plot(xdata, ydata, 'o', label='Real data')

plt.plot(xdata2, fitted, label='SIR prediction')

plt.title("Fit of SIR model to the country infected cases")

plt.ylabel("Population infected")

plt.xlabel("Days")

plt.legend(loc='best')

plt.show()
max(fitted)
len(xdata)
fitted[len(xdata)-1]
remaining_days = []

# Loop on different initial start date for the simulation

for i in range(52,57):

    remaining_to_peak, _, _, _, _ = fit_sir_country(country, country_population, i, 60)

    remaining_days.append(remaining_to_peak)

print(remaining_days)
from scipy import stats



def rmNegative(L):

    index = len(L) - 1

    while index >= 0:

        if L[index] < 0:

            del L[index]

        index = index - 1



rmNegative(remaining_days)

stats.describe(remaining_days)

print("On average, the peak of infected cases in ", country, " is coming in : ", np.mean(remaining_days) ,"days")