import numpy as np, pandas as pd, scipy as sp

import matplotlib.pyplot as plt

from sklearn.metrics import mean_squared_error

from datetime import datetime, timedelta

from time import strptime
k, L = 2.5, 25

hill = lambda t : 1 / (1 + (t/L)**k)

times = np.arange(100)



plt.plot(hill(times))

plt.ylabel('Decay')

plt.xlabel('Days')

plt.title('Hill function with k=2.5, L=25')

plt.show()
train = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-2/train.csv')

populations = pd.read_csv('/kaggle/input/covid19-population-data/population_data.csv')

populations = populations.drop(columns=['Type']).set_index('Name').transpose()

populations = populations.to_dict()

train.columns = ['Id', 'State', 'Country', 'Date', 'ConfirmedCases', 'Fatalities']
# visualization function for later



def multi_plot(M, susceptible = True, labels=False, interventions=False):

    n = M.shape[0]

    CC0 = M[2,0]+M[3,0]+M[4,0]

    CCases = np.diff(M[2]+M[3]+M[4], prepend=CC0).cumsum()

    Deaths = M[4]

    fig = plt.figure()

    ax1 = fig.add_axes([0.1, 1, 1.25, 1], ylabel='# of People')

    ax2 = fig.add_axes([0.1, 0, 1.25, 1], ylabel='# of People')    

    if susceptible == True:

        rows=range(0,n)

    else:

        rows=range(1,n)

    for ii in rows:

        if labels == False:

            ax1.plot(M[ii])

        else:

            ax1.plot(M[ii], label = labels[ii])

    if interventions==False:

        ax1.set_title('Time Evolution without intervention')

    else:

        ax1.set_title('Time Evolution with intervention')

        for action, day in zip(list(interventions.keys()), [interventions[kk]['day'] for kk in list(interventions.keys())]):

            ax1.axvline(x=day,label=action, linestyle='--')

            ax2.axvline(x=day,label=action, linestyle='--')

    ax1.legend(loc='best')

    ax2.plot(CCases, label='ConfirmedCases', color='brown')

    ax2.plot(Deaths, label='Deaths', color='black')

    ax2.legend(loc='best')

    ax2.set_xlabel('Days')

    plt.show()



categories = ['Susceptible','Exposed','Infected','Recovered','Deceased']
# SEIRD model for simulation



def reproduction(t):

    intervention_days = [interventions[kk]['day'] for kk in list(interventions.keys())]

    reproduction_rates = [interventions[kk]['reproduction_rate'] for kk in list(interventions.keys())]

    ix=np.where(np.array(intervention_days)<t)[0]

    

    if len(ix)==0:

        return R0

    else:

        return reproduction_rates[ix.max()]





def dS_dt(S, I, reproduction_t, alpha1, alpha2):

    return -alpha1*reproduction_t*S*I -alpha2*reproduction_t*S*I



def dE_dt(S, I, E, reproduction_t, alpha1, alpha2, beta):

    return alpha1*reproduction_t*S*I + alpha2*reproduction_t*S*I - beta*E



def dI_dt(E, I, beta, gamma, psi):

    return beta*E - gamma*I - psi*I



def dR_dt(I, gamma):

    return gamma*I



def dD_dt(I, psi):

    return psi*I





def ODE_model(t, y, Rt, alpha1, alpha2, beta, gamma, psi):



    if callable(Rt):

        reproduction_t = Rt(t)

    else:

        reproduction_t = Rt

    

    S, E, I, R, D = y

    St = dS_dt(S, I, reproduction_t, alpha1, alpha2)

    Et = dE_dt(S, I, E, reproduction_t, alpha1, alpha2, beta)

    It = dI_dt(E, I, beta, gamma, psi)

    Rt = dR_dt(I, gamma)

    Dt = dD_dt(I, psi)

    return [St, Et, It, Rt, Dt]
# Fix some population parameters for simulation (refer to previous sections for interpretations)



N = 1000 # population size

T_inc = 10 # days for average incubation

T_rec = 14 # days for average recovery

T_die = 10 # days for average infection duration given death

R0 = 5 # average number of contacts an infected person has per day

tau = 0.2 # probability of transmission given S <-> I contact

p_live = 0.95 # average survival rate

p_die = 0.05 # average pmortality rate

ndays = 100 # number of days simulated



# ODE parameters

alpha1 = tau*p_live/N # average % of susceptible people who get infected by survivor

alpha2 = tau*p_die/N # average % of susceptible people who get infected by non-survivor

beta = 1/T_inc # transition rate of incubation to infection

gamma = p_live/T_rec # transition rate of infection to recovery

psi = p_die/T_die # transition rate of infection to mortality



y0 = [N-1, 1, 0, 0, 0] # initial conditions

t_span = [0, ndays] # dayspan to evaluate

t_eval = np.arange(ndays) # days to evaluate

# Here we look at the evolution given no intervention



interventions = {}



solution = sp.integrate.solve_ivp(fun = ODE_model, t_span = t_span, t_eval = t_eval, y0 = y0, 

                                  args = (R0, alpha1, alpha2, beta, gamma, psi))



Y = np.maximum(solution.y,0)



multi_plot(Y, labels=categories)
# In this simulation, we look at the evolution given 2 interventions

# we may consider what happens to the process after some intervention, which

# reduces the average number of contacts an infected person has per day

# such as social distancing and lockdown



interventions = {'social_distancing':{'day':25, 'reproduction_rate':2},

                 'lockdown':{'day':35, 'reproduction_rate':.5}}



solution = sp.integrate.solve_ivp(fun = ODE_model, t_span = t_span, t_eval = t_eval, y0 = y0, 

                                  args = (reproduction, alpha1, alpha2, beta, gamma, psi))



Y = np.maximum(solution.y,0)



multi_plot(Y, labels=categories, interventions=interventions)

ndays = 500

t_span = [0, ndays] # dayspan to evaluate

t_eval = np.arange(ndays) # days to evaluate



solution = sp.integrate.solve_ivp(fun = ODE_model, t_span = t_span, t_eval = t_eval, y0 = y0, 

                                  args = (reproduction, alpha1, alpha2, beta, gamma, psi))



Y = np.maximum(solution.y,0)



multi_plot(Y, labels=categories, interventions=interventions)

def dS_dt(S, I, alpha1_t, alpha2_t):

    return -alpha1_t*S*I -alpha2_t*S*I



def dE_dt(S, I, E, alpha1_t, alpha2_t, beta):

    return alpha1_t*S*I + alpha2_t*S*I - beta*E



def dI_dt(E, I, beta, gamma, psi):

    return beta*E - gamma*I - psi*I



def dR_dt(I, gamma):

    return gamma*I



def dD_dt(I, psi):

    return psi*I





def ODE_model(t, y, alpha1t, alpha2t, beta, gamma, psi):



    alpha1_t = alpha1t(t)

    alpha2_t = alpha2t(t)

    

    S, E, I, R, D = y

    St = dS_dt(S, I, alpha1_t, alpha2_t)

    Et = dE_dt(S, I, E, alpha1_t, alpha2_t, beta)

    It = dI_dt(E, I, beta, gamma, psi)

    Rt = dR_dt(I, gamma)

    Dt = dD_dt(I, psi)

    return [St, Et, It, Rt, Dt]
def loss(theta, data, population, k, L, nforecast=0, error=True):

    alpha1_0, alpha2_0, beta, gamma, psi = theta

    

    Infected_0 = data.ConfirmedCases.iloc[0]

    ndays = nforecast

    ntrain = data.shape[0]

    y0 = [(population-Infected_0)/population, 0, Infected_0/population, 0, 0]

    t_span = [0, ndays] # dayspan to evaluate

    t_eval = np.arange(ndays) # days to evaluate

    

    def a1_t(t):

        return alpha1_0 / (1 + (t/L)**k)



    def a2_t(t):

        return alpha2_0 / (1 + (t/L)**k)



    sol = sp.integrate.solve_ivp(fun = ODE_model, t_span = t_span, t_eval = t_eval, y0 = y0, 

                                 args = (a1_t, a2_t, beta, gamma, psi))

    

    pred_all = np.maximum(sol.y, 0)

    ccases_pred = np.diff((pred_all[2] + pred_all[3] + pred_all[4])*population, n = 1, prepend = Infected_0).cumsum()

    deaths_pred = pred_all[4]*population

    ccases_act = data.ConfirmedCases.values

    deaths_act = data.Fatalities.values

    

    if ccases_act[-1]<ccases_act[-2]:

        ccases_act[-1]=ccases_act[-2]

    if deaths_act[-1]<deaths_act[-2]:

        deaths_act[-1]=deaths_act[-2]

    

    weights =  np.exp(np.arange(data.shape[0])/10)/np.exp((data.shape[0]-1)/10) 



    ccases_rmse = np.sqrt(mean_squared_error(ccases_act, ccases_pred[0:ntrain], sample_weight=weights))

    deaths_rmse = np.sqrt(mean_squared_error(deaths_act, deaths_pred[0:ntrain], sample_weight=weights))



    loss = np.mean((ccases_rmse, deaths_rmse))

    

    if error == True:

        return loss

    else:

        return loss, ccases_pred, deaths_pred
train['location'] = train['State'].fillna(train['Country'])

locations=list(train['location'].drop_duplicates())

train.set_index(['location', 'Date'], inplace=True)

train.head()
parms0 = [1.5, 1.5, 0.5, 0.05, 0.001]

bnds = ((0.001, None), (0.001, None), (0, 10), (0, 10), (0, 10))
def fit_ODE_model(location, k, L):

        

    dat = train.loc[location].query('ConfirmedCases > 0')

    nforecast = 75

    population = populations[location]['Population']

    n_infected = train['ConfirmedCases'].iloc[0]

        

    res = sp.optimize.minimize(fun = loss, x0 = parms0, 

                               args = (dat, population, k, L, nforecast),

                               method='L-BFGS-B', bounds=bnds)

    

    dates_all = [str(datetime.strptime(dat.index[0], '%Y-%m-%d') + timedelta(days = ii))[0:10] for ii in range(nforecast)]

    

    err, ccases_pred, deaths_pred = loss(theta = res.x, data = dat, population = population, k=k, L=L, 

                                         nforecast=nforecast, error=False)

    

    predictions = pd.DataFrame({'ConfirmedCases': ccases_pred,

                                'Fatalities': deaths_pred}, index=dates_all)

    

    train_true = dat[['ConfirmedCases',  'Fatalities']]

    predictions.columns = ['ConfirmedCases_pred',  'Fatalities_pred']



    plot_df = pd.merge(predictions,train_true,how='left', left_index=True, right_index=True)



    plt.plot(plot_df.ConfirmedCases_pred.values, color='green',linestyle='--', linewidth=0.5, label='Confirmed Cases (pred)')

    plt.plot(plot_df.Fatalities_pred.values, color='blue',linestyle='--', linewidth=0.5, label='Fatalities (pred)')

    plt.plot(plot_df.Fatalities.values, color='red', label='Fatalities (real)')

    plt.plot(plot_df.ConfirmedCases.values, color='orange', label='Confirmed Cases (real)')

    plt.title(location)

    plt.xlabel('Days since first case in '+location)

    plt.ylabel('Confirmed Cases')

    plt.legend(loc='best')

    plt.show()        

    

    print(res.x)

fit_ODE_model('Korea, South', k, L)
fit_ODE_model('Hubei', k, L)
fit_ODE_model('Netherlands', k, L)
fit_ODE_model('Spain', k, L)
fit_ODE_model('Poland', k, L)