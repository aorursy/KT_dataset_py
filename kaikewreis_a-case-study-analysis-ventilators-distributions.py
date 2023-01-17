# Standard modules

import numpy as np

import warnings



# Data modules

import pandas as pd



# Modelling modules

from scipy import integrate

from scipy import optimize



# Plot modules

import seaborn as sns

import matplotlib.pyplot as plt

import plotly.express as px

import plotly.offline as pyo

import plotly.graph_objs as go



# Notebook commands

warnings.filterwarnings('ignore')

pyo.init_notebook_mode()

%matplotlib inline
# FUNCTION - Define SEIR Model ODE System

def seir_model_ode(y, t, params): 

    '''

    Arguments:

    - y: dependent variables

    - t: independent variable (time)

    - params: Model params

    '''

    # Parameters to find

    infection_rate = params[0]

    recovery_rate = params[1]

    exposed_rate = params[2]

    

    # Y variables

    s = y[0]

    e = y[1]

    i = y[2]

    r = y[3]

    

    # SEIR EDO System 

    dsdt = -exposed_rate*s*i

    dedt = (exposed_rate*s*i) - (infection_rate*e)

    didt = (infection_rate*e) - (recovery_rate*i)

    drdt = recovery_rate*i

    

    # Return our system

    return (dsdt, dedt, didt, drdt)
# FUNCTION - Calculate SEIR Model in t (time as days) based on given parameters

def calculate_seir_model(params, t, initial_condition):

    # Create an alias to our ode model to pass guessed params

    seir_ode = lambda y,t:seir_model_ode(y, t, params)

    

    # Calculate ode solution, return values to each

    ode_result = integrate.odeint(func=seir_ode, y0=initial_condition, t=t)

    

    # Return results

    return ode_result
# FUNCTION - Auxiliar function to find the best parameters

def fit_seir_model(params_to_fit, t, initial_condition, i_r_true):

    # Calculate ODE solution for possible parameter, return values to each dependent variable:

    # (s, e, i and r)

    fit_result = calculate_seir_model(params_to_fit, t, initial_condition)

    

    # Calculate residual value between predicted VS true

    ## Note: ode_result[0] is S result

    residual_i = i_r_true[0] - fit_result[:,2]

    residual_r = i_r_true[1] - fit_result[:,3]



    # Create a np.array of all residual values for both (i) and (r)

    residual = np.concatenate((residual_i, residual_r))

    

    # Return results

    return residual
# Importing dataset

df_ita_status = pd.read_csv('/kaggle/input/uncover/UNCOVER/github/covid-19-italy-situation-monitoring-by-region.csv')



# Removing columns

df_ita_status.drop(axis=1, inplace=True, columns=['stato', 'codice_regione', 'lat', 'long', 'totale_positivi', 'isolamento_domiciliare','variazione_totale_positivi', 'nuovi_positivi', 'tamponi', 'note_it', 'note_en'])



# Change column names (turn my life easier)

df_ita_status.columns = ['date', 'state', 'hospitalized_with_symptom', 'icu', 'hospitalized_total', 'recovered', 'deaths', 'infected']



# Create a list of Italy states

ita_states = list(df_ita_status['state'].unique())



# Visualize dataset

df_ita_status
# Define figure and plot!

fig = px.line(df_ita_status, x='date', y='hospitalized_total', color='state')

pyo.iplot(fig)
# Define figure and plot!

fig2 = px.line(df_ita_status, x='date', y='deaths', color='state')

pyo.iplot(fig2)
# Define figure and plot!

fig3 = px.line(df_ita_status, x='date', y='infected', color='state')

pyo.iplot(fig3)
# Get Lombardia region

rgn_status = df_ita_status[df_ita_status['state'] == 'Lombardia'][['date', 'recovered', 'infected']].reset_index().drop(axis=1, columns=['index'])
N = 10060574
# Define Initial Condition (necessary for ODE solve)

R_start = rgn_status.loc[0, 'recovered']/N

I_start = rgn_status.loc[0, 'infected']/N

E_start = (rgn_status.loc[4, 'infected'] - rgn_status.loc[3, 'infected'])/N

S_start = 1 - E_start - I_start

# Set this values as a tuple

ic = (S_start, E_start, I_start, R_start)
print('Start condition:')

print('s(0): ', ic[0])

print('e(0): ', ic[1])

print('i(0): ', ic[2])

print('r(0): ', ic[3])
# Define a time array measure in days, but with values

time_opt = range(0, len(rgn_status))



# Create a tuple with the true values in fraction for Infected/Recovered cases (necessary for error measurement)

i_r_true = (list(rgn_status['infected']/N), list(rgn_status['recovered']/N))



# Define a start guess for our parameters [infection_rate, recovered rate, exposed rate]

params_start_guess = [0.1, 0.01, 0.1]
optimal_params, sucess = optimize.leastsq(fit_seir_model,

                                          x0=params_start_guess,

                                          args=(time_opt, ic, i_r_true),

                                          ftol=1.49012e-20)
print('## Lombardia')

print('Optimize infection rate: ', optimal_params[0])

print('Optimize recovered rate: ', optimal_params[1])

print('Optimize exposed rate: ', optimal_params[2])
# Get the optimal parameters

ir = optimal_params[0]

rr = optimal_params[1]

er = optimal_params[2]
# Calculate a curve based on those parameters

fit_result = calculate_seir_model((ir, rr, er), time_opt, ic)
# Plot the results for Infected/Recovered

## Define plot object

fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(20,10))

## Plot process

axes[0].plot(time_opt, i_r_true[0], 'g')

axes[0].plot(time_opt, fit_result[:,2], 'y')

axes[0].legend(['Ground truth', 'Predicted'],loc=2, fontsize=15)

axes[0].set_title('Infected cases - Lombardia',fontsize=20)

axes[1].plot(time_opt, i_r_true[1], 'g')

axes[1].plot(time_opt, fit_result[:,3], 'y')

axes[1].legend(['Ground truth', 'Predicted'],loc=2, fontsize=15)

axes[1].set_title('Recovered cases - Lombardia',fontsize=20);
# Get prediction full period time in datetime object and the convert to string

datetime_pred = pd.date_range(start="2020-02-24",end="2020-07-31", freq='D')

time_pred = [x.strftime("%Y-%m-%d") for x in datetime_pred]



# Get a list from 01/April to 31/July 

time_pred_range = range(0, len(time_pred))
# Calculate a SEIR prediction 

future_pred = calculate_seir_model((ir, rr, er), time_pred_range, ic)
# Plot results

## Define Date axis to better visualization (only first/half/last day of every month)

time_axis = [time_pred[i] for i in [0,6,20,37,51,67,81,98,112,128,142,158]]

## Define plot object

fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(20,10))

## Plot SEIR

sns.lineplot(x=time_pred, y=future_pred[:,0], ax=axes, color = 'blue')

sns.lineplot(x=time_pred, y=future_pred[:,1], ax=axes, color = 'red')

sns.lineplot(x=time_pred, y=future_pred[:,2], ax=axes, color = 'purple')

sns.lineplot(x=time_pred, y=future_pred[:,3], ax=axes, color = 'green')

plt.setp(axes.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

axes.legend(loc=1, labels=['Suscetible', 'Exposed', 'Infected','Recovered'], fontsize=10)

axes.set_title('LOMBARDIA - SEIR predictions', fontsize=20)

axes.set_xlabel('Date', fontsize=15)

axes.set_ylabel('Total cases', fontsize=15)

axes.set_xticks(time_axis);
# Calculate ventilators curve based in SEIR infected curve

future_pred_vent = 0.13*future_pred[:,2]
# Plot results

## Define plot object

fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(20,10))

## Plot SIR

sns.lineplot(x=time_pred, y=future_pred_vent, ax=axes, color = 'red')

sns.lineplot(x=time_pred, y=future_pred[:,2], ax=axes, color = 'purple')

plt.setp(axes.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

axes.legend(loc=1, labels=['Ventilators necessity curve','Infected curve'], fontsize=10)

axes.set_title('Lombardia', fontsize=20)

axes.set_xlabel('Date', fontsize=15)

axes.set_ylabel('Total count', fontsize=15)

axes.set_xticks(time_axis);
# Get the maximum curve value and transform to absolute value multiplying by region population

max_vent_necessity = N*max(future_pred_vent)

# Show results

print('Lombardia would need: ', int(max_vent_necessity),'ventilators.')
# Italy risk regions

ita_risk_states = ['Veneto', 'Piemonte', 'Emilia-Romagna']

# Population for each risk region

pop_states = [4906000.0, 4356000.0, 4459000.0] 
# Magical function

def extrapolating_italy(population_in_states=pop_states,states_in_italy=ita_risk_states):

    '''

    Function to return a list of tuples: (state_name, infection_rate, recovery_rate, exposed_rate, state_ventilators_curve)

    '''

    # Create a return tuple

    result = list()

    # Loop for each state

    for state, population in zip(states_in_italy, population_in_states):

        # Get a region

        rgn_status = df_ita_status[df_ita_status['state'] == state][['date', 'recovered', 'infected']].reset_index(

        ).drop(axis=1, columns=['index'])

        # Set N population value

        N = population

        # Find a pandemic start where at least recovered or infected have one value!

        for i in range(0, len(rgn_status)):

            if rgn_status.loc[i, 'recovered'] != 0.0 and rgn_status.loc[i, 'infected'] != 0.0:

                index = i

                break

            else:

                index = 0

        # Define Initial Condition (necessary for ODE solve)

        R_start = rgn_status.loc[index, 'recovered']/N

        I_start = rgn_status.loc[index, 'infected']/N

        E_start = (rgn_status.loc[index + 4, 'infected'] - rgn_status.loc[index + 3, 'infected'])/N

        S_start = 1 - E_start - I_start

        # Set this values as a tuple for initial condition

        ic = (S_start, E_start, I_start, R_start)

        # Define a time array measure in days, but with values

        time_opt = range(index, len(rgn_status))

        # Create a tuple with the true values in fraction for Infected/Recovered cases (necessary for error measurement)

        i_r_true = (list(rgn_status.loc[index:, 'infected']/N), list(rgn_status.loc[index:, 'recovered']/N))

        # Define a start guess for our parameters [infection_rate, recovered rate, exposed rate]

        params_start_guess = [0.1, 0.01, 0.1]

        # Optimization

        optimal_params, sucess = optimize.leastsq(fit_seir_model,

                                                  x0=params_start_guess,

                                                  args=(time_opt, ic, i_r_true),

                                                  ftol=1.49012e-15,  maxfev=10000)

        # Get calculated parameters

        ir = optimal_params[0]

        rr = optimal_params[1]

        er = optimal_params[2]

        # Get prediction full period time in datetime object and the convert to string

        datetime_pred = pd.date_range(start="2020-02-24",end="2020-07-31", freq='D')

        time_pred = [x.strftime("%Y-%m-%d") for x in datetime_pred]

        # Get a list from 01/April to 31/July 

        time_pred_range = range(0, len(time_pred))

        # Calculate a SEIR prediction 

        future_pred = calculate_seir_model((ir, rr, er), time_pred_range, ic)

        # Generate tuple result for this state

        state_result = (state, ir, rr, er, 0.13*future_pred[:,2])

        # Append to my list

        result.append(state_result)

    # Return

    return result    
# Get results for each region

italy_general_results = extrapolating_italy()
# Create legend

legend_state = ['Lombardia', 'Veneto', 'Piemonte', 'Emilia-Romagna']

# Plot results

## Define plot object

fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(20,10))

## Plot first Lombardia curve

sns.lineplot(x=time_pred, y=future_pred_vent, ax=axes)

## Plot SEIR

for state in italy_general_results:

    state_name = state[0]

    vent_curve = state[4]

    sns.lineplot(x=time_pred, y=vent_curve, ax=axes)

plt.setp(axes.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

axes.legend(loc=0, labels=legend_state, fontsize=15)

axes.set_title('Predicted Ventilators curve for Risk regions in Italy', fontsize=25)

axes.set_xlabel('Date', fontsize=20)

axes.set_ylabel('Total count', fontsize=20)

axes.set_xticks(time_axis);
# Define a sum to count all ventilators for those regions

sum_italy_vents = 0

# Print each Ventilator necessity for each reagion

print('# Risk regions')

print('## Lombardia')

print('ventilators units: ', int(max_vent_necessity))

sum_italy_vents += int(max_vent_necessity)

for i, N in zip(italy_general_results, pop_states):

    print('## ', i[0])

    print('ventilators units: ', int(max(i[4]*N)))

    sum_italy_vents += int(max(i[4]*N))

    

# Print final result

print('\nIn total for this Scenario, Italy would need for only 4 regions: ', sum_italy_vents, ' ventilators.')