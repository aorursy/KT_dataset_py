# Algebra

import numpy as np



# Dataframe

import pandas as pd



# Missing Analysis

import missingno as msno



# Modelling

from scipy import integrate

from scipy import optimize



# Plotting

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns



# Map

import folium



# Datetime

from datetime import datetime

# Global cases

global_confirm_csv = 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv'

global_death_csv = 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_global.csv'

global_recover_csv = 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_recovered_global.csv'

# US cases

us_confirm_csv = 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_US.csv'

us_death_csv = 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_US.csv'

us_state_loc_csv = 'https://gist.githubusercontent.com/mbostock/9535021/raw/eaed7e5632735a6609f02d0ba0e55c031e14200d/us-state-capitals.csv'
Confirmed = pd.read_csv(global_confirm_csv)

Death = pd.read_csv(global_death_csv) 

Recovered = pd.read_csv(global_recover_csv)

Confirmed_us = pd.read_csv(us_confirm_csv)

Death_us = pd.read_csv(us_death_csv)

us_state_location = pd.read_csv(us_state_loc_csv)
pd.set_option('display.max_rows', None)

pd.set_option('display.max_columns', None)

pd.set_option('display.width', None)

pd.set_option('display.max_colwidth', None)
# Global cases

Confirmed.drop(axis=1, inplace=True, columns=['Province/State', 'Lat', 'Long'])

Death.drop(axis=1, inplace=True, columns=['Province/State', 'Lat', 'Long'])

Recovered.drop(axis=1, inplace=True, columns=['Province/State', 'Lat', 'Long'])
display(Confirmed)
# function to extract the data of different cases in different country

def extract_national_data(df, country, case):

    df_extract = df[df['Country/Region'] == country].sum(axis=0)

    df_extract = df_extract.T.reset_index()

    df_extract = df_extract.iloc[1:]

    df_extract.columns = ['date', case]

    df_extract.date = pd.to_datetime(df_extract.date)

    return df_extract
# function to join three types of cases in one country

def join_national_data(confirmed, death, recovered):

    total_case = confirmed.join(death.set_index('date'), on='date').join(recovered.set_index('date'), on='date')

    total_case.reset_index(inplace=True, drop=True)

    return total_case
us_confirmed = extract_national_data(Confirmed, 'US', 'positive')

us_death = extract_national_data(Death, 'US', 'death')

us_recovered = extract_national_data(Recover, 'US', 'recovered')
us_status = join_national_data(us_confirmed, us_death, us_recovered)
display(us_status.head())
itl_confirmed = extract_national_data(Confirmed, 'Italy', 'positive')

itl_death = extract_national_data(Death, 'Italy', 'death')

itl_recovered = extract_national_data(Recovered, 'Italy', 'recovered')
itl_status = join_national_data(itl_confirmed, itl_death, itl_recovered)
display(itl_status.head())
frc_confirmed = extract_national_data(Confirmed, 'France', 'positive')

frc_death = extract_national_data(Death, 'France', 'death')

frc_recovered = extract_national_data(Recovered, 'France', 'recovered')
frc_status = join_national_data(frc_confirmed, frc_death, frc_recovered)
display(frc_status.head())
# map

us_state_loc.drop(axis=1, inplace=True, columns=['description'])

us_state_loc = us_state_loc.rename(columns={'name': 'State'})
# state-wise death cases dataset

Death_us_state.drop(axis=1, inplace=True, columns=['Lat', 'Long_', 'UID', 'iso2','iso3', 'code3', 'FIPS', 'Country_Region', 'Combined_Key'])

Death_us_state = Death_us_state.groupby('Province_State').sum()

Death_us_state.reset_index(inplace=True)

Death_us_state = Death_us_state.rename(columns={"Province_State": 'State'})

us_state_death = Death_us_state.join(us_state_loc.set_index('State'), on='State')

us_state_death = us_state_death.dropna()

state_pop = us_state_death.loc[:, ['State','Population']]



# state-wise confirmed cases dataset

Confirmed_us_state.drop(axis=1, inplace=True, columns=['Lat', 'Long_', 'UID', 'iso2','iso3', 'code3', 'FIPS', 'Country_Region', 'Combined_Key'])

Confirmed_us_state = Confirmed_us_state.groupby('Province_State').sum()

Confirmed_us_state.reset_index(inplace=True)

Confirmed_us_state = Confirmed_us_state.rename(columns={"Province_State": 'State'})

Confirmed_state_loc = Confirmed_us_state.join(us_state_loc.set_index('State'), on='State')

Confirmed_state_loc_pop = Confirmed_state_loc.join(state_pop.set_index('State'), on= 'State')

Confirmed_state_loc_pop = Confirmed_state_loc_pop.dropna()

cols = Confirmed_state_loc_pop.columns.tolist()

cols.insert(1, cols.pop(cols.index('Population')))

Confirmed_state_loc_pop = Confirmed_state_loc_pop.reindex(columns=cols)
today='7/5/20'

Top_states = Confirmed_us_state.sort_values([today], ascending=False)

Top_states.reset_index(inplace=True, drop=True)

Top_states = Top_states.loc[0:9,['State', today]]

print(Top_states)
Confirmed_us_state_t = Confirmed_us_state.transpose()

Confirmed_us_state_t.reset_index(inplace=True)

header = Confirmed_us_state_t.iloc[0]

Confirmed_us_state_t = Confirmed_us_state_t[1:]

Confirmed_us_state_t.columns = header

Confirmed_us_state_t = Confirmed_us_state_t.rename(columns={'State' : 'Date'})
yaxis = np.linspace(0,500000, num=11, endpoint=True)

yaxis = yaxis.astype(int)

x = np.linspace(1,180,num=10,endpoint=False)

x = x.astype(int)

xaxis = Confirmed_us_state_t.iloc[x,0]
sns.set()

plt.figure(figsize = (20,12))

for state in Top_states['State']:

    plt.plot(Confirmed_us_state_t['Date'], Confirmed_us_state_t[state])

plt.legend(header[1:], loc=0)

plt.yticks(yaxis, yaxis)

plt.xticks(xaxis, xaxis)

plt.show()
us_map_confirm = folium.Map(

         location=[37.0902, -95.7129],

         zoom_start = 4,

         tiles='CartoDB dark_matter')



    

for i in range(0,len(Confirmed_state_loc_pop)):

    r = (int(Confirmed_state_loc_pop.iloc[i, -3])/((int(Confirmed_state_loc_pop.iloc[i, 1])/1000)))*5

    lat = float(Confirmed_state_loc_pop.iloc[i, -2])

    long = float(Confirmed_state_loc_pop.iloc[i, -1])

    folium.vector_layers.CircleMarker(

            name = 'Confirmed case',

            radius = r,

            location = [lat, long],

            tooltip=str(Confirmed_state_loc_pop.iloc[i, 0]) + ' confirmed: ' + str(Confirmed_state_loc_pop.iloc[i, -3]),

            color = 'cadetblue',

            weight = .5,

            fill = True,

            alpha = 0.3,

            ).add_to(us_map_confirm)
us_map_confirm
us_map_death = folium.Map(

         location=[37.0902, -95.7129],

         zoom_start = 4,

         tiles='CartoDB dark_matter')



for i in range(0,len(us_state_death)):

    r = (int(us_state_death.iloc[i, -3])/(int(us_state_death.iloc[i, 1]/1000)))*50

    lat = float(us_state_death.iloc[i, -2])

    long = float(us_state_death.iloc[i, -1])

    folium.vector_layers.CircleMarker(

            name = 'Death case', 

            radius = r,

            location = [lat, long],

            tooltip=str(us_state_death.iloc[i, 0]) + 'death: ' + str(us_state_death.iloc[i, -3]),

            color = 'crimson',

            weight = .5,

            fill = True,

            alpha = 0.3,

            ).add_to(us_map_death)
us_map_death
# Function

def seir_model_ode(y,t, params): 

    '''

    Arguments:

    - y: dependent variables

    - t: independent variable (time)

    - params: Model parameters

    '''

    # Parameters to find

    infection_rate = params[0]

    recovery_rate = params[1]

    exposed_rate = params[2]

    death_rate = params[3]

    reinfection_rate = params[4]

    

    # Y variables

    s = y[0]

    e = y[1]

    i = y[2]

    d = y[3]

    r = y[4]

    

    # SIR ODE System 

    dsdt = -exposed_rate*s*(i+e) + reinfection_rate*r

    dedt = (exposed_rate*s*(i+e)) - (infection_rate*e)

    didt = (infection_rate*e) - (recovery_rate*i) - (death_rate*i)

    dddt = death_rate*i

    drdt = recovery_rate*i - reinfection_rate*r

    

    # Return our system

    return (dsdt, dedt, didt, dddt, drdt)
# FUNCTION - Calculate SEIR Model in t (time as days) based on given parameters

def calculate_seir_model(params, t, initial_condition):

    # Create an alias to our seir ode model to pass params to try

    seir_ode = lambda y,t:seir_model_ode(y,t, params)

    

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

    residual_d = i_r_true[1] - fit_result[:,3]

    residual_r = i_r_true[2] - fit_result[:,4]

    

    # Create a np.array of all residual values for both (i) and (r)

    residual = np.concatenate((residual_i, residual_d, residual_r))

    

    # Return results

    return residual
def countryLockdown(N, status,lockdownDay,countryName):

    

    """inputs:

    N: population in the country

    status: dataframe of the country

    lockdownDay: number of days since begining of data to the date of lockdown

    """

    

    # Define Initial Condition before lockdown

    I_start = status.loc[0, 'positive']/N

    E_start = (status.loc[14, 'positive'] - status.loc[0, 'positive'])/N

    S_start = 1 - E_start - I_start

    D_start = 0

    R_start = status.loc[0, 'recovered']/N

    

    # Set this values as a tuple of initial condition

    ic = (S_start, E_start, I_start, D_start, R_start)

    

    # Create a tuple with the true values in fraction for Infected/Recovered cases (necessary for error measurement)

    beforelockdown=status.loc[0:lockdownDay]

    afterlockdown=status.loc[lockdownDay+1:]

    i_r_true_bf = (list(beforelockdown['positive']/N),list(beforelockdown['death']/N), list(beforelockdown['recovered']/N))

    i_r_true_af = (list(afterlockdown['positive']/N), list(afterlockdown['death']/N), list(afterlockdown['recovered']/N))



    # Define a time array measure in days

    time_opt_bf = range(0, lockdownDay+1)

    time_opt_af = range(0, len(afterlockdown))

    time_opt =range(0,len(status))

    

    

    # define initial condition after lockdown

    E_start_day = min(len(status['date']),len(beforelockdown)+14)

    

    E_start_af = (status.loc[E_start_day, 'positive'] - status.loc[len(beforelockdown), 'positive'])/N

    I_start_af = status.loc[len(beforelockdown), 'positive']/N

    S_start_af = 1 - E_start_af - I_start_af

    D_start_af = status.loc[len(beforelockdown), 'death']/N

    R_start_af = status.loc[len(beforelockdown), 'recovered']/N

   

    # Set this values as a tuple

    ic_af = (S_start_af, E_start_af, I_start_af, D_start_af, R_start_af)

    

    # Define a start guess for our parameters [infection_rate, recovered rate]

    params_start_guess = [0.00, 0.000, 0.0, 0.00, 0.00]

    optimal_params, sucess = optimize.leastsq(fit_seir_model,

                                          x0=params_start_guess,

                                          args=(time_opt_bf, ic, i_r_true_bf),

                                          ftol=1.49012e-22)

    optimal_params_af, sucess = optimize.leastsq(fit_seir_model,

                                          x0=params_start_guess,

                                          args=(time_opt_af, ic_af, i_r_true_af),

                                          ftol=1.49012e-22)

    

    print('## '+countryName+' before lockdown')

    print('Optimized infection rate: ', optimal_params[0])

    print('Optimized recovered rate: ', optimal_params[1])

    print('Optimized exposed rate: ', optimal_params[2])

    print('Optimized death rate: ', optimal_params[3])

    print('Optimized reinfection rate: ', optimal_params[4])

    print('\n')

    print('## '+countryName+' after lockdown')

    print('Optimize infection rate: ', optimal_params_af[0])

    print('Optimize recovered rate: ', optimal_params_af[1])

    print('Optimize exposed rate: ', optimal_params_af[2])

    print('Optimized death rate: ', optimal_params_af[3])

    print('Optimized reinfection rate: ', optimal_params_af[4])

            

            

    # Fit test

    ## Get the optimal parameters

    ir = optimal_params[0]

    rr = optimal_params[1]

    er = optimal_params[2]

    dr = optimal_params[3]

    rir = optimal_params[4]

    ir_af = optimal_params_af[0]

    rr_af = optimal_params_af[1]

    er_af = optimal_params_af[2]

    dr_af = optimal_params_af[3]

    rir_af = optimal_params_af[4]

    

    ## Calculate a curve based on those parameters

    fit_result_bf = calculate_seir_model((ir, rr, er, dr, rir), time_opt_bf, ic)

    fit_result_af = calculate_seir_model((ir_af, rr_af, er_af, dr_af, rir_af), time_opt_af, ic_af)

    print(fit_result_af[0,2])

    print(ic_af[2])

    ## Define plot object

    fig, (ax1, ax2, ax3) = plt.subplots(nrows=3, ncols=1, sharex=True, figsize=[12, 18])

    ## Plot real and predicted infection

    ax1.set_title('Infected cases - '+ countryName,fontsize=20)

    ax1.plot(time_opt, i_r_true_bf[0]+i_r_true_af[0], 'ro', markersize = 1)

    ax1.plot(time_opt, np.hstack((fit_result_bf[:,2],fit_result_af[:,2])), 'co')

    ax1.legend(['Actual infection', 'Predicted infection'],loc=2, fontsize=8)

    ax1.set_ylabel('Proportion of population', fontsize=12)

    ## Plot real and predicted death

    ax2.set_title('Death cases - '+ countryName,fontsize=20)

    ax2.plot(time_opt, i_r_true_bf[1]+i_r_true_af[1], 'ro', markersize = 1)

    ax2.plot(time_opt, np.hstack((fit_result_bf[:,3],fit_result_af[:,3])), 'bo')

    ax2.legend(['Actual death', 'Predicted death'],loc=2, fontsize=8)

    ax2.set_ylabel('Proportion of population', fontsize=12)

    ## Plot real and predicted recover

    ax3.set_title('Recovered cases - '+countryName,fontsize=20)      

    ax3.plot(time_opt, i_r_true_bf[2]+i_r_true_af[2], 'ro', markersize = 1)

    ax3.plot(time_opt, np.hstack((fit_result_bf[:,4],fit_result_af[:,4])), 'go')

    ax3.legend(['Real recover', 'Predicted recover'],loc=2, fontsize=8)

    ax3.set_xlabel('Days since Jan-22-2020', fontsize=12)

    ax3.set_ylabel('Proportion of population', fontsize=12)

    

    # Prediction

    ## Get prediction full period time in datetime object and the convert to string

    datetime_pred = pd.date_range(start="2020-02-01",end="2021-01-01", freq='D')

    pred_time = [x.strftime("%Y-%m-%d") for x in datetime_pred]

    pred_range = range(0, len(pred_time))

    pred_result = calculate_seir_model((ir_af, rr_af, er_af, dr_af, rir_af), pred_range, ic_af)

    time_axis = [pred_time[i] for i in[0, 29, 60, 90, 121, 151, 182, 213, 243, 274, 304, 335]]

    time_labels = ['Feb.', 'Mar.', 'Apr.', 'May', 'June', 'July', 'Aug.', 'Sept.', 'Oct.', 'Nov.', 'Dec.', 'Jan.']

    ## Plot SEIDR

    fig, ax = plt.subplots(figsize=[12,10])

    ax.plot(pred_time, pred_result[:,0],color='blue') #susceptible

    ax.plot(pred_time, pred_result[:,1],color='red') #exposed

    ax.plot(pred_time, pred_result[:,2],color='cyan') #infected

    ax.plot(pred_time, pred_result[:,3], color = 'black') #death

    ax.plot(pred_time, pred_result[:,4],color='green') #recovered

    ax.legend(loc=1, labels=['Susceptible', 'Exposed', 'Infected', 'Death', 'Recovered'], fontsize=8)

    ax.set_title('SEIR predictions', fontsize=20)

    ax.set_xlabel('Month', fontsize=12)

    ax.set_ylabel('Proportion of population', fontsize=12)

    plt.xticks(time_axis, time_labels, rotation='vertical');
def countryNolockdown(N, status,countryName):

    # Define Initial Condition (necessary for ODE solve)

    I_start = status.loc[0, 'positive']/N

    E_start = (status.loc[14, 'positive'] - status.loc[0, 'positive'])/N

    S_start = 1 - E_start - I_start

    R_start = status.loc[0, 'recovered']/N

    ## Set this values as a tuple

    ic = (S_start, E_start, I_start, R_start)

    i_r_true = (list(status['positive']/N), list(status['recovered']/N))

    time_opt =range(0,len(status))

    # Define a start guess for our parameters [infection_rate, recovered rate]

    params_start_guess = [0.01, 0.001, 0.01]

    optimal_params, sucess = optimize.leastsq(fit_seir_model,

                                          x0=params_start_guess,

                                          args=(time_opt, ic, i_r_true),

                                          ftol=1.49012e-15)

    print('## '+countryName)

    print('Optimize infection rate: ', optimal_params[0])

    print('Optimize recovered rate: ', optimal_params[1])

    print('Optimize exposed rate: ', optimal_params[2])

    # Get the optimal parameters

    ir = optimal_params[0]

    rr = optimal_params[1]

    er = optimal_params[2]

    fit_result = calculate_seir_model((ir, rr, er), time_opt, ic)

    

    # Plot the results for Infected/Recovered

    ## Define plot object

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(20,10))



    ## Plot process

    axes[0].plot(time_opt, i_r_true[0], 'ro')

    axes[0].plot(time_opt, fit_result[:,2], 'p')

    axes[0].legend(['Ground truth', 'Predicted'],loc=2, fontsize=15)

    axes[0].set_title('Infected cases - '+countryName,fontsize=20)

    axes[1].plot(time_opt, i_r_true[1], 'ro')

    axes[1].plot(time_opt, fit_result[:,3], 'r')

    axes[1].legend(['Ground truth', 'Predicted'],loc=2, fontsize=15)

    axes[1].set_title('Recovered cases - '+countryName,fontsize=20);

    plt.show()

    

    # Prediction

    # Get prediction full period time in datetime object and the convert to string

    datetime_pred = pd.date_range(start="2020-04-01",end="2020-12-31", freq='D')

    pred_time = [x.strftime("%Y-%m-%d") for x in datetime_pred]

    pred_range = range(0, len(pred_time))

    pred_result = calculate_seir_model((ir, rr, er), pred_range, ic)

    pred_icu = (pred_result[:,1]+pred_result[:,2])*0.08 #predict icu requirement based on 5% death rate

    time_axis = [pred_time[i] for i in[0, 30, 61, 91, 122, 153, 183, 214, 244, 274]]

    ## Plot SEIR

    plt.figure(figsize=[20,10])

    plt.plot(pred_time, pred_result[:,0], color = 'blue') 

    plt.plot(pred_time, pred_result[:,1], color = 'red')

    plt.plot(pred_time, pred_result[:,2], color = 'purple')

    plt.plot(pred_time, pred_result[:,3], color = 'green')

    plt.plot(pred_time, pred_icu,color='black')

    plt.legend(loc=1, labels=['Susceptible', 'Exposed', 'Infected','Recovered','ICU'], fontsize=10)

    plt.title('SEIR predictions', fontsize=20)

    plt.xlabel('Date', fontsize=15)

    plt.ylabel('Total cases', fontsize=15)

    plt.xticks(time_axis)

    #plt.hlines(y=icu/N, xmin=pred_time[0], xmax=pred_time[274], color = 'yellow', linestyles = 'dashed');
countryLockdown(328*1e6,us_status,58,'US')
countryLockdown(60.36*1e6, itl_status,47,'Italy')
countryLockdown(66.99*1e6,frc_status,55,'France')
