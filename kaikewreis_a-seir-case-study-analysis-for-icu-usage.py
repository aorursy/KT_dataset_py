# Standard modules

import numpy as np



# Data modules

import pandas as pd



# Missing Analysis modules

import missingno as msno



# Modelling modules

from scipy import integrate

from scipy import optimize



# Plot modules

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline
# Italy

df_ita_status = pd.read_csv('/kaggle/input/uncover/UNCOVER/github/covid-19-italy-situation-monitoring-by-region.csv')
# Spain

df_spn_status = pd.read_csv('/kaggle/input/uncover/UNCOVER/github/covid19-spain-cases.csv')
# Brazil

df_bra_status = pd.read_csv('/kaggle/input/uncover/UNCOVER/einstein/diagnosis-of-covid-19-and-its-clinical-spectrum.csv')
# Italy

df_ita_status.drop(axis=1, inplace=True, columns=['stato', 'lat', 'long', 'note_it', 'note_en', 'codice_regione', 'denominazione_regione'])
# Spain

df_spn_status.drop(axis=1, inplace=True, columns=['ccaa','ia', 'muertes', 'hospitalizados', 'nuevos'])
# Brazil

## Get only COVID-19 positive cases

df_bra_status = df_bra_status[df_bra_status['sars_cov_2_exam_result'] == 'positive']

## Remove unused columns

df_bra_status.drop(axis=1, inplace=True, columns=['patient_id', 'sars_cov_2_exam_result'])

## Change index

df_bra_status.index = range(0, len(df_bra_status))
# ITALY

## Group by date

ita_status = df_ita_status[['data', 'totale_casi', 'terapia_intensiva','dimessi_guariti']].groupby('data').sum().reset_index()

## Change columns to a specific pattern

ita_status.columns = ['date','infected','icu','recovered']
# SPAIN

## Group by date

spn_status = df_spn_status[['fecha', 'casos', 'uci', 'curados']].groupby('fecha').sum().reset_index()

## Change columns to a specific pattern

spn_status.columns = ['date','infected','icu','recovered']
# BRAZIL

## Create categorical column for patient destination

df_bra_status['patient_destination'] = np.nan

## Loop to insert values

for i in range(0, len(df_bra_status)):

    # Var that holds patient destination

    patient_state = (df_bra_status.loc[i,'patient_addmited_to_regular_ward_1_yes_0_no'], df_bra_status.loc[i, 'patient_addmited_to_semi_intensive_unit_1_yes_0_no'], df_bra_status.loc[i, 'patient_addmited_to_intensive_care_unit_1_yes_0_no'])

    # Define value

    if patient_state == ('f', 'f', 'f'): 

        df_bra_status.loc[i, 'patient_destination'] = 'no_hospitalization'

    elif patient_state == ('t', 'f', 'f'):

        df_bra_status.loc[i, 'patient_destination'] = 'regular_ward'

    elif patient_state == ('f', 't', 'f'):

        df_bra_status.loc[i, 'patient_destination'] = 'semi_intensive_unit'

    elif patient_state == ('f', 'f', 't'):

        df_bra_status.loc[i, 'patient_destination'] = 'intensive_care_unit'

## Drop again unused columns

df_bra_status.drop(axis=1, inplace=True, columns=['patient_addmited_to_regular_ward_1_yes_0_no', 'patient_addmited_to_semi_intensive_unit_1_yes_0_no', 'patient_addmited_to_intensive_care_unit_1_yes_0_no'])
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

    

    # SIR EDO System 

    dsdt = -exposed_rate*s*i

    dedt = (exposed_rate*s*i) - (infection_rate*e)

    didt = (infection_rate*e) - (recovery_rate*i)

    drdt = recovery_rate*i

    

    # Return our system

    return (dsdt, dedt, didt, drdt)
# FUNCTION - Calculate SEIR Model in t (time as days) based on given parameters

def calculate_seir_model(params, t, initial_condition):

    # Create an alias to our seir ode model to pass params to try

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
# Define plot object

fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(20,10))



# Define df to plot here!

df = ita_status



# Plots - ITALY

sns.scatterplot(x='date', y='infected', data=df, ax=axes, color = 'red')

sns.scatterplot(x='date', y='recovered', data=df, ax=axes, color = 'green')

sns.scatterplot(x='date', y='icu', data=df, ax=axes, color = 'darkred')

plt.setp(axes.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

axes.legend(loc=2, labels=['Infected','Recovered','ICU'], fontsize=10)

axes.set_title('ITALY', fontsize=20)

axes.set_xlabel('Date', fontsize=15)

axes.set_ylabel('Total cases', fontsize=15);
# Create a icu_ratio in % for ITALY

for i in range(0, len(ita_status)):

    ita_status.loc[i, 'icu_ratio'] = round(100*(float(ita_status.loc[i, 'icu'])/float(ita_status.loc[i, 'infected'])), 2)
# Get some fast statistics: maximum, minimum and average ICU ratio - ITALY

print('### Italy')

print('Maximum ICU ratio: ', np.nanmax(ita_status['icu_ratio']))

print('Average ICU ratio: ', round(np.nanmean(ita_status['icu_ratio']),2))

print('Minimum ICU ratio: ', np.nanmin(ita_status['icu_ratio']))
# Define plot object

fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(20,10))



# Define df to plot here!

df = ita_status



# Plot - ITALY

sns.barplot(x='date', y='icu_ratio', data=df, ax=axes, color = 'red')

plt.setp(axes.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

axes.set_title('ITALY - ICU Ratio', fontsize=20)

axes.set_xlabel('Date', fontsize=15)

axes.set_ylabel('Ratio', fontsize=15);
# Country population (~)

N = 1000000*60.59
# Define Initial Condition (necessary for ODE solve)

I_start = ita_status.loc[0, 'infected']/N

E_start = (ita_status.loc[4, 'infected'] - ita_status.loc[3, 'infected'])/N

S_start = 1 - E_start - I_start

R_start = ita_status.loc[0, 'recovered']/N

## Set this values as a tuple

ic = (S_start, E_start, I_start, R_start)
print('Start condition:')

print('s(0): ', ic[0])

print('e(0): ', ic[1])

print('i(0): ', ic[2])

print('r(0): ', ic[3])
# Define a time array measure in days

time_opt = range(0, len(ita_status))



# Create a tuple with the true values in fraction for Infected/Recovered cases (necessary for error measurement)

i_r_true = (list(ita_status['infected']/N), list(ita_status['recovered']/N))



# Define a start guess for our parameters [infection_rate, recovered rate]

params_start_guess = [0.01, 0.001, 0.01]
optimal_params, sucess = optimize.leastsq(fit_seir_model,

                                          x0=params_start_guess,

                                          args=(time_opt, ic, i_r_true),

                                          ftol=1.49012e-15)
print('## Italy')

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

axes[0].set_title('Infected cases - ITALY',fontsize=20)

axes[1].plot(time_opt, i_r_true[1], 'g')

axes[1].plot(time_opt, fit_result[:,3], 'y')

axes[1].legend(['Ground truth', 'Predicted'],loc=2, fontsize=15)

axes[1].set_title('Recovered cases - ITALY',fontsize=20);
# Get prediction full period time in datetime object and the convert to string

datetime_pred = pd.date_range(start="2020-02-24",end="2020-07-31", freq='D')

time_pred = [x.strftime("%Y-%m-%d") for x in datetime_pred]



# Get a list from 01/April to 31/July 

time_pred_range = range(0, len(time_pred))
# Calculate a SIR prediction 

future_pred = calculate_seir_model((ir, rr, er), time_pred_range, ic)
# Plot results

## Define Date axis to better visualization (only first/half/last day of every month)

time_axis = [time_pred[i] for i in [0,6,20,37,51,67,81,98,112,128,142,158]]

## Define plot object

fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(20,10))

## Plot SIR

sns.lineplot(x=time_pred, y=future_pred[:,0], ax=axes, color = 'blue')

sns.lineplot(x=time_pred, y=future_pred[:,1], ax=axes, color = 'red')

sns.lineplot(x=time_pred, y=future_pred[:,2], ax=axes, color = 'purple')

sns.lineplot(x=time_pred, y=future_pred[:,3], ax=axes, color = 'green')

plt.setp(axes.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

axes.legend(loc=1, labels=['Suscetible', 'Exposed', 'Infected','Recovered'], fontsize=10)

axes.set_title('ITALY - SEIR predictions', fontsize=20)

axes.set_xlabel('Date', fontsize=15)

axes.set_ylabel('Total cases', fontsize=15)

axes.set_xticks(time_axis);
# Calculate icu curve based in SEIR infected curve

future_pred_icu = 0.05*future_pred[:,2]
# Plot results

## Define plot object

fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(20,10))

## Plot SIR

sns.lineplot(x=time_pred, y=future_pred_icu, ax=axes, color = 'red')

sns.lineplot(x=time_pred, y=future_pred[:,2], ax=axes, color = 'purple')

plt.setp(axes.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

axes.legend(loc=1, labels=['ICU - OMS Statement','Infected'], fontsize=10)

axes.set_title('ITALY - SEIR ICU and Infected curves', fontsize=20)

axes.set_xlabel('Date', fontsize=15)

axes.set_ylabel('Total cases', fontsize=15)

axes.set_xticks(time_axis);
# ICU beds in Italy

icu_beds_value = 12.5*100000/N



# ICU beds array

icu_beds_100 = [icu_beds_value] * len(future_pred_icu)
# Create icu threshold scenarios - 25%

icu_beds_threshold = icu_beds_value*0.25

icu_beds_25 = [icu_beds_threshold] * len(future_pred_icu)

# Create icu threshold scenarios - 50%

icu_beds_threshold = icu_beds_value*0.50

icu_beds_50 = [icu_beds_threshold] * len(future_pred_icu)

# Create icu threshold scenarios - 75%

icu_beds_threshold = icu_beds_value*0.75

icu_beds_75 = [icu_beds_threshold] * len(future_pred_icu)
# Plot results

## Define plot object

fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(20,10))

## Plot SEIR

sns.lineplot(x=time_pred, y=future_pred_icu, ax=axes, color = 'purple')

sns.lineplot(x=time_pred, y=icu_beds_100, ax=axes, color = 'red')

sns.lineplot(x=time_pred, y=icu_beds_75, ax=axes, color = 'darkred')

sns.lineplot(x=time_pred, y=icu_beds_50, ax=axes, color = 'gray')

sns.lineplot(x=time_pred, y=icu_beds_25, ax=axes, color = 'black')

plt.setp(axes.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

axes.legend(loc=1, labels=['ICU - OMS Statement','ICU beds (100%)','ICU beds (75%)','ICU beds (50%)','ICU beds (25%)'], fontsize=10)

axes.set_title('ITALY - SIR ICU necessity and ICU beds', fontsize=20)

axes.set_xlabel('Date', fontsize=15)

axes.set_ylabel('Total', fontsize=15)

axes.set_xticks(time_axis);
# Define plot object

fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(20,10))



# Define df to plot here!

df = spn_status



# Plots - ITALY

sns.scatterplot(x='date', y='infected', data=df, ax=axes, color = 'red')

sns.scatterplot(x='date', y='recovered', data=df, ax=axes, color = 'green')

sns.scatterplot(x='date', y='icu', data=df, ax=axes, color = 'darkred')

plt.setp(axes.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

axes.legend(loc=2, labels=['Infected','Recovered','ICU'], fontsize=10)

axes.set_title('SPAIN', fontsize=20)

axes.set_xlabel('Date', fontsize=15)

axes.set_ylabel('Total cases', fontsize=15);
# Create a icu_ratio in % for SPAIN

for i in range(0, len(spn_status)):

    if spn_status.loc[i, 'icu'] != 0 and spn_status.loc[i, 'infected'] != 0:

        spn_status.loc[i, 'icu_ratio'] = round(100*(float(spn_status.loc[i, 'icu'])/float(spn_status.loc[i, 'infected'])), 2)

    else:

        spn_status.loc[i, 'icu_ratio'] = 0
# Get some fast statistics: maximum, minimum and average ICU ratio - SPAIN

print('### Spain')

print('Maximum ICU ratio: ', np.nanmax(spn_status['icu_ratio']))

print('Average ICU ratio: ', round(np.nanmean(spn_status['icu_ratio']),2))

print('Minimum ICU ratio: ', np.nanmin(spn_status['icu_ratio']))
# Define plot object

fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(20,10))



# Define df to plot here!

df = spn_status



# Plot - ITALY

sns.barplot(x='date', y='icu_ratio', data=df, ax=axes, color = 'red')

plt.setp(axes.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

axes.set_title('SPAIN - ICU Ratio', fontsize=20)

axes.set_xlabel('Date', fontsize=15)

axes.set_ylabel('Ratio', fontsize=15);
ita_status
spn_status
# Country population (~)

N = 1000000*47.01
# Define Initial Condition (necessary for ODE solve)

I_start = spn_status.loc[0, 'infected']/N

E_start = (spn_status.loc[4, 'infected'] - spn_status.loc[3, 'infected'])/N

S_start = 1 - E_start - I_start

R_start = spn_status.loc[0, 'recovered']/N

## Set this values as a tuple

ic = (S_start, E_start, I_start, R_start)
print('Start condition:')

print('s(0): ', ic[0])

print('e(0): ', ic[1])

print('i(0): ', ic[2])

print('r(0): ', ic[3])
# Define a time array measure in days

time_opt = range(0, len(spn_status))



# Create a tuple with the true values in fraction for Infected/Recovered cases (necessary for error measurement)

i_r_true = (list(spn_status['infected']/N), list(spn_status.loc[21:, 'recovered']/N))



# Define a start guess for our parameters [infection_rate, recovered rate]

params_start_guess = [0.01, 0.001, 0.01]
# FUNCTION - Auxiliar function to find the best parameters

def fit_seir_model(params_to_fit, t, initial_condition, i_r_true):

    # Calculate ODE solution for possible parameter, return values to each dependent variable:

    # (s, e, i and r)

    fit_result = calculate_seir_model(params_to_fit, t, initial_condition)

    

    # Calculate residual value between predicted VS true

    ## Note: ode_result[0] is S result

    residual_i = i_r_true[0] - fit_result[:,2]

    residual_r = i_r_true[1] - fit_result[21:,3]



    # Create a np.array of all residual values for both (i) and (r)

    residual = np.concatenate((residual_i, residual_r))

    

    # Return results

    return residual
optimal_params, sucess = optimize.leastsq(fit_seir_model,

                                          x0=params_start_guess,

                                          args=(time_opt, ic, i_r_true),

                                          ftol=1.49012e-15)
print('## Spain')

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

axes[0].set_title('Infected cases - SPAIN',fontsize=20)

axes[1].plot(range(21,31), i_r_true[1], 'g')

axes[1].plot(range(21,31), fit_result[21:,3], 'y')

axes[1].legend(['Ground truth', 'Predicted'],loc=2, fontsize=15)

axes[1].set_title('Recovered cases (03/23 to 04/01) - SPAIN',fontsize=20);
# Get prediction full period time in datetime object and the convert to string

datetime_pred = pd.date_range(start="2020-03-02",end="2020-07-31", freq='D')

time_pred = [x.strftime("%Y-%m-%d") for x in datetime_pred]
# Get a list from 01/April to 31/July 

time_pred_range = range(0, len(time_pred))
# Calculate a SIR prediction 

future_pred = calculate_seir_model((ir, rr, er), time_pred_range, ic)
# Plot results

## Define Date axis to better visualization (only first/half/last day of every month)

time_axis = [time_pred[i] for i in [0,13,30,44,60,74,91,105,121,135, 151]]

## Define plot object

fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(20,10))

## Plot SIR

sns.lineplot(x=time_pred, y=future_pred[:,0], ax=axes, color = 'blue')

sns.lineplot(x=time_pred, y=future_pred[:,1], ax=axes, color = 'red')

sns.lineplot(x=time_pred, y=future_pred[:,2], ax=axes, color = 'purple')

sns.lineplot(x=time_pred, y=future_pred[:,3], ax=axes, color = 'green')

plt.setp(axes.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

axes.legend(loc=1, labels=['Suscetible', 'Exposed', 'Infected','Recovered'], fontsize=10)

axes.set_title('SPAIN - SEIR predictions', fontsize=20)

axes.set_xlabel('Date', fontsize=15)

axes.set_ylabel('Total cases', fontsize=15)

axes.set_xticks(time_axis);

# Calculate icu curve based in SEIR infected curve

future_pred_icu = 0.05*future_pred[:,2]
# Plot results

## Define plot object

fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(20,10))

## Plot SIR

sns.lineplot(x=time_pred, y=future_pred_icu, ax=axes, color = 'red')

sns.lineplot(x=time_pred, y=future_pred[:,2], ax=axes, color = 'purple')

plt.setp(axes.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

axes.legend(loc=1, labels=['ICU - OMS Statement','Infected'], fontsize=10)

axes.set_title('SPAIN - SEIR ICU and Infected curves', fontsize=20)

axes.set_xlabel('Date', fontsize=15)

axes.set_ylabel('Total cases', fontsize=15)

axes.set_xticks(time_axis);

# ICU beds in Spain

icu_beds_value = 9.7*100000/N



# ICU beds array

icu_beds_100 = [icu_beds_value] * len(future_pred_icu)
# Create icu threshold scenarios - 25%

icu_beds_threshold = icu_beds_value*0.25

icu_beds_25 = [icu_beds_threshold] * len(future_pred_icu)

# Create icu threshold scenarios - 50%

icu_beds_threshold = icu_beds_value*0.50

icu_beds_50 = [icu_beds_threshold] * len(future_pred_icu)

# Create icu threshold scenarios - 75%

icu_beds_threshold = icu_beds_value*0.75

icu_beds_75 = [icu_beds_threshold] * len(future_pred_icu)
# Plot results

## Define plot object

fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(20,10))

## Plot SIR

sns.lineplot(x=time_pred, y=future_pred_icu, ax=axes, color = 'purple')

sns.lineplot(x=time_pred, y=icu_beds_100, ax=axes, color = 'red')

sns.lineplot(x=time_pred, y=icu_beds_75, ax=axes, color = 'darkred')

sns.lineplot(x=time_pred, y=icu_beds_50, ax=axes, color = 'gray')

sns.lineplot(x=time_pred, y=icu_beds_25, ax=axes, color = 'black')

plt.setp(axes.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

axes.legend(loc=1, labels=['ICU - OMS Statement','ICU beds (100%)','ICU beds (75%)','ICU beds (50%)','ICU beds (25%)'], fontsize=10)

axes.set_title('SPAIN - SIR ICU necessity and ICU beds', fontsize=20)

axes.set_xlabel('Date', fontsize=15)

axes.set_ylabel('Total', fontsize=15)

axes.set_xticks(time_axis);
# Eval missing value for each column again

nan_per_column = pd.DataFrame(df_bra_status.isna().sum(),columns=['nanValues']).reset_index()



# Calculate NaN %

for i in range(0,len(nan_per_column)):

    nan_per_column.loc[i, 'nanValuesPct'] = 100*round(nan_per_column.loc[i, 'nanValues']/len(df_bra_status),3)

    

# Plot - % of missing rows for each column

plt.figure(figsize=(20,10))

sns.barplot(x="index", y="nanValuesPct", data=nan_per_column)

plt.xlabel('Variables', fontsize=20)

plt.ylabel('Missing %', fontsize=20)

plt.title('Missing % for each column', fontsize=30)

plt.yticks([0,10,20,30,40,50,60,70,80,90,100])

plt.xticks(rotation=90);
# Visualize how much hospitalized patient samples do we have

df_bra_status['patient_destination'].value_counts()
# Create a new dataset to contains only hospitalized patients

df_bra_hosp_status = df_bra_status[df_bra_status['patient_destination'] != 'no_hospitalization']
# Missing plot

# Eval missing value for each column again

nan_per_column = pd.DataFrame(df_bra_hosp_status.isna().sum(),columns=['nanValues']).reset_index()



# Calculate NaN %

for i in range(0,len(nan_per_column)):

    nan_per_column.loc[i, 'nanValuesPct'] = 100*round(nan_per_column.loc[i, 'nanValues']/len(df_bra_hosp_status),3)

    

# Plot - % of missing rows for each column

plt.figure(figsize=(20,10))

sns.barplot(x="index", y="nanValuesPct", data=nan_per_column)

plt.xlabel('Variables', fontsize=20)

plt.ylabel('Missing %', fontsize=20)

plt.title('Missing % for each column', fontsize=30)

plt.yticks([0,10,20,30,40,50,60,70,80,90,100])

plt.xticks(rotation=90);
# Define age and quantile list

age_list = ['0-5','6-10','11-15','16-20','21-25','26-30','31-35','36-40','41-45','46-50','51-55','56-60','61-65','66-70','71-75','76-80','81-85','86-90','91-95','96-100']



# Create specific column

df_bra_status['age_interval'] = np.nan
for i in range(0, len(df_bra_status)):

    # Get index related (age_list have the equivalent index of patient_age_quantile_column)

    age_list_index = int(df_bra_status.loc[i, 'patient_age_quantile'])

    # Insert range

    df_bra_status.loc[i, 'age_interval'] = age_list[age_list_index]    
plt.figure(figsize=(20,10))

# Plot both columns

sns.countplot(x='age_interval', hue='patient_destination', data=df_bra_status)

plt.xlabel('Age interval', fontsize=20)

plt.ylabel('Count', fontsize=20)

plt.title('Age distribution in brazilian positive cases', fontsize=30)

plt.legend(loc=2)

plt.xticks(range(0,20), age_list, rotation=45);