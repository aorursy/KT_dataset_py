# data wrangling

import pandas as pd

import numpy as np

from datetime import datetime, date, timedelta



# data visualization

import matplotlib.pyplot as plt

import seaborn as sns

import plotly.express as px



# offline interactive visualization

from plotly.offline import plot, iplot, init_notebook_mode

init_notebook_mode(connected=True)



# regression

import statsmodels.api as sm

from statsmodels.formula.api import ols

import statsmodels.graphics.api as smg



import warnings

warnings.filterwarnings("ignore")
# Worldometer data

# ================



worldometer_data = pd.read_csv('../input/corona-virus-report/worldometer_data.csv')



# Replace missing values '' with NAN and then 0

worldometer_data = worldometer_data.replace('', np.nan).fillna(0)



# Correcting Country name 

worldometer_data['Country/Region'].replace({'USA':'US', 'UAE':'United Arab Emirates', 'S. Korea':'South Korea', \

                                           'UK':'United Kingdom'}, inplace=True)



# Grouped by day, country

# =======================



full_grouped = pd.read_csv('../input/corona-virus-report/full_grouped.csv')



# Merge in population data

full_grouped = full_grouped.merge(worldometer_data[['Country/Region', 'Population']], how='left', on='Country/Region')



full_grouped['Date'] = pd.to_datetime(full_grouped['Date'], format = '%Y-%m-%d')
def sir_model(I0=0.01, beta=0.6, gamma=0.1, days=365, date=date.today()):

    """

    Function will take in initial state for infected population,

    Transmission rate (beta) and recovery rate(gamma) as input.

    

    The function returns the maximum percentage of infectious population,

    the number of days to reach the maximum (inflection point),

    the maximum percentage of population infected,

    the number of days to reach 80% of the maximum percentage of population infected.

    

    """

    ## Initialize model parameters

    N = 1          #Total population in percentage, i.e., 1 = 100%

    I = I0         #Initial state of I default value 1% of population, i.e., I0 = 0.01

    S = N - I      #Initial state of S

    R = 0          #Initial State of R

    C = I          #Initial State of Total Cases

    beta  = beta   #Transmission Rate

    gamma = gamma  #Recovery Rate



    ## Initialize empty lists

    inf  = []       # List of Infectious population for each day

    day  = []       # Time period in day

    suc  = []       # List of Susceptible population for each day

    rec  = []       # List of Recovered population for each day

    conf = []       # List of Total Cases population for each day

    

    ## Project into the future

    for i in range(days):

        day.append(i)

        inf.append(I)

        suc.append(S)

        rec.append(R)

        conf.append(C)



        new_inf= I*S*beta/N            #New infections equation (1)   

        new_rec= I*gamma               #New Recoveries equation (2)

        

        I=I+new_inf-new_rec            #Total infectious population for next day

        S=max(min(S - new_inf, N), 0)  #Total infectious population for next day

        R=min(R + new_rec, N)          #Total recovered population for next day

        

        C=C+new_inf                    #Total confirmed cases for next day



    ## Pinpoint important milestones    

    max_inf = round(np.array(inf).max()*100,2)        #Peak infectious population in percentage

    inflection_day = inf.index(np.array(inf).max())   #Peak infectious population in days

    max_conf = round(np.array(conf).max()*100,2)      #Overall infected population in percentage

    plateau_day = np.array(np.where(np.array(conf) >= 0.8*np.array(conf).max())).min()   #Peak infectious population in days

        

    print(f"Maximum Infectious population at a time :{max_inf}%")

    print(f"Number of Days to Reach Maximum Infectious Population (Inflection Point):{inflection_day} days or {date + timedelta(days=inflection_day)}")

    print(f"Total Infected population :{max_conf}%")

    print(f"Number of Days to Reach 80% of the Projected Confirmed Cases (Plateau Point):{plateau_day} days or {date + timedelta(days=plateau_day.item())}")



    ## Visualize the model outputs

    sns.set(style="darkgrid")

    plt.figure(figsize=(10,6))

    plt.title(f"SIR Model: R = {round(beta/gamma,2)}", fontsize=18)

    sns.lineplot(day,inf, label="Infectious")

    sns.lineplot(day,suc,label="Succeptible")

    sns.lineplot(day,rec, label="Recovered")

    

    plt.legend()

    plt.xlabel("Time (in days)")

    plt.ylabel("Fraction of Population")

    plt.show()
sir_model(beta=0.05, gamma=0.02, days=365)
def sir_model_betalist(I0 = 0.01, betalist = [0.5,0.8], gammalist = [0.15,0.25,0.5], days = 365):

    """

    Function takes Initial Infected Population(I0), list of transmission rates (betalist)

    and list of recovery rates(gammalist) as arguments.

    Plots Infectious population and Infected Population vs time for input parameters

    """

    

    for gamma in gammalist:

        

        # A. Plot Infectious Population

        plt.figure(figsize=(10,6))

        sns.set(style="darkgrid")

        plt.title("SIR Model: Infectious Population", fontsize=18)

        

        # Initialize model parameters

        for beta in betalist:

            N=1

            I=I0

            S=N-I

            gamma=gamma

            R=beta/gamma

            

            # Initialize empty lists

            inf=[]

            day=[]

            

            # Project into the future

            for i in range(days):

                day.append(i)

                inf.append(I)

                new_inf= I*S*beta

                new_rec= I*gamma

                I=I+new_inf-new_rec

                S=S-new_inf

            

            # Create plot objects by gamma and beta

            inf_max=round(np.array(inf).max()*100,1)

            sns.lineplot(day,inf, label=f"Beta: {beta} Gamma: {gamma} R0: {round(R,2)} Peak: {inf_max}%")

            plt.legend()

            

        # Show all plots objects

        plt.show()

        

        # B. Plot Total Infected Population

        plt.figure(figsize=(10,6))

        plt.title("SIR Model: Total Confirmed Cases", fontsize=18)       

        

        # Initialize model parameters

        for beta in betalist:

            N=1

            I=I0

            S=N-I

            C=I

            gamma=gamma

            R=beta/gamma

            

            # Initialize empty lists

            day=[]

            conf=[]



            # Project into the future            

            for i in range(days):

                day.append(i)

                conf.append(C)



                new_inf= I*S*beta

                new_rec= I*gamma

                I=I+new_inf-new_rec

                S=S-new_inf

                C=C+new_inf



            # Create plot objects by gamma and beta

            conf_max=round(np.array(conf).max()*100,1)

            sns.lineplot(day,conf, label=f"Beta: {beta} Gamma: {gamma} R0: {round(R,2)} Total :{conf_max}%")

            plt.legend()

            

        # Show all plots objects            

        plt.show()
sir_model_betalist(I0=0.01,betalist=[0.01, 0.02, 0.04, 0.08, 0.12], gammalist=[0.02, 0.04])
# date = date of the most recent subwave of covid19 to project into the future

# date format yyyy-mm-dd, e.g., '2020-07-04'



def plot_country(country, date): 

    temp = full_grouped[full_grouped['Country/Region']==country]

    temp['recent_wave'] = np.where(temp['Date'] >= date,1,0)



    fig = px.line(temp, x='Date', y='Confirmed', color='recent_wave', \

                  title = 'Infections for ' + str(country), height=600)      

    fig.show()

    

    fig = px.line(temp, x='Date', y='Recovered', color='recent_wave', \

              title = 'Recovered Patients ' + str(country), height=600)      

    fig.show()

    

    return country, date
country, date = plot_country('US', '2020-07-04')
# Calibrate model



def estimate_sir_param(country, date):

    

    # Assume everyone is at risk

    # Identify the maximum population and the latest date in the time series for the country

    population  = full_grouped[full_grouped['Country/Region']==country]["Population"].max()

    latest_date = full_grouped[full_grouped['Country/Region']==country]["Date"].max()

    

    time_series_length = (latest_date - datetime.strptime(date,'%Y-%m-%d')).days + 1



    temp = full_grouped[full_grouped['Country/Region']==country]

    temp['recent_wave'] = np.where(temp['Date'] >= date,1,0)

    

    # Initialize Numpy arrays for total population (the maximum population), 

    # susceptible population (empty), and change in time (i.e., 1 day)

    N  = np.array([population] * time_series_length)

    S  = np.array([])

    dt = np.array([1] * (time_series_length-1))



    # Apply the condition N = S+I+(R+D)

    # Filter time-series to those of the recent wave

    I = np.array(temp[temp['recent_wave']==1]['Active'])

    R = np.array(temp[temp['recent_wave']==1]['Recovered'])

    D = np.array(temp[temp['recent_wave']==1]['Deaths'])



    # R includes both Recovered and Death for brevity

    S = N - I - (R + D)



    ## 1. Estimate beta

    

    x = (S * I) / N

    

    # Copy all elements except the last

    x = x[:-1].copy()

    

    # Take the first difference

    dS = np.diff(S)

    y = dS/dt



    # Fit into a linear regression

    results = sm.OLS(y, x, missing='drop').fit()

    beta = results.params

    print(results.summary())

    print('\n')

    print('*'*80)

    print(f"Transmission rate or Beta is: {beta}")

    print('*'*80)

    

    ## 2. Estimate gamma

    

    x = I[:-1].copy()

    dR = np.diff(R+D)

    y = dR/dt



    results = sm.OLS(endog=y, exog=x, missing='drop').fit()

    gamma = results.params

    print (results.summary())

    print('\n')

    print('*'*80)

    print(f"Recovery (and Mortality) rate or Gamma is: {gamma}")

    print('*'*80)

    

    #3. Calculate R



    print('\n')

    print('*'*80)

    print(f"Reproduction number or R is: {-beta/gamma}")

    print('*'*80)

    

    return -beta.astype('float'), gamma.astype('float'), datetime.strptime(date,'%Y-%m-%d').date()

beta, gamma, date = estimate_sir_param(country, date)
sir_model(I0=0.06, beta = beta.item(), gamma = gamma.item(), days=730, date = date)