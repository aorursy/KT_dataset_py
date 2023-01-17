import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline 

!pip install mpld3

import mpld3

mpld3.enable_notebook()
train = pd.read_csv('../input/covid19-global-forecasting-week-2/train.csv')

test = pd.read_csv('../input/covid19-global-forecasting-week-2/test.csv')

submission = pd.read_csv('../input/covid19-global-forecasting-week-2/submission.csv')
train
train["Country_Region"] = [country_name.replace("'","") for country_name in train["Country_Region"]]
LAST_DATE = train.iloc[-1]["Date"]
train[train["Country_Region"]=="Italy"][["ConfirmedCases", "Fatalities", "Date"]].plot(x="Date", figsize=(8, 4), title="Covid-19 cases and fatalities in Italy");
train.groupby("Date").sum()[["ConfirmedCases", "Fatalities"]].plot(figsize=(8, 4), title="Covid-19 total cases and fatalities (world)");
print("Countries with no fatalities as of " + LAST_DATE)

print(*train.groupby("Country_Region").sum()[train.groupby("Country_Region").sum()["Fatalities"] == 0].index.tolist(), sep=", ")
train[train["Date"] == LAST_DATE].sort_values("Fatalities", ascending=False)[["Country_Region", "ConfirmedCases", "Fatalities"]].head(10)
tmp = train[train["Date"] == LAST_DATE].copy()

tmp["CaseFatalityRate"] = tmp["Fatalities"] / tmp["ConfirmedCases"] * 100  # CFR here is Fatalities/ConfirmedCases * 100 (so that it's in percent)

print("Mean CFR (%):", tmp["CaseFatalityRate"].mean())



heights = tmp[tmp["Fatalities"] >= 100].sort_values("CaseFatalityRate", ascending=False)["CaseFatalityRate"].values

bars = tmp[tmp["Fatalities"] >= 100].sort_values("CaseFatalityRate", ascending=False)["Country_Region"].values

y_pos = np.arange(len(bars))



plt.figure(figsize=(11,4))

plt.bar(y_pos, heights, width=0.5)

 

plt.xticks(y_pos, bars, size="small")

plt.yticks(np.arange(0.0, 11.0, 1.0))

plt.title("Preliminary Case Fatality Rate in Percent by Country")



plt.show();
from scipy.integrate import odeint # a lot of the code for SIR from https://scipython.com/book/chapter-8-scipy/additional-examples/the-sir-epidemic-model/
# The SIR model differential equations.

def deriv(y, t, N, beta, gamma):

    S, I, R = y

    dSdt = -beta * S * I / N

    dIdt = beta * S * I / N - gamma * I

    dRdt = gamma * I

    return dSdt, dIdt, dRdt
def SIR_model(N, D, R_0, CaseFatalityRate, max_days):

    '''

    N: total population

    D, R_0, CaseFatalityRate: see texts above

    '''

    I0, R0 = 1, 0  # Initial number of infected and recovered individuals (1 infected, 0 recovered) [this R0 has nothing to do with the basic reproduction number R0]

    S0 = N - I0 - R0 # Initial number of susceptible (everyone else)



    gamma = 1.0 / D  # see texts above

    beta = R_0 * gamma  # see texts above

    alpha = CaseFatalityRate



    t = np.linspace(0, max_days, max_days) # Grid of time points (in days)



    # Initial conditions vector

    y0 = S0, I0, R0

    # Integrate the SIR equations over the time grid, t.

    ret = odeint(deriv, y0, t, args=(N, beta, gamma))

    S, I, R = ret.T



    # Adding deaths (see text above)

    X = np.zeros(max_days)

    for day in range(13, max_days):

        X[day] = sum(I[:day-13])

    X = alpha * beta * X





    # Plot the data on three separate curves for S(t), I(t) and R(t)

    f, ax = plt.subplots(1,1,figsize=(10,4))

    ax.plot(t, S, 'b', alpha=0.7, linewidth=2, label='Susceptible')

    ax.plot(t, I, 'y', alpha=0.7, linewidth=2, label='Infected')

    ax.plot(t, X, 'r', alpha=0.7, linewidth=2, label='Dead')

    ax.plot(t, R, 'g', alpha=0.7, linewidth=2, label='Recovered')



    ax.set_xlabel('Time (days)')

    ax.title.set_text('SIR-Model. Total Population: ' + str(N) + ", Days Infectious: " + str(D) + ", R_0: " + str(R_0) + ", CFR: " + str(CaseFatalityRate*100) + "%")

    # ax.set_ylabel('Number (1000s)')

    # ax.set_ylim(0,1.2)

    ax.yaxis.set_tick_params(length=0)

    ax.xaxis.set_tick_params(length=0)

    ax.grid(b=True, which='major', c='w', lw=2, ls='-')

    legend = ax.legend()

    legend.get_frame().set_alpha(0.5)

    for spine in ('top', 'right', 'bottom', 'left'):

        ax.spines[spine].set_visible(False)

    plt.show();
SIR_model(N=1_000_000, D=14.0, R_0=2.0, CaseFatalityRate=0.05, max_days=360)
def SIR_model_with_lockdown(N, D, R_0, CaseFatalityRate, max_days, L, R_0_2):

    '''

    N: total population

    D, R_0, CaseFatalityRate, ...: see texts above

    '''

    # BEFORE LOCKDOWN (same code as first model)

    I0, R0 = 1, 0  # Initial number of infected and recovered individuals (1 infected, 0 recovered) [this R0 has nothing to do with the basic reproduction number R0]

    S0 = N - I0 - R0 # Initial number of susceptible (everyone else)



    gamma = 1.0 / D  # see texts above

    beta = R_0 * gamma  # see texts above

    alpha = CaseFatalityRate



    t = np.linspace(0, L, L)  # Grid of time points (in days)

    

    # Initial conditions vector

    y0 = S0, I0, R0

    # Integrate the SIR equations over the time grid, t.

    ret = odeint(deriv, y0, t, args=(N, beta, gamma))

    S, I, R = ret.T

    

    

    # AFTER LOCKDOWN

    I0_2, R0_2, S0_2 = I[-1], R[-1], S[-1]  # beginning of lockdown -> starting Infected/Susceptible/Recovered numbers are the numbers at the end of no-lockdown period



    gamma = 1.0 / D  # same after lockdown

    beta_2 = R_0_2 * gamma

    alpha = CaseFatalityRate  # same after lockdown



    t_2 = np.linspace(0, max_days - L + 1, max_days - L + 1)

    

    # Initial conditions vector

    y0_2 = S0_2, I0_2, R0_2

    # Integrate the SIR equations over the time grid, t.

    ret_2 = odeint(deriv, y0_2, t_2, args=(N, beta_2, gamma))

    S_2, I_2, R_2 = ret_2.T



    

    # COMBINING PERIODS

    S_full = np.concatenate((S, S_2[1:]))

    I_full = np.concatenate((I, I_2[1:]))

    R_full = np.concatenate((R, R_2[1:]))

    t_full = np.linspace(0, max_days, max_days)

    

    # Adding deaths

    X = np.zeros(max_days)

    for day in range(13, max_days):

        for valid_day in range(day-13):

            if valid_day < L:

                X[day] += alpha * beta * I_full[valid_day]

            else:

                X[day] += alpha * beta_2 * I_full[valid_day]



    



    # Plot the data on three separate curves for S(t), I(t) and R(t)

    f, ax = plt.subplots(1,1,figsize=(10,4))

    ax.plot(t_full, S_full, 'b', alpha=0.7, linewidth=2, label='Susceptible')

    ax.plot(t_full, I_full, 'y', alpha=0.7, linewidth=2, label='Infected')

    ax.plot(t_full, X, 'r', alpha=0.7, linewidth=2, label='Dead')

    ax.plot(t_full, R_full, 'g', alpha=0.7, linewidth=2, label='Recovered')



    ax.set_xlabel('Time (days)')

    ax.title.set_text('SIR-Model with Lockdown. Total Population: ' + str(N) + 

                      ", Days Infectious: " + str(D) + ", R_0: " + str(R_0) + 

                      ", CFR: " + str(CaseFatalityRate*100) + " R_0_2: " + str(R_0_2) + 

                      ", L: " + str(L) + " days")

    # ax.set_ylabel('Number (1000s)')

    # ax.set_ylim(0,1.2)

    plt.text(L,N/20,'Lockdown')

    plt.plot(L, 0, marker='o', markersize=6, color="red")

    ax.yaxis.set_tick_params(length=0)

    ax.xaxis.set_tick_params(length=0)

    ax.grid(b=True, which='major', c='w', lw=2, ls='-')

    legend = ax.legend()

    legend.get_frame().set_alpha(0.5)

    for spine in ('top', 'right', 'bottom', 'left'):

        ax.spines[spine].set_visible(False)

    plt.show();
SIR_model(N=1_000_000, D=4, R_0=3.0, CaseFatalityRate=0.05, max_days=60)
SIR_model_with_lockdown(N=1_000_000, D=4, R_0=3.0, CaseFatalityRate=0.05, max_days=60, L=30, R_0_2=0.9)
SIR_model_with_lockdown(N=1_000_000, D=4, R_0=3.0, CaseFatalityRate=0.05, max_days=60, L=22, R_0_2=0.9)
SIR_model_with_lockdown(N=1_000_000, D=4, R_0=3.0, CaseFatalityRate=0.05, max_days=60, L=15, R_0_2=0.9)
# SIR-Model's Fatality Curve (no plotting etc.):

def SIR_model_with_lockdown_deaths(x, N, D, R_0, CaseFatalityRate, max_days, L, R_0_2):

    # BEFORE LOCKDOWN (same code as first model)

    I0, R0 = 1, 0  # Initial number of infected and recovered individuals (1 infected, 0 recovered) [this R0 has nothing to do with the basic reproduction number R0]

    S0 = N - I0 - R0 # Initial number of susceptible (everyone else)



    gamma = 1.0 / D  # see texts above

    beta = R_0 * gamma  # see texts above

    alpha = CaseFatalityRate



    t = np.linspace(0, L, L)  # Grid of time points (in days)

    

    # Initial conditions vector

    y0 = S0, I0, R0

    # Integrate the SIR equations over the time grid, t.

    ret = odeint(deriv, y0, t, args=(N, beta, gamma))

    S, I, R = ret.T

    

    

    # AFTER LOCKDOWN

    I0_2, R0_2, S0_2 = I[-1], R[-1], S[-1]  # beginning of lockdown -> starting Infected/Susceptible/Recovered numbers are the numbers at the end of no-lockdown period



    gamma = 1.0 / D  # same after lockdown

    beta_2 = R_0_2 * gamma

    alpha = CaseFatalityRate  # same after lockdown



    t_2 = np.linspace(0, max_days - L + 1, max_days - L + 1)

    

    # Initial conditions vector

    y0_2 = S0_2, I0_2, R0_2

    # Integrate the SIR equations over the time grid, t.

    ret_2 = odeint(deriv, y0_2, t_2, args=(N, beta_2, gamma))

    S_2, I_2, R_2 = ret_2.T



    

    # COMBINING PERIODS

    S_full = np.concatenate((S, S_2[1:]))

    I_full = np.concatenate((I, I_2[1:]))

    R_full = np.concatenate((R, R_2[1:]))

    t_full = np.linspace(0, max_days, max_days)

    

    # Adding deaths

    X = np.zeros(max_days)

    for day in range(13, max_days):

        for valid_day in range(day-13):

            if valid_day < L:

                X[day] += alpha * beta * I_full[valid_day]

            else:

                X[day] += alpha * beta_2 * I_full[valid_day]

    return X[x]
!pip install lmfit

from lmfit import Model
# Load countries data file (from https://www.kaggle.com/saga21/covid-global-forecast-sir-model-ml-regressions)

world_population = pd.read_csv("/kaggle/input/population-by-country-2020/population_by_country_2020.csv")



# Select desired columns and rename some of them

world_population = world_population[['Country (or dependency)', 'Population (2020)', 'Density (P/Km²)', 'Land Area (Km²)', 'Med. Age', 'Urban Pop %']]

world_population.columns = ['Country (or dependency)', 'Population (2020)', 'Density', 'Land Area', 'Med Age', 'Urban Pop']



# Replace United States by US

world_population.loc[world_population['Country (or dependency)']=='United States', 'Country (or dependency)'] = 'US'



# Remove the % character from Urban Pop values

world_population['Urban Pop'] = world_population['Urban Pop'].str.rstrip('%')



# Replace Urban Pop and Med Age "N.A" by their respective modes, then transform to int

world_population.loc[world_population['Urban Pop']=='N.A.', 'Urban Pop'] = int(world_population.loc[world_population['Urban Pop']!='N.A.', 'Urban Pop'].mode()[0])

world_population['Urban Pop'] = world_population['Urban Pop'].astype('int16')

world_population.loc[world_population['Med Age']=='N.A.', 'Med Age'] = int(world_population.loc[world_population['Med Age']!='N.A.', 'Med Age'].mode()[0])

world_population['Med Age'] = world_population['Med Age'].astype('int16')
lockdown_dates = {"Italy": "2020-03-10", "Spain": "2020-03-15", "Germany": "2020-03-23"}



def fit_SIR(country_name, lockdown_date=None, region_name=None):

    """

    y_data: the fatalities data of one country/region (array)

    population: total population of country

    lockdown_date: format YYYY-MM-DD

    """

    if lockdown_date is None:

        lockdown_date = lockdown_dates[country_name]



    if region_name:

        y_data = train[(train["Country_Region"] == country_name) & (train["Region"] == region_name)].Fatalities.values

    else:

        if len(train["Country_Region"] == country_name) > len(train["Country_Region"] == "Germany"):  # country with several regions and no region provided

            y_data = train[(train["Country_Region"] == country_name) & (train["Region"].isnull())].Fatalities.values

        else:

            y_data = train[train["Country_Region"] == country_name].Fatalities.values

        

    max_days = len(train.groupby("Date").sum().index) # constant for all countries



    # country specific values

    N = world_population.loc[world_population['Country (or dependency)'] == country_name]["Population (2020)"].values[0]

    L = train.groupby("Date").sum().index.tolist().index(lockdown_date)  # index of the lockdown date



    # x_data is just [0, 1, ..., max_days] array

    x_data = np.linspace(0, max_days - 1, max_days, dtype=int)

    

    # curve fitting from here

    mod = Model(SIR_model_with_lockdown_deaths)



    # initial values and bounds

    mod.set_param_hint('N', value=N)

    mod.set_param_hint('max_days', value=max_days)

    mod.set_param_hint('L', value=L)

    mod.set_param_hint('D', value=10, min=4, max=25)

    mod.set_param_hint('CaseFatalityRate', value=0.01, min=0.0001, max=0.1)

    mod.set_param_hint('R_0', value=2.0, min=0.1, max=5.0)

    mod.set_param_hint('R_0_2', value=2.0, min=0.1, max=5.0)



    params = mod.make_params()



    # fixing constant parameters

    params['N'].vary = False

    params['max_days'].vary = False

    params['L'].vary = False



    result = mod.fit(y_data, params, x=x_data, method="least_squares")

    

    return result, country_name



def fitted_plot(result, country_name, region_name=None):

    if region_name:

        y_data = train[(train["Country_Region"] == country_name) & (train["Region"] == region_name)].Fatalities.values

    else:

        if len(train["Country_Region"] == country_name) > len(train["Country_Region"] == "Germany"):  # country with several regions and no region provided

            y_data = train[(train["Country_Region"] == country_name) & (train["Region"].isnull())].Fatalities.values

        else:

            y_data = train[train["Country_Region"] == country_name].Fatalities.values



    max_days = len(train.groupby("Date").sum().index)

    x_data = np.linspace(0, max_days - 1, max_days, dtype=int)

    x_ticks = train[train["Country_Region"] == "Germany"].Date.values  # same for all countries

    

    plt.figure(figsize=(10,5))

    

    real_data, = plt.plot(x_data, y_data, 'bo', label="real data")

    SIR_fit = plt.plot(x_data, result.best_fit, 'r-', label="SIR model")

    

    plt.xlabel("Day")

    plt.xticks(x_data[::10], x_ticks[::10])

    plt.ylabel("Fatalities")

    plt.title("Real Data vs SIR-Model in " + country_name)

    plt.legend(numpoints=1, loc=2, frameon=None)

    plt.show()
result, _ = fit_SIR("Italy")

print(result.fit_report())

fitted_plot(result, "Italy")
result, _ = fit_SIR("Spain")

print(result.fit_report())

fitted_plot(result, "Spain")
result, _ = fit_SIR("Germany")

print(result.fit_report())

fitted_plot(result, "Germany")
# extended SIR model differential equations. Beta is now a function.

def extended_deriv(y, t, N, beta, gamma):

    S, I, R = y

    dSdt = -beta(t) * S * I / N

    dIdt = beta(t) * S * I / N - gamma * I

    dRdt = gamma * I

    return dSdt, dIdt, dRdt
def extended_SIR(N, D, max_days, CFR_OPT, CFR_scaling_factor, R_0, **R_0_kwargs):

    '''

    R_0: callable

    '''

    I0, R0 = 1, 0  # Initial number of infected and recovered individuals (1 infected, 0 recovered) [this R0 has nothing to do with the basic reproduction number R0]

    S0 = N - I0 - R0 # Initial number of susceptible (everyone else)



    gamma = 1.0 / D  # see texts above



    def beta(t):

        return R_0(t, **R_0_kwargs) * gamma



    t = np.linspace(0, max_days, max_days)  # Grid of time points (in days)

    

    # Initial conditions vector

    y0 = S0, I0, R0

    # Integrate the SIR equations over the time grid, t.

    ret = odeint(extended_deriv, y0, t, args=(N, beta, gamma))

    S, I, R = ret.T



    def CFR(t):

        return CFR_OPT + CFR_scaling_factor * (I[t] / N)



    # Adding deaths

    X = np.zeros(max_days)

    for day in range(13, max_days):

        for valid_day in range(day-13):

            X[day] += CFR(valid_day) * beta(valid_day) * I[valid_day]



    return t, S, I, R, X, [R_0(t, **R_0_kwargs) for t in range(max_days)], N, [CFR(t) for t in range(max_days)]
def plot_extended_SIR(t, S, I, R, X, R_0, N, CFR):

    # Plot the data on three separate curves for S(t), I(t) and R(t)

    f, ax = plt.subplots(1,1,figsize=(10,4))

    ax.plot(t, S, 'b', alpha=0.7, linewidth=2, label='Susceptible')

    ax.plot(t, I, 'y', alpha=0.7, linewidth=2, label='Infected')

    ax.plot(t, X, 'r', alpha=0.7, linewidth=2, label='Dead')

    ax.plot(t, R, 'g', alpha=0.7, linewidth=2, label='Recovered')



    ax.set_xlabel('Time (days)')

    ax.title.set_text('SIR-Model with varying R_0 and CFR')

    # ax.set_ylabel('Number (1000s)')

    # ax.set_ylim(0,1.2)

    ax.yaxis.set_tick_params(length=0)

    ax.xaxis.set_tick_params(length=0)

    ax.grid(b=True, which='major', c='w', lw=2, ls='-')

    legend = ax.legend()

    legend.get_frame().set_alpha(0.5)

    for spine in ('top', 'right', 'bottom', 'left'):

        ax.spines[spine].set_visible(False)

    plt.show();

    

    

    # plt.figure(figsize=(10,4))

    

    f = plt.figure(figsize=(10,4))

    

    # sp1

    ax1 = f.add_subplot(121)

    ax1.plot(t, R_0, 'b--', alpha=0.7, linewidth=2, label='R_0')

    

    ax1.set_xlabel('Time (days)')

    ax1.title.set_text('R_0 over time')

    # ax.set_ylabel('Number (1000s)')

    # ax.set_ylim(0,1.2)

    ax1.yaxis.set_tick_params(length=0)

    ax1.xaxis.set_tick_params(length=0)

    ax1.grid(b=True, which='major', c='w', lw=2, ls='-')

    legend = ax1.legend()

    legend.get_frame().set_alpha(0.5)

    for spine in ('top', 'right', 'bottom', 'left'):

        ax.spines[spine].set_visible(False)



    # sp2

    ax2 = f.add_subplot(122)

    ax2.plot(t, CFR, 'r--', alpha=0.7, linewidth=2, label='CFR')

    

    ax2.set_xlabel('Time (days)')

    ax2.title.set_text('CFR over time')

    # ax.set_ylabel('Number (1000s)')

    # ax.set_ylim(0,1.2)

    ax2.yaxis.set_tick_params(length=0)

    ax2.xaxis.set_tick_params(length=0)

    ax2.grid(b=True, which='major', c='w', lw=2, ls='-')

    legend = ax2.legend()

    legend.get_frame().set_alpha(0.5)

    for spine in ('top', 'right', 'bottom', 'left'):

        ax.spines[spine].set_visible(False)



    plt.show();
N = 1_000

D = 4

max_days = 100



I0, R0 = 1, 0

S0 = N - I0 - R0

s = CFR_scaling_factor = 0.1

CFR_OPT = 0.02  # noone in hospital -> only 2% die



def new_R0(t, a, b, c):

    return a / (1 + (t/c)**b)





plot_extended_SIR(*extended_SIR(N, D, max_days, CFR_OPT, CFR_scaling_factor, new_R0, a=3.0, b=1.5, c=50))
def fit_extended_SIR(country_name, R_0_function, region_name=None, fit_method="least_squares", **R_0_kwargs):



    if region_name:

        y_data = train[(train["Country_Region"] == country_name) & (train["Region"] == region_name)].Fatalities.values

    else:

        if len(train["Country_Region"] == country_name) > len(train["Country_Region"] == "Germany"):  # country with several regions and no region provided

            y_data = train[(train["Country_Region"] == country_name) & (train["Region"].isnull())].Fatalities.values

        else:

            y_data = train[train["Country_Region"] == country_name].Fatalities.values

        

    max_days = len(train.groupby("Date").sum().index) # constant for all countries

 

    # country specific values

    N = world_population.loc[world_population['Country (or dependency)'] == country_name]["Population (2020)"].values[0]



    # x_data is just [0, 1, ..., max_days] array

    x_data = np.linspace(0, max_days - 1, max_days, dtype=int)



    # curve fitting from here

    def extended_SIR_deaths(x, N, D, max_days, CFR_OPT, CFR_scaling_factor, **R_0_kwargs):

        t_, S_, I_, R_, X, R_0_, N_, CFR_ = extended_SIR(N, D, max_days, CFR_OPT, CFR_scaling_factor, R_0=R_0_function, **R_0_kwargs)

        return X[x]



    mod = Model(extended_SIR_deaths)



    # initial values and bounds

    mod.set_param_hint('N', value=N, vary=False)

    mod.set_param_hint('max_days', value=max_days, vary=False)



    mod.set_param_hint('D', value=10, min=4, max=25)

    mod.set_param_hint('CFR_OPT', value=0.01, min=0.0001, max=0.1)

    mod.set_param_hint('CFR_scaling_factor', value=0.1, min=0.0001, max=1.0)

    if R_0_kwargs:

        for arg in R_0_kwargs:

            mod.set_param_hint(arg, value=R_0_kwargs[arg])



    params = mod.make_params()

    # print(params)

    result = mod.fit(y_data, params, method=fit_method, x=x_data)

    

    # fetch some result parameters

    CFR_OPT = result.params["CFR_OPT"].value

    CFR_scaling_factor = result.params["CFR_scaling_factor"].value

    R_0_result_params = {}

    for val in R_0_kwargs:

        R_0_result_params[val] = result.params[val].value



    

    # return result, country_name

    return result, country_name, N, D, max_days, CFR_OPT, CFR_scaling_factor, R_0_function, R_0_result_params



def fitted_plot(result, country_name, region_name=None):

    if region_name:

        y_data = train[(train["Country_Region"] == country_name) & (train["Region"] == region_name)].Fatalities.values

    else:

        if len(train["Country_Region"] == country_name) > len(train["Country_Region"] == "Germany"):  # country with several regions and no region provided

            y_data = train[(train["Country_Region"] == country_name) & (train["Region"].isnull())].Fatalities.values

        else:

            y_data = train[train["Country_Region"] == country_name].Fatalities.values



    max_days = len(train.groupby("Date").sum().index)

    x_data = np.linspace(0, max_days - 1, max_days, dtype=int)

    x_ticks = train[train["Country_Region"] == "Germany"].Date.values  # same for all countries

    

    plt.figure(figsize=(10,5))

    

    real_data, = plt.plot(x_data, y_data, 'bo', label="real data")

    SIR_fit = plt.plot(x_data, result.best_fit, 'r-', label="SIR model")

    

    plt.xlabel("Day")

    plt.xticks(x_data[::10], x_ticks[::10])

    plt.ylabel("Fatalities")

    plt.title("Real Data vs SIR-Model in " + country_name)

    plt.legend(numpoints=1, loc=2, frameon=None)

    plt.show()
def new_R0(t, a, b, c):

    return a / (1 + (t/c)**b)



result, country_name, N, D, max_days, CFR_OPT, CFR_scaling_factor, R_0_function, R_0_result_params = fit_extended_SIR("Italy", new_R0, region_name=None, fit_method="least_squares", a=3.0, b=1.5, c=50)

print(result.fit_report())

fitted_plot(result, "Italy");

plot_extended_SIR(*extended_SIR(N, D, max_days, CFR_OPT, CFR_scaling_factor, R_0_function, **R_0_result_params))
def logistic_R_0(t, R_0_start, k, x0, R_0_end):

    return (R_0_start-R_0_end) / (1 + np.exp(-k*(-t+x0))) + R_0_end
x = np.linspace(0, 100, 100)

plt.title("logistic R_0: initial R_0 2.0, final R_0 1.4, x0=50, varying k-values")

plt.plot(x, logistic_R_0(x, R_0_start=2, k=1.0, x0=50, R_0_end=1.4), label="k=1.0")

plt.plot(x, logistic_R_0(x, R_0_start=2, k=0.5, x0=50, R_0_end=1.4), label="k=0.5")

plt.plot(x, logistic_R_0(x, R_0_start=2, k=0.1, x0=50, R_0_end=1.4), label="k=0.1")

plt.legend()

plt.show();
def extended_deriv_SEIR(y, t, N, beta, gamma, sigma):

    S, E, I, R = y

    dSdt = -beta(t) * S * I / N  # same as before

    dEdt = beta(t) * S * I / N - sigma * E  # changed

    dIdt = sigma * E - gamma * I  # changed

    dRdt = gamma * I  # same as before

    return dSdt, dEdt, dIdt, dRdt
def extended_SEIR(N, D, max_days, CFR_OPT, CFR_scaling_factor, R_0, **R_0_kwargs):

    '''

    R_0: callable

    '''

    I0, R0, E0 = 0, 0, 1  # changed: one exposed at the beginning

    S0 = N - I0 - R0 - E0



    gamma = 1.0 / D

    sigma = 1.0 / 3.0  # changed: 3 days until infectious



    def beta(t):

        return R_0(t, **R_0_kwargs) * gamma



    t = np.linspace(0, max_days, max_days)



    # Initial conditions vector

    y0 = S0, E0, I0, R0

    # Integrate the SIR equations over the time grid, t.

    ret = odeint(extended_deriv_SEIR, y0, t, args=(N, beta, gamma, sigma))

    S, E, I, R = ret.T



    def CFR(t):

        if t < 7:

            return CFR_OPT

        else:

            return CFR_OPT + CFR_scaling_factor * (I[t - 7] / N)  # changed: implemented 7-day shift until patients get to hospital



    # Adding deaths

    X = np.zeros(max_days)

    for day in range(16, max_days):  # changed: changed to 19 days until death minus 3 for the three "exposed days"

        for valid_day in range(day-16):

            X[day] += CFR(valid_day) * beta(valid_day) * I[valid_day]



    return t, S, E, I, R, X, [R_0(t, **R_0_kwargs) for t in range(max_days)], N, [CFR(t) for t in range(max_days)]
def plot_extended_SEIR(t, S, E, I, R, X, R_0, N, CFR, x_ticks=None):

    # Plot the data on three separate curves for S(t), I(t) and R(t)

    f, ax = plt.subplots(1,1,figsize=(10,4))

    ax.plot(t, S, 'b', alpha=0.7, linewidth=2, label='Susceptible')

    ax.plot(t, E, 'y--', alpha=0.7, linewidth=2, label='Exposed')

    ax.plot(t, I, 'y', alpha=0.7, linewidth=2, label='Infected')

    ax.plot(t, X, 'r', alpha=0.7, linewidth=2, label='Dead')

    ax.plot(t, R, 'g', alpha=0.7, linewidth=2, label='Recovered')



    ax.set_xlabel('Time (days)')

    ax.title.set_text('SEIR-Model with varying R_0 and CFR')

    # ax.set_ylabel('Number (1000s)')

    # ax.set_ylim(0,1.2)

    ax.yaxis.set_tick_params(length=0)

    ax.xaxis.set_tick_params(length=0)



    if x_ticks is not None:

        ax.set_xticks(t[::21])

        ax.set_xticklabels(x_ticks[::21])    



    ax.grid(b=True, which='major', c='w', lw=2, ls='-')

    legend = ax.legend()

    legend.get_frame().set_alpha(0.5)

    for spine in ('top', 'right', 'bottom', 'left'):

        ax.spines[spine].set_visible(False)

    plt.show();

    

    f = plt.figure(figsize=(10,4))

    # sp1

    ax1 = f.add_subplot(121)

    ax1.plot(t, R_0, 'b--', alpha=0.7, linewidth=2, label='R_0')

 

    ax1.set_xlabel('Time (days)')

    ax1.title.set_text('R_0 over time')

    # ax.set_ylabel('Number (1000s)')

    # ax.set_ylim(0,1.2)

    ax1.yaxis.set_tick_params(length=0)

    ax1.xaxis.set_tick_params(length=0)

    if x_ticks is not None:

        ax1.set_xticks(t[::35])

        ax1.set_xticklabels(x_ticks[::35])    

    ax1.grid(b=True, which='major', c='w', lw=2, ls='-')

    legend = ax1.legend()

    legend.get_frame().set_alpha(0.5)

    for spine in ('top', 'right', 'bottom', 'left'):

        ax.spines[spine].set_visible(False)



    # sp2

    ax2 = f.add_subplot(122)

    ax2.plot(t, CFR, 'r--', alpha=0.7, linewidth=2, label='CFR')

    

    ax2.set_xlabel('Time (days)')

    ax2.title.set_text('CFR over time')

    # ax.set_ylabel('Number (1000s)')

    # ax.set_ylim(0,1.2)

    ax2.yaxis.set_tick_params(length=0)

    ax2.xaxis.set_tick_params(length=0)

    if x_ticks is not None:

        ax2.set_xticks(t[::70])

        ax2.set_xticklabels(x_ticks[::70])

    ax2.grid(b=True, which='major', c='w', lw=2, ls='-')

    legend = ax2.legend()

    legend.get_frame().set_alpha(0.5)

    for spine in ('top', 'right', 'bottom', 'left'):

        ax.spines[spine].set_visible(False)



    plt.show();
N = 80_000_000

D = 9

max_days = 400



s = CFR_scaling_factor = 0.1  # everyone infected at same time -> 12% instead of 2% die

CFR_OPT = 0.02  # noone in hospital -> only 2% die
plot_extended_SEIR(*extended_SEIR(N, D, max_days, CFR_OPT, CFR_scaling_factor, logistic_R_0, R_0_start=2.5, k=0.3, x0=200, R_0_end=1.9))
plot_extended_SEIR(*extended_SEIR(N, D, max_days, CFR_OPT, CFR_scaling_factor, logistic_R_0, R_0_start=2.5, k=0.3, x0=170, R_0_end=0.2))
def fit_extended_SEIR(country_name, missing_days=0, region_name=None, fit_method="least_squares", **R_0_kwargs):



    if region_name is not None:

        y_data = train[(train["Country_Region"] == country_name) & (train["Province_State"] == region_name)].Fatalities.values

    else:

        if len(train["Country_Region"] == country_name) > len(train["Country_Region"] == "Germany"):  # country with several regions and no region provided

            # print("ok")

            y_data = train[(train["Country_Region"] == country_name) & (train["Province_State"].isnull())].Fatalities.values

        else:

            y_data = train[train["Country_Region"] == country_name].Fatalities.values

        

    max_days = len(train.groupby("Date").sum().index) + missing_days # constant for all countries

    y_data = np.concatenate((np.zeros(missing_days), y_data))

    # country specific values

    N = world_population.loc[world_population['Country (or dependency)'] == country_name]["Population (2020)"].values[0]



    # x_data is just [0, 1, ..., max_days] array

    x_data = np.linspace(0, max_days - 1, max_days, dtype=int)



    # curve fitting from here

    def extended_SEIR_deaths(x, N, D, CFR_OPT, CFR_scaling_factor, R_0_delta, **R_0_kwargs):

        # print(x)

        t_, S_, E_, I_, R_, X, R_0_, N_, CFR_ = extended_SEIR(N, D, max_days, CFR_OPT, CFR_scaling_factor, R_0=logistic_R_0, **R_0_kwargs)

        # return np.concatenate((np.zeros(int(outbreak)), X))

        return X[x]



    mod = Model(extended_SEIR_deaths)



    # initial values and bounds

    mod.set_param_hint('N', value=N, vary=False)

    # mod.set_param_hint('max_days', value=max_days, vary=False)

    mod.set_param_hint('D', value=9, vary=False)



    mod.set_param_hint('CFR_OPT', value=0.01, min=0.0001, max=0.1)

    mod.set_param_hint('CFR_scaling_factor', value=0.1, min=0.0001, max=1.0)

    

    mod.set_param_hint('R_0_start', value=2.5, min=1.0, max=5.0)

    mod.set_param_hint('R_0_end', value=0.7, min=0.01, max=5.0)

    # mod.set_param_hint('outbreak', value=20, min=0, max=150)

    mod.set_param_hint('x0', value=30.0, min=0.0, max=float(max_days))

    mod.set_param_hint('k', value=0.1, min=0.01, max=5.0)

    '''

    if R_0_kwargs:

        for arg in R_0_kwargs:

            mod.set_param_hint(arg, value=R_0_kwargs[arg])

    '''



    params = mod.make_params()

    params.add('R_0_delta', value=1.0, min=0.0, expr="R_0_start - R_0_end")  # add constraint R_0_start >= R_0_end

    # print(params)

    result = mod.fit(y_data, params, method=fit_method, x=x_data)



    # fetch some result parameters

    CFR_OPT = result.params["CFR_OPT"].value

    CFR_scaling_factor = result.params["CFR_scaling_factor"].value

    R_0_result_params = {}

    for val in R_0_kwargs:

        R_0_result_params[val] = result.params[val].value



    return result, country_name, y_data, N, D, max_days, CFR_OPT, CFR_scaling_factor, R_0_result_params





def extended_SEIR_fitted_plot(result, country_name, y_data):

#    max_days = len(train.groupby("Date").sum().index)

#   x_data = np.linspace(0, max_days - 1, max_days, dtype=int)

#    x_ticks = train[train["Country_Region"] == "Germany"].Date.values  # same for all countries

    np.datetime64(LAST_DATE)



    # x_ticks = pd.date_range(end=LAST_DATE, periods=len(y_data))

    x_ticks = np.arange(np.datetime64(LAST_DATE) - np.timedelta64(len(y_data),'D'), np.datetime64(LAST_DATE), step=np.timedelta64(1,'D'))

    x_ticks = [np.datetime_as_string(t, unit='D') for t in x_ticks]



    plt.figure(figsize=(10,5))

    x_data = np.linspace(0, len(y_data), len(y_data))

    real_data, = plt.plot(x_data, y_data, 'bo', label="real data")

    SIR_fit = plt.plot(x_data, result.best_fit, 'r-', label="SIR model")

    

    plt.xlabel("Day")

    plt.xticks(x_data[::30], x_ticks[::30])

    # print(x_ticks)

    plt.ylabel("Fatalities")

    plt.title("Real Data vs SIR-Model in " + country_name)

    plt.legend(numpoints=1, loc=2, frameon=None)

    plt.show()
result, country_name, y_data, N, D, max_days, CFR_OPT, CFR_scaling_factor, R_0_result_params = fit_extended_SEIR("Italy", missing_days=30, fit_method="least_squares", 

                                                                                                                 R_0_start=2.5, k=0.3, x0=170, R_0_end=0.2)



print(result.fit_report())

extended_SEIR_fitted_plot(result, "Italy", y_data);



future = 100

x_ticks = np.arange(np.datetime64(LAST_DATE) - np.timedelta64(len(y_data),'D'), np.datetime64(LAST_DATE) + np.timedelta64(future, 'D'), step=np.timedelta64(1,'D'))

x_ticks = [pd.to_datetime(str(t)).strftime("%m/%d") for t in x_ticks]

plot_extended_SEIR(*extended_SEIR(N, D, max_days + future, CFR_OPT, CFR_scaling_factor, logistic_R_0, **R_0_result_params), x_ticks=x_ticks)
y_data = train[(train["Country_Region"] == "Italy") & (train["Province_State"].isnull())].Fatalities.values



x_orig = np.linspace(100, len(y_data)+100, len(y_data))

# print(x_orig.shape)

plt.plot(x_orig, y_data)



zero_part = np.zeros(100)

y_2 = np.concatenate((zero_part, y_data))

noise = np.random.normal(0,1,y_2.shape)

plt.plot(y_2 + noise)

# 0 is the mean of the normal distribution you are choosing from

# 1 is the standard deviation of the normal distribution

# 100 is the number of elements you get in array noise





plt.show();