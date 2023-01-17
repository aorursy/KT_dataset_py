import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from matplotlib.dates import DateFormatter

import matplotlib.dates as mdates

from scipy import interpolate, integrate

from scipy import optimize
#df = pd.read_csv('../input/covid19saopaulosp/covidsp.csv',sep=',',index_col=0, parse_dates=True)

df = pd.read_csv('../input/covid19saopaulosp/covidsp.csv',sep=',')

df['Date'] = pd.to_datetime(df['Date'])  # Converts date strings to timestamp

df.set_index('Date', inplace = True)
print(df.info(verbose=True) ,"\n")



print ("Dados dos últimos 7 dias absolutos" , "\n")

df.tail(7)
daily_data = pd.DataFrame()

daily_data['DeathPerDay']=df['DeathsSivep'].diff()

daily_data['DeathTotal']=df['DeathsSivep']

daily_data['ConfirmedTotal']=df['ConfirmedCases']

daily_data['ConfirmedPerDay']=df['ConfirmedCases'].diff()

daily_data['SuspectedPerDay']=df['Suspects'].diff()

daily_data['DeathPerDayAvg']=df['DeathsSivep'].diff(periods=7)/7

daily_data['ConfirmedPerDayAvg']=df['ConfirmedCases'].diff(periods=7)/7

daily_data['CtiUsage']=df['CtiUsage']

daily_data['Letality'] = 100.0*(df['DeathsSivep']/df['ConfirmedCases'])

daily_data['LetalityLast30days'] = 100.0*(df['DeathsSivep'].diff(periods=30)/df['ConfirmedCases'].diff(periods=30))

daily_data['LetalityShiftConfirmed'] = 100.0*(df['DeathsSivep']/df['ConfirmedCases'].shift(13))

daily_data['ReceoveredEstimated'] = df['Recovered']

daily_data['Active'] = daily_data['ConfirmedTotal'] - daily_data['DeathTotal'] - daily_data['ReceoveredEstimated']  # Active Cases

daily_data['ActiveEstOnConfirmed'] = daily_data['ConfirmedTotal'] - daily_data['ConfirmedTotal'].shift(13)





death_rate = 0.5 #taxa de morte do coronavirus em porcentagem

population_sao_paulo = 12252023

daily_data['EstimatedPercentPopulation'] = 100*(df['DeathsSivep']/(death_rate/100.0))/(population_sao_paulo)

print ("Dados dos últimos dias")

print(daily_data.info(verbose=True))

daily_data.tail(10)
def logistic_model(x, a, b, c):

    """Uses Logistic Model (https://en.wikipedia.org/wiki/Logistic_regression) to fit the curve of infected

        individuals to a Logistic Curve f(x, a, b, c) = c / (1 + exp(-(x-b)/a))



        Args:

        - x : (float) Time to evaluate the infected curve

        - a, b, c : (float) Logistic Curve paramters

    """

    return c / (1 + np.exp(-(x - b) / a))
def fit(time_series, model, lbounds, gbounds, guess=None):

    """Fit the real data to a model. Lower and greater bounds must be provided

       in order to fit a problem with many free parameters. An initial guess

       should be provided in order to improve the accuracy of the model.



       Args:

       - values : panda series

       - model : (models.model) A function from models.py library

       - lbounds : (list) List of floats for lower bounds of each free parameter of the model

       - gounds : (list) List of floats for greater bounds of each free parameter of the model

       - guess : (list) List of floats for an initial guess between bounds (default None)

    """



    xdata = time_series.index.values.astype(float)

    xdata = np.arange(0, len(time_series)).astype(float)

    ydata = time_series.to_numpy().astype(float)

    sol = optimize.curve_fit(model, xdata, ydata, p0=guess, bounds=(lbounds, gbounds))[0]  # Fit the curve

    return sol
def format_axis(ax, values, days_range, tmax=None):



    """ Given an matplotlib.pyplot.axis object, it returns the same object formatting the date values on the x-axis



        Parameters:

        - ax (matplotlib.pyplot.axis) The axis to format

        - values (pandas.DataFrame or pandas.Series) The data which appears on the plot

        - days_range (int) Days between x-label ticks

        - tmax (int) Maximum number of days to show, default chooses the lenght of values (Default None)"""



    initial_date = values.index.values[0]  # Select the initial date



    if tmax:

        days_ticks = range(tmax)  # Expand or truncate to tmax if necesary

    else:

        days_ticks = range(len(values))  # Automatically chooses the lenght of values



    # Format the labels to day / month strings

    days_labels = [str(pd.to_datetime(initial_date + np.timedelta64(i, 'D')).day).zfill(2) + '/' +

                   str(pd.to_datetime(initial_date + np.timedelta64(i, 'D')).month).zfill(2) for i in days_ticks]



    ax.set_xticks(days_ticks[::days_range])  # Define the matplotlib xticks

    ax.set_xticklabels(days_labels[::days_range])  # Define the matplotlib xlabels

    return ax



def plot_chart_model(time_series, model,model_parameters,number_days_future,title):

    

    values = time_series.to_numpy().astype(float)

    sol = model_parameters

        

    days = np.arange(0, len(values))  # Defines the days array for the real data plot

    tmin=0

    tmax=len(values)+ number_days_future

    scale=1

    days_range = 14

      

    t = np.arange(tmin, tmax)  # Defines the time array for the model plot



    fig, ax = plt.subplots(figsize=(12, 8), ncols=1)  # Creates the Figure



    ax.plot(days, values / scale, 'k', alpha=0.5, lw=2, marker='x', label='Dados Reais')  # Plot the real data

    ax.plot(t, model(t, *sol) / scale, 'r', alpha=0.5, lw=3, label='Modelo')  # Plot the model data



    ax.legend().get_frame().set_alpha(0.5)  # Style the legend

    ax.grid(b=True, which='major', c='k', lw=0.25, ls='-')  # Style the grid



    ax.set_xlabel('Data')  # Set the x-axis label

    ax.set_ylabel('Numero (' + str(scale) + ')')  # Set the y-axis label



    ax.yaxis.set_tick_params(length=0)  # Style the y-axis labels

    ax.xaxis.set_tick_params(length=0)  # Style the x-axis labels



    ax.set_xlim([tmin, tmax])



    ax = format_axis(ax, time_series, days_range, tmax)



    ax.set_title(title)  # Set the title



    for spine in ('top', 'right', 'bottom', 'left'):

        ax.spines[spine].set_visible(False)  # Style the figure box

    plt.tight_layout()

    plt.show()

    
def sir_model(x, initial_infected, beta, gamma, population, t0, initial_recovered=0, tmax=365, n=1000):



    """"Uses SIR Model (https://en.wikipedia.org/wiki/Compartmental_models_in_epidemiology#The_SIR_model)

        to return the value of the Infected Curve at time x, given the SIR Model parameters.



        Parameters:

        - x : (float) Time to evaluate the infected curve

        - initial_infected : (float) SIR Model I0 parameter. Initial infected subjects

        - beta : (float) Transmission rate

        - gamma : (float) Typical time between contacts

        - population : (float) Susceptible population

        - t0 : (float) Initial time to fix origin

        - initial_recovered : (float) Initial recovered individuals (Default 0)

        - tmax : (float) Time for integrating the differential equations, in days (Default 365)

        - n : (int) Number of time points for integrating the differential equations (Default 1000)"""



    gamma = 1 / gamma

    initial_susceptible = population - initial_infected - initial_recovered  # Everyone who is susceptible to infection

    t = np.linspace(t0, tmax, n)  # Time vector for integrating



    def derivatives(y, _):



        """SIR Model Differential Equations



            Parameters:

            - y : (np.ndarray) Array containing [Susceptible, Infected, Recovered] points

            - _ : (None) Empty parameter for consistency with scipy.integrate.odeint method"""



        s, i, _ = y

        derivative_a = -beta * s * i / population  # dS/dt

        derivative_b = beta * s * i / population - gamma * i  # dI/dt

        derivative_c = gamma * i  # dR / dt

        return derivative_a, derivative_b, derivative_c



    y0 = initial_susceptible, initial_infected, initial_recovered  # Initial conditions vector

    sol = integrate.odeint(derivatives, y0, t)  # Integrate the SIR equations over the time grid, total_time

    infected = sol[:, 1]  # Infected individuals for each day

    interp = interpolate.interp1d(t, infected, fill_value='extrapolate')  # Creates an interpolator with the vectors

    resp =  interp(x)

    #print('sir model x=',x , 'infected=',resp)

    return resp
lbounds = [0, 0, 0]

gbounds = [np.inf, np.inf, np.inf]

guess = [1, 1, 1]

data_series = daily_data['DeathTotal']

model_param = fit(data_series,  logistic_model, lbounds, gbounds, guess)

a=model_param[0]

b=model_param[1]

c=model_param[2]

print('Parâmetros Modelo Logistico Mortos','a=',a ,'b=',b,'c=',c)



number_days_future = 90

plot_chart_model(data_series,logistic_model,model_param,number_days_future,'Estimativa de Número de Mortos Modelo Logistico')

lbounds = [0, 0, 0]

gbounds = [np.inf, np.inf, np.inf]

guess = [1, 1, 1]

data_series = daily_data['ConfirmedTotal']

model_param = fit(data_series,  logistic_model, lbounds, gbounds, guess)

a=model_param[0]

b=model_param[1]

c=model_param[2]

print('Parâmetros Modelo Logistico Confirmados','a=',a ,'b=',b,'c=',c)

number_days_future = 90

plot_chart_model(data_series,logistic_model,model_param,number_days_future,'Estimativa de Número de Casos Modelo Logistico')
# Fit the SIR Model

"""

        - initial_infected : (float) SIR Model I0 parameter. Initial infected subjects

        - beta : (float) Transmission rate

        - gamma : (float) Typical time between contacts

        - population : (float) Susceptible population

        - t0 : (float) Initial time to fix origin

"""



lbounds = [0, 0, 1, 0, -np.inf]

gbounds = [np.inf, np.inf, np.inf, np.inf, np.inf]

guess = [1, 1, 50, 3.5e6, 0]

data_series = daily_data['Active']

model_param = fit(data_series,  sir_model, lbounds, gbounds, guess)

initial_infected=model_param[0]

beta=model_param[1]

gamma=model_param[2]

population=model_param[3]

t0=model_param[4]

print('Parâmetros Modelo SIR','initial_infected=',initial_infected ,'beta=',beta,'gamma=',gamma,'population=',population, 't0=',t0)

number_days_future = 90

plot_chart_model(data_series,sir_model,model_param,number_days_future,'Estimativa de casos com Covid Ativa no Modelo SIR')
def sir_model_confirmed(x, initial_infected, beta, gamma, population, t0, initial_recovered=0, tmax=365, n=500):



    """"Uses SIR Model (https://en.wikipedia.org/wiki/Compartmental_models_in_epidemiology#The_SIR_model)

        to return the value of the Infected plus recovered Curve at time x, given the SIR Model parameters.



        Parameters:

        - x : (float) Time to evaluate the infected curve

        - initial_infected : (float) SIR Model I0 parameter. Initial infected subjects

        - beta : (float)   infection (or contact) rate 

        - gamma : (float)  recovery rate

        - population : (float) Susceptible population

        - t0 : (float) Initial time to fix origin

        - initial_recovered : (float) Initial recovered individuals (Default 0)

        - tmax : (float) Time for integrating the differential equations, in days (Default 365)

        - n : (int) Number of time points for integrating the differential equations (Default 1000)"""



    

    initial_susceptible = population - initial_infected - initial_recovered  # Everyone who is susceptible to infection

    t = np.linspace(t0, tmax, n)  # Time vector for integrating



    def derivatives(y, _):



        """SIR Model Differential Equations



            Parameters:

            - y : (np.ndarray) Array containing [Susceptible, Infected, Recovered] points

            - _ : (None) Empty parameter for consistency with scipy.integrate.odeint method"""



        s, i, _ = y

        derivative_a = -beta * s * i / population  # dS/dt

        derivative_b = beta * s * i / population - gamma * i  # dI/dt

        derivative_c = gamma * i  # dR / dt

        return derivative_a, derivative_b, derivative_c



    y0 = initial_susceptible, initial_infected, initial_recovered  # Initial conditions vector

    sol = integrate.odeint(derivatives, y0, t)  # Integrate the SIR equations over the time grid, total_time

    

    infected = sol[:, 1]  # Infected individuals for each day

    interp = interpolate.interp1d(t, infected, fill_value='extrapolate')  # Creates an interpolator with the vectors

    infected_interpolated =  interp(x)

    

    recovered = sol[:, 2]  # recoverd individuals

    interp = interpolate.interp1d(t, recovered, fill_value='extrapolate')  # Creates an interpolator with the vectors

    recovered_interpolated =  interp(x)

    

    infected_plus_recovered = infected_interpolated + recovered_interpolated

    #print('sir model','initial_infected=',initial_infected, 'beta=',beta , 'gamma=',gamma , 'population=',population,)

    #raise Exception("Stop Script") 

    #%debug

    return infected_plus_recovered
# Fit the SIR Model based on deaths

"""

        - initial_infected : (float) SIR Model I0 parameter. Initial infected subjects

        - beta : (float)   infection (or contact) rate 

        - gamma : (float)  recovery rate

        - population : (float) Susceptible population

        - t0 : (float) Initial time to fix origin

"""



lbounds = [0, 0, 0, 0, -100]

gbounds = [np.inf, np.inf, np.inf, 13e6, np.inf]

guess = [1, 1, 1, 7.0e6, -20]

data_series = daily_data['DeathTotal']/(0.005)

model_param = fit(data_series,  sir_model_confirmed, lbounds, gbounds, guess)

initial_infected=model_param[0]

beta=model_param[1]

gamma=model_param[2]

population=model_param[3]

t0=model_param[4]

r_0=beta/gamma

print('Parâmetros Modelo SIR','initial_infected=',initial_infected ,'beta=',beta,'gamma=',gamma, 'r_0=', r_0, 'population=',population, 't0=',t0)

number_days_future = 90

plot_chart_model(data_series,sir_model_confirmed,model_param,number_days_future,'Estimativa de casos no Modelo SIR baseado número de mortos e mortalidade de 0.5%')
def plot_chart(df, column_name, chart_title):

  """Mostra gráfico de barras 

  Args:

        df (Dataframe): Dados do Dataframe do arquivo CSV.

        column_name (str): nome da coluna do dataframe com os dados.

        chart_title (str): Título do Gráfico.

  """

    

  

  #https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.plot.html#pandas.DataFrame.plot

  #ax = df[column_name].plot(kind='bar', figsize=(10,5), title=chart_title, grid=True);

  #https://matplotlib.org/api/axes_api.html#matplotlib.axes.Axes

  #ax.set_xticklabels([temp.strftime("%m-%d") for temp in df.index]);

    

  #https://www.earthdatascience.org/courses/use-data-open-source-python/use-time-series-data-in-python/date-time-types-in-pandas-python/  

  fig, ax = plt.subplots(figsize=(14, 8));

    

    

  ax.xaxis.set_major_formatter(mdates.DateFormatter('%d-%m (%U)'))

  ax.xaxis.set_major_locator(mdates.DayLocator(interval=7))

  fig.autofmt_xdate()  

    

  ax.grid(zorder=0)

  # Add x-axis and y-axis

  ax.bar(df.index.values,

        df[column_name],

        color='blue');

  # Set title and labels for axes

  ax.set(xlabel="Data",

       title=chart_title);

  # Rotate tick marks on x-axis

  plt.setp(ax.get_xticklabels(), rotation=90)

  plt.show()
plot_chart(daily_data,'ConfirmedTotal','Confirmados Total')
plot_chart(daily_data,'ConfirmedPerDay','Confirmados/dia')
plot_chart(daily_data,'ConfirmedPerDayAvg','Confirmados/dia  Média 7 dias anteriores')
plot_chart(daily_data,'DeathTotal','Mortes Total')
plot_chart(daily_data,'DeathPerDay','Mortes/dia')
plot_chart(daily_data,'DeathPerDayAvg','Mortes/dia Média 7 dias anteriores')
plot_chart(daily_data,'EstimatedPercentPopulation','Estimativa % população contagiada supondo 0.5% letalidade e população de ' + str(population_sao_paulo))
plot_chart(daily_data,'CtiUsage','Uso de UTI em %')
plot_chart(daily_data,'Letality','Letalidade em %')
plot_chart(daily_data,'LetalityShiftConfirmed','Letalidade em % comparando confirmados 13 dias atrás')
plot_chart(daily_data,'ReceoveredEstimated','Estimativa recuperados comparando mortes com casos de 13 dias atrás')