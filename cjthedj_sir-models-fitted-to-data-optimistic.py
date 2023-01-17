import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from scipy.integrate import odeint

from scipy.optimize import curve_fit

import matplotlib.pyplot as plt  



import os

#for dirname, _, filenames in os.walk('/kaggle/input'):

#    for filename in filenames:

#        print(os.path.join(dirname, filename))



def getConfirmedCases(country, startingAt=5):

    data = pd.read_csv("/kaggle/input/covid19-global-forecasting-week-2/train.csv")

    dataOfCountry = data[data.Country_Region == country]

    result = dataOfCountry.ConfirmedCases[dataOfCountry.ConfirmedCases > startingAt]

    return result







# Any results you write to the current directory are saved as output.
def SIR(X, t, beta, gamma, N):

    """

    S = X[0], I=X[1], R=X[2]

    """

    dSdt = -beta*X[0]*X[1]/N

    dIdt = -dSdt - gamma*X[1]

    dRdt = gamma*X[1]

    return [dSdt, dIdt, dRdt]

    beta, gamma, N = 0.9, 0.2, 6*10**7

    I0 = 5

    X0 = [N-I0, I0, 0]

    ts = np.linspace(0, 100 - 1, 100)

    Xs = odeint(SIR, X0, ts, args=(beta, gamma, N))

    

    f = plt.figure(figsize=(10,5))

    plt.plot(ts, Xs[:,0], label='susceptible');

    plt.plot(ts, Xs[:,1], label='infected, not yet quarantined');

    plt.plot(ts, Xs[:,2], label='confirmed infections');

    plt.ylabel("number of people", fontsize=10);

    plt.xlabel("time [days]", fontsize=10);

    plt.legend()

    plt.show()


def confirmedSIR(t, beta, gamma):

    N = 6*10**7

    #N = 1000

    I0 = 1

    X0 = [N-I0, I0, 0]

    return odeint(SIR, X0, t, args=(beta, gamma, N))[:,2]



def allSIR(t, beta, gamma):

    N = 6*10**7

    #N = 1000

    I0 = 1

    X0 = [N-I0, I0, 0]

    return odeint(SIR, X0, t, args=(beta, gamma, N))

    observed = getConfirmedCases("Spain")



    N = 6*10**7

    X0 = [N-I0, I0, 0]

    ts = np.linspace(0, len(observed) - 1, len(observed))

    

    popt, pcov = curve_fit(confirmedSIR, ts, observed)

    

    longerTime = np.linspace(0, 100, 1000)

    fitted = allSIR(longerTime, *popt)

    

    f = plt.figure(figsize=(10,5))

    plt.title("Best fitting SIR model for Spain")

    plt.plot(ts, observed, "ro")

    plt.plot(longerTime, fitted[:,1], label='Infected, not quarantined yet')

    plt.plot(longerTime, fitted[:,2], label='Confirmed cases')

    plt.legend()

    plt.show()

              
print("fitted infectiousness",popt[0])

print("fitted chance of being quarantined when sick: ", popt[1])
def plotSIR(country, usingLogScale=False, I0=5):



    observed = getConfirmedCases(country, I0)

    observed = observed[~np.isnan(observed)]



    beta, gamma, N = 0.6, 0.2, 6*10**7

    X0 = [N-I0, I0, 0]

    ts = np.linspace(0, len(observed) - 1, len(observed))

    Xs = odeint(SIR, X0, ts, args=(beta, gamma, N))



    shortening = 10

    popt_partialData, pcov_partialData = curve_fit(confirmedSIR,

                                       ts[:len(ts) - shortening],

                                       observed[:len(ts) - shortening])

    longerTime = np.linspace(0, 100, 1000)

    fitted_partialData = allSIR(longerTime, *popt_partialData)



    popt, pcov = curve_fit(confirmedSIR, ts, observed)

    fitted = allSIR(longerTime, *popt)



    predicted = confirmedSIR(ts, *popt)

    residuals = predicted - observed

    predicted_partialData = confirmedSIR(ts[:len(ts) - shortening], *popt_partialData)

    residuals_partialData = predicted_partialData - observed[:len(ts) - shortening]



    f = plt.figure(figsize=(20,10))

    plt.subplot(311)

    plt.title("SIR model for " + country)

    plt.legend(loc='best')

    if usingLogScale:

        plt.ylabel("Logarithm of confirmed infections", fontsize=10);

        plt.plot(ts, np.log(observed),'ro', label='observed')

        plt.plot(longerTime, np.log(fitted[:,2]), label='fitted')

        plt.plot(longerTime, np.log(fitted_partialData[:,2]), label='fitted on partial data')

    else:

        plt.xlabel("time [days]", fontsize=10);

        plt.plot(ts,observed,'ro', label='observed')

        plt.plot(longerTime,fitted[:,2], label='fitted')

        plt.plot(longerTime, fitted_partialData[:,2], label='fitted on partial data')

    plt.legend()



    plt.subplot(312)

    plt.ylabel("Infected, not quarantined")

    plt.plot(longerTime, fitted[:,1], label='fitted')

    plt.plot(longerTime, fitted_partialData[:,1], label='fitted on partial data')



    plt.subplot(313)

    plt.ylabel("Residuals")

    plt.plot(ts, residuals, 'o')

    plt.plot(ts[:len(ts) - shortening], residuals_partialData, 'o')

    plt.xlim(0, longerTime[-1])

    plt.xlabel("time [days]", fontsize=10);

    plt.show()

plotSIR("Spain", usingLogScale=True)

plotSIR("Italy", usingLogScale=True)
plotSIR("Korea, South", I0=np.exp(4.5), usingLogScale=True)
from sklearn.model_selection import TimeSeriesSplit

from sklearn.metrics import mean_squared_error

from sklearn.metrics import mean_absolute_error



def evaluateModel(countries=["Spain"], usingPlotting=False):

    tscv = TimeSeriesSplit(n_splits=5)

    scores = []

    parameters = []



    for country in countries:

        observed = getConfirmedCases(country).to_numpy()

        observed = observed[~np.isnan(observed)]



        I0, N = 5, 6*10**7

        X0 = [N-I0, I0, 0]

        for train, test in tscv.split(observed):

            popt, pcov = curve_fit(confirmedSIR, train, observed[train])

            fitted = allSIR(np.linspace(0,50 -1, 50), *popt)



            scores.append(mean_absolute_error(fitted[test,2], observed[test]))

            parameters.append(popt)

            if usingPlotting:

                plt.plot(test,np.log(fitted[test,2]))

        if usingPlotting:

            plt.plot(np.log(observed), 'o')

            plt.show()

    return parameters, scores

params, scores = evaluateModel(usingPlotting=True)

print("Prediction error: ", *scores)