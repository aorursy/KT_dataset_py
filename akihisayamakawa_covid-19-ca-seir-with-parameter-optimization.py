import numpy as np 

import pandas as pd 

import matplotlib.pyplot as plt

import seaborn as sns

from scipy import integrate, optimize

from sklearn.linear_model import LinearRegression
ca_train = pd.read_csv('../input/covid19-local-us-ca-forecasting-week-1/ca_train.csv')

ca_test = pd.read_csv('../input/covid19-local-us-ca-forecasting-week-1/ca_test.csv')

ca_submission = pd.read_csv('../input/covid19-local-us-ca-forecasting-week-1/ca_submission.csv')



train_df = ca_train

test_df =  ca_test

submission_df =  ca_submission
train_df.head()
reported = train_df[train_df['Date']>= '2020-03-10'].reset_index()

reported['day_count'] = list(range(1,len(reported)+1))

reported.head()
ydata = [i for i in reported.ConfirmedCases.values]

xdata = reported.day_count

ydata = np.array(ydata, dtype=float)

xdata = np.array(xdata, dtype=float)
N = 36000000 #population of California

inf0 = ydata[0] #Infectious

sus0 = N - inf0 #Susceptible

exp0 = 0.0 #Exposed

rec0 = 0.0 #Recovered

init_state = [sus0, exp0, inf0, rec0]

#beta = 1.0 #constant.

#gamma = 1.0 / 7.0 #constant.
# Define differential equation of SEIR model



'''

dS/dt = -beta * S * I / N

dE/dt = beta* S * I / N - epsilon * E

dI/dt = epsilon * E - gamma * I

dR/dt = gamma * I



[v[0], v[1], v[2], v[3]]=[S, E, I, R]



dv[0]/dt = -beta * v[0] * v[2] / N

dv[1]/dt = beta * v[0] * v[2] / N - epsilon * v[1]

dv[2]/dt = epsilon * v[1] - gamma * v[2]

dv[3]/dt = gamma * v[2]



'''



def seir_model(v, x, beta, epsilon, gamma, N):

    return [-beta * v[0] * v[2] / N ,beta * v[0] * v[2] / N - epsilon * v[1],

            epsilon * v[1] - gamma * v[2],gamma * v[2]]



def fit_odeint(x, beta, epsilon, gamma):

    return integrate.odeint(seir_model, init_state, x, args=(beta, epsilon, gamma, N))[:,2]
popt, pcov = optimize.curve_fit(fit_odeint, xdata, ydata)

fitted = fit_odeint(xdata, *popt)
print("Optimal parameters: beta = ", popt[0], "epsilon = ", popt[1], ", gamma = ", popt[2])
N = 36000000 #population of California

inf0 = ydata[0] #Infectious

sus0 = N - inf0 #Susceptible

exp0 = 0.0 #Exposed

rec0 = 0.0 #Recovered

init_state = [sus0, exp0, inf0, rec0]

beta = 1.0 #constant.

#gamma = 1.0 / 7.0 #constant.
# Define differential equation of SEIR model

def seir_model(v, x, beta, epsilon, gamma, N):

    return [-beta * v[0] * v[2] / N ,beta * v[0] * v[2] / N - epsilon * v[1],

            epsilon * v[1] - gamma * v[2],gamma * v[2]]



def fit_odeint(x, epsilon, gamma):

    return integrate.odeint(seir_model, init_state, x, args=(beta, epsilon, gamma, N))[:,2]
popt, pcov = optimize.curve_fit(fit_odeint, xdata, ydata)

fitted = fit_odeint(xdata, *popt)
print("Optimal parameters: epsilon = ", popt[0], ", gamma = ", popt[1])
N = 36000000 #population of California

inf0 = ydata[0] #Infectious

sus0 = N - inf0 #Susceptible

exp0 = 0.0 #Exposed

rec0 = 0.0 #Recovered

init_state = [sus0, exp0, inf0, rec0]

beta = 1.0 #constant.

gamma = 1.0 / 7.0 #constant.
# Define differential equation of SEIR model

def seir_model(v, x, beta, epsilon, gamma, N):

    return [-beta * v[0] * v[2] / N ,beta * v[0] * v[2] / N - epsilon * v[1],

            epsilon * v[1] - gamma * v[2],gamma * v[2]]



def fit_odeint(x, epsilon):

    return integrate.odeint(seir_model, init_state, x, args=(beta, epsilon, gamma, N))[:,2]
popt, pcov = optimize.curve_fit(fit_odeint, xdata, ydata)

fitted = fit_odeint(xdata, *popt)
inf_period = 1.0/gamma

lat_period = 1.0/popt[0]

print("Optimal parameters: gamma =", gamma, ", epsilon = ", popt[0], "\ninfectious period(day) = ", inf_period, ", latency period(day) = ", lat_period)
plt.plot(xdata, ydata, 'o')

plt.plot(xdata, fitted)

plt.title("Fit of SEIR model to global infected cases")

plt.ylabel("Population infected")

plt.xlabel("Days")

plt.show()
# parameters

t_max = 100 #days

dt = 1



N = 36000000 #population of California

inf0 = ydata[0] #Infectious

sus0 = N - inf0 #Susceptible

exp0 = 0.0 #Exposed

rec0 = 0.0 #Recovered

init_state = [sus0, exp0, inf0, rec0]

beta_const = 1.0 #Assumption: Infection rate is constant.

epsilon_const = popt[0]

gamma_const = 1.0 / 7.0 #Assumption: Recovery rate is constant.
# numerical integration

times = np.arange(0, t_max, dt)

args = (beta_const, epsilon_const, gamma_const, N)



# Numerical Solution using scipy.integrate

# Solver SEIR model

result = integrate.odeint(seir_model, init_state, times, args)

# plot

plt.plot(times, result)

plt.legend(['Susceptible', 'Exposed', 'Infectious', 'Removed'])

plt.title("SEIR model  COVID-19")

plt.xlabel('time(days)')

plt.ylabel('population')

plt.grid()



plt.show()
result_df = pd.DataFrame(data=result, columns=['Susceptible', 'Exposed', 'Infectious', 'Removed'])

result_df.shape
lr = LinearRegression()

X_train = reported[['ConfirmedCases']].values

Y_train = reported[['Fatalities']].values

lr.fit(X_train, Y_train)

print('coefficient = ', lr.coef_[0], '(which means Fatality rate)')

print('intercept = ', lr.intercept_)
X_pred = result_df[['Infectious']].values

Y_pred = lr.predict(X_pred)

plt.scatter(X_train, Y_train, c='blue')

plt.plot(X_pred, Y_pred, c='red')

plt.title("Regression Line")

plt.xlabel('ConfirmedCases')

plt.ylabel('Fatalities')

plt.grid()



plt.xlim([100,800])

plt.ylim([0,20])



plt.show()
Y_pred_df = pd.DataFrame(Y_pred)

result_df['Fatalities'] = Y_pred_df

result_df.head()
submission = result_df[0:len(submission_df)].reset_index()

submission_df['ConfirmedCases'] = submission['Infectious']

submission_df['Fatalities'] = submission['Fatalities']

submission_df.head()
submission_df.to_csv("submission.csv", index=False)