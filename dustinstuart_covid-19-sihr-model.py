import numpy as np

import matplotlib.pyplot as plt

import scipy.integrate
def f(y, t, beta, gamma):

    S, I, R = y

    dS = - beta * I * S

    dI = beta * I * S - gamma * I

    dR = gamma * I

    return np.array([dS, dI, dR])



y0 = [1, 1e-6, 0]

t = np.linspace(0, 100)

y = scipy.integrate.odeint(f, y0, t, (.5, .1))

plt.semilogy(t, y)

plt.legend(['Susceptible', 'Infected', 'Recovered'])
ns = 5 # number of states



def f(y, t, beta, gamma, gamma2, hosp, dead):

    s, i, h, r, d = y.reshape((ns, -1))

    lambd = np.dot(beta, i)

    ds = - lambd * s

    di = lambd * s - gamma * i

    dh = gamma * i * hosp - gamma2 * h

    dr = gamma * i * (1 - hosp) + gamma2 * h * (1 - dead)

    dd = gamma2 * h * dead

    return np.concatenate((ds, di, dh, dr, dd))



n = 2 # Number of classes (e.g. age, geographic location)

class_distribution = np.ones(n) / n # Start with equal population in each class

beta = .5 * np.ones((n, n)) # Infection rate between classes (number of transmissions per person per day)

gamma = .1 # Removal rate (1 / length of illness in days)

gamma2 = .1 # Removal rate from hospital (1 / length of time spent in hospital)

hosp = np.array([0, .1]) # Probability of hospitalisation for each class

dead = np.array([0, .2]) # Probability of death (after hospitalisation) for each class

y0 = np.zeros(n*ns)

y0[:n] = class_distribution

y0[n] = 1e-6 # Initial infection 

t = np.linspace(0,100)

y = scipy.integrate.odeint(f, y0, t, (beta, gamma, gamma2, hosp, dead))



yT = y.reshape((-1,ns,n)).sum(2)

plt.semilogy(t, yT)

plt.legend(['Susceptible', 'Infected', 'Hospital', 'Recovered', 'Dead'])

plt.ylim(1e-6)