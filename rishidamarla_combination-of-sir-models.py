# Importing all necessary libraries.

import pandas as pd

import numpy as np 

import matplotlib.pyplot as plt

import scipy

import scipy.integrate

from scipy.integrate import odeint
def SIR_model(y, t, beta, gamma):

    S, I, R = y

    dS_dt = -beta*S*I

    dI_dt = beta*S*I - gamma*I

    dR_dt = gamma*I

    

    return([dS_dt, dI_dt, dR_dt])
S0 = 0.9

I0 = 0.1

R0 = 0.0

beta = 0.35

gamma = 0.1



t = np.linspace(0, 100, 10000)





solution = scipy.integrate.odeint(SIR_model, [S0, I0, R0], t, args = (beta, gamma))

solution = np.array(solution)
plt.figure(figsize=[10,10])

plt.plot(t, solution[:, 0], label = 'S(t)')

plt.plot(t, solution[:, 1], label = 'I(t)')

plt.plot(t, solution[:, 2], label = 'R(t)')

plt.grid()

plt.xlabel("Time")

plt.ylabel("Proportions")

plt.title("SIR Model")

plt.legend()

plt.show()
def deriv(y, t, N, beta, gama):

        S, I, R = y

        dSdt = -beta * S * I / N

        dIdt = beta * S * I / N - gama * I

        dRdt = gama * I

        return dSdt, dIdt, dRdt



def sir(N, beta, gama=1/10, I0=1, R0=0, t=90):

        t = np.linspace(0, t, t)

        S0 = N - I0 - R0

        y0 = S0, I0, R0

        ret = odeint(deriv, y0, t, args=(N, beta, gama))

        S, I, R = ret.T

        return {'S': S, 'I': I, 'R': R, 't': t}



def curves(s, title):

        fig, ax = plt.subplots(figsize=(12,8))

        plt.plot(s['t'], s['S'], 'b', alpha=0.5, lw=4, label='Susceptible')

        plt.plot(s['t'], s['I'], 'r', alpha=0.5, lw=4, label='Infected')

        plt.plot(s['t'], s['R'], 'g', alpha=0.5, lw=4, label='Recovered')

        plt.grid(which='major', axis='y')

        plt.ticklabel_format(scilimits=(6,6), axis='y')

        plt.text(0.9,1,s='β: %.3f γ: %.2f' % (beta, gama), transform=ax.transAxes, fontsize=10)

        plt.title(title, fontsize=14, fontweight='bold', color='#333333')

        plt.xlabel('Days', fontsize=12)

        plt.ylabel('Number of cases (million)', fontsize=12)

        legend = plt.legend(loc=5, fontsize=12)

        ax.set_ylim(0)

        [ax.spines[spine].set_visible(False) for spine in ('top', 'right', 'left')]

        plt.gca().xaxis.set_major_locator(plt.MultipleLocator(10))

plt.show();
gama = 1/5.2

beta = 1

curves(sir(51000000 , beta, gama, t=180, I0=100), 'COVID-19 SIR Model: No-action')
# Effects of Social Distancing

beta = 0.88

curves(sir(51000000, beta, gama, t=120, I0=100), 'COVID-19 SIR Model: Social Distancing')
# Effects of Social Distancing and Quarantine.

beta = 0.449

curves(sir(51000000, beta, gama, t=120, I0=100), 'COVID-19 SIR Model: Social Distancing + Quarantine')
# The SIR model differential equations.

def deriv(y, t, N, beta, gamma):

    S, I, R = y

    dSdt = -beta * S * I / N

    dIdt = beta * S * I / N - gamma * I

    dRdt = gamma * I

    return dSdt, dIdt, dRdt



def plot_sir_curves(title, N, I0, R0, beta, gamma):

    R_0 = beta / gamma



    # A grid of time points (in days)

    t = np.linspace(0, 365 * 2, 1000)

    # Initial conditions vector

    y0 = S0, I0, R0

    # Integrate the SIR equations over the time grid, t.

    ret = odeint(deriv, y0, t, args=(N, beta, gamma))

    S, I, R = ret.T



    # Plot the data on three separate curves for S(t), I(t) and R(t)

    fig = plt.figure(facecolor='w', figsize=(20, 10))



    ax = fig.add_subplot(111) #, axisbelow=True)

    ax.plot(t, S, 'b', alpha=0.5, lw=2, label='Susceptible')

    ax.plot(t, I, 'r', alpha=0.5, lw=2, label='Infected')

    ax.plot(t, R, 'g', alpha=0.5, lw=2, label='Recovered with immunity')

    ax.set_xlabel('Time /days')

    ax.set_ylabel('Number')

    ax.set_ylim(0, N * 1.1)

    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, loc: "{:,}".format(int(x))))



    legend = ax.legend()

    legend.get_frame().set_alpha(1.0)



    ax.minorticks_on()

    ax.grid(b=True, axis='y', which='minor', linestyle=':')

    ax.grid(b=True, axis='y', which='major', lw=2, ls='-')

    ax.grid(b=True, axis='x', which='major', lw=2, ls='-', alpha=0.5)

    

    plt.title("{:}\n(R_0: {:.3f}, beta: {:.3f})".format(title, R_0, beta))



    plt.show()
# This is a model of a pessimistic scenario with no social distancing at all.

# Total population, N.

N = 327000000



# Initial number of infected and recovered individuals, I0 and R0.

I0, R0 = 3500, 0

# Everyone else, S0, is susceptible to infection initially.

S0 = N - I0 - R0



# mean recovery rate, gamma, (in 1/days).

gamma = 1./14



# Contact rate, beta (percent of population touched+infected by an infected person daily).

# Also, R_0 = beta / gamma

R_0 = 4

beta = R_0 * gamma



# Contact rate, beta (percent of population touched+infected by an infected person daily).

# Also, R_0 = beta / gamma

R_0 = 4

beta = R_0 * gamma



plot_sir_curves("US - Worst Case Estimate, No Lockdown Scenario", N, I0, R0, beta, gamma)
# This is a more optimistic scenario with this still no social distancing but a lower R0 Number.

# Total population, N.

N = 327000000



# Initial number of infected and recovered individuals, I0 and R0.

I0, R0 = 3500, 0

# Everyone else, S0, is susceptible to infection initially.

S0 = N - I0 - R0



# mean recovery rate, gamma, (in 1/days).

gamma = 1./14



# Contact rate, beta (percent of population touched+infected by an infected person daily).

# Also, R_0 = beta / gamma

R_0 = 1.4

beta = R_0 * gamma



plot_sir_curves("US - Optimistic Estimate, No Lockdown Scenario", N, I0, R0, beta, gamma)
# This is a model of a more reasonable scenario with a lower R0 than the pessimistic scenario but a higher one than the 

# optimistic one.

# Total population, N.

N = 327000000



# Initial number of infected and recovered individuals, I0 and R0.

I0, R0 = 3500, 0

# Everyone else, S0, is susceptible to infection initially.

S0 = N - I0 - R0



# mean recovery rate, gamma, (in 1/days).

gamma = 1./14



# Contact rate, beta (percent of population touched+infected by an infected person daily).

# Also, R_0 = beta / gamma

R_0 = 2.5

beta = R_0 * gamma



plot_sir_curves("US - Reasonable $R_0$ with no lockdown", N, I0, R0, beta, gamma)
# Susceptible equation

def fa(N, a, b, beta):

    fa = -beta*a*b

    return fa



# Infected equation

def fb(N, a, b, beta, gamma):

    fb = beta*a*b - gamma*b

    return fb



# Recovered/deceased equation

def fc(N, b, gamma):

    fc = gamma*b

    return fc
def rK4(N, a, b, c, fa, fb, fc, beta, gamma, hs):

    a1 = fa(N, a, b, beta)*hs

    b1 = fb(N, a, b, beta, gamma)*hs

    c1 = fc(N, b, gamma)*hs

    ak = a + a1*0.5

    bk = b + b1*0.5

    ck = c + c1*0.5

    a2 = fa(N, ak, bk, beta)*hs

    b2 = fb(N, ak, bk, beta, gamma)*hs

    c2 = fc(N, bk, gamma)*hs

    ak = a + a2*0.5

    bk = b + b2*0.5

    ck = c + c2*0.5

    a3 = fa(N, ak, bk, beta)*hs

    b3 = fb(N, ak, bk, beta, gamma)*hs

    c3 = fc(N, bk, gamma)*hs

    ak = a + a3

    bk = b + b3

    ck = c + c3

    a4 = fa(N, ak, bk, beta)*hs

    b4 = fb(N, ak, bk, beta, gamma)*hs

    c4 = fc(N, bk, gamma)*hs

    a = a + (a1 + 2*(a2 + a3) + a4)/6

    b = b + (b1 + 2*(b2 + b3) + b4)/6

    c = c + (c1 + 2*(c2 + c3) + c4)/6

    return a, b, c
def SIR(N, b0, beta, gamma, hs):

    

    """

    N = total number of population

    beta = transition rate S->I

    gamma = transition rate I->R

    k =  denotes the constant degree distribution of the network (average value for networks in which 

    the probability of finding a node with a different connectivity decays exponentially fast

    hs = jump step of the numerical integration

    """

    

    # Initial condition

    a = float(N-1)/N -b0

    b = float(1)/N +b0

    c = 0.



    sus, inf, rec= [],[],[]

    for i in range(10000): # Run for a certain number of time-steps

        sus.append(a)

        inf.append(b)

        rec.append(c)

        a,b,c = rK4(N, a, b, c, fa, fb, fc, beta, gamma, hs)



    return sus, inf, rec
# Parameters of the model

N = 7800*(10**6)

b0 = 0

beta = 0.7

gamma = 0.2

hs = 0.1



sus, inf, rec = SIR(N, b0, beta, gamma, hs)



f = plt.figure(figsize=(8,5)) 

plt.plot(sus, 'b.', label='susceptible');

plt.plot(inf, 'r.', label='infected');

plt.plot(rec, 'c.', label='recovered/deceased');

plt.title("SIR model")

plt.xlabel("time", fontsize=10);

plt.ylabel("Fraction of population", fontsize=10);

plt.legend(loc='best')

plt.xlim(0,1000)

plt.savefig('SIR_example.png')

plt.show()
# Feel free to change these parameters and see how the model changes

N = 15000



S0 = 14990

I0 = 10

R0 = 0



b = 0.243

k = 0.143



no_of_days = 100



# Calculating Basic Reproduction Number Ro

reproduction_number = b / k



# A grid of time points (in days)

t = np.linspace(0, no_of_days + 1, no_of_days + 1)



# Initial conditions vector

y0 = S0, I0, R0





# Integrate the SIR equations over the time grid, t.

ret = odeint(deriv, y0, t, args=(N, b, k))

S, I, R = ret.T



# Rounding off

S = np.round(np.array(S))

I = np.round(np.array(I))

R = np.round(np.array(R))



# Plot the data on three separate curves for S(t), I(t) and R(t)

fig = plt.figure(facecolor='w')

ax = fig.add_subplot(111, facecolor='#dddddd', axisbelow=True)

ax.plot(t, S, 'b', alpha=0.5, lw=2, label='Susceptible')

ax.plot(t, I, 'r', alpha=0.5, lw=2, label='Infected')

ax.plot(t, R, 'g', alpha=0.5, lw=2, label='Recovered')

ax.set_xlabel('Time /days')

ax.set_ylabel('Number')

ax.set_ylim(0,1.2*N)

ax.yaxis.set_tick_params(length=0)

ax.xaxis.set_tick_params(length=0)

ax.grid(b=True, which='major', c='w', lw=2, ls='-')

legend = ax.legend()

legend.get_frame().set_alpha(0.5)

for spine in ('top', 'right', 'bottom', 'left'):

    ax.spines[spine].set_visible(False)

plt.show()

# The SIR model differential equations.

def deriv(state, t, N, beta, gamma):

    S, I, R = state

    # Change in S population over time

    dSdt = -beta * S * I / N

    # Change in I population over time

    dIdt = beta * S * I / N - gamma * I

    # Change in R population over time

    dRdt = gamma * I

    return dSdt, dIdt, dRdt
effective_contact_rate = 0.8

recovery_rate = 1/4



# We'll compute this for fun

print("R0 is", effective_contact_rate / recovery_rate)



# What's our start population look like?

# Everyone not infected or recovered is susceptible

total_pop = 1000

recovered = 0

infected = 1

susceptible = total_pop - infected - recovered



# A list of days, 0-160

days = range(0, 160)



# Use differential equations magic with our population

ret = odeint(deriv,

             [susceptible, infected, recovered],

             days,

             args=(total_pop, effective_contact_rate, recovery_rate))

S, I, R = ret.T



# Build a dataframe because why not

df = pd.DataFrame({

    'suseptible': S,

    'infected': I,

    'recovered': R,

    'day': days

})



plt.style.use('ggplot')

df.plot(x='day',

        y=['infected', 'suseptible', 'recovered'],

        color=['#bb6424', '#aac6ca', '#cc8ac0'],

        kind='area',

        stacked=False)



# If you get the error:

#

#     When stacked is True, each column must be either all

#     positive or negative.infected contains both...

#

# just change stacked=True to stacked=False