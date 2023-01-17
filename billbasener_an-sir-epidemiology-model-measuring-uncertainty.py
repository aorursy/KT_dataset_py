# First, import required modules

import numpy as np # numerical Python functions and linear algebra

from scipy.integrate import solve_ivp # numerical differential equations solver

import matplotlib.pyplot as plt
# defins the SIR model ODE

def SIR_ODE(t, z, beta, gamma):

    S, I, R, = z

    return [-beta*I*S/(S+I+R), beta*I*S/(S+I+R)-gamma*I, gamma*I]
# paramters for the SIR model

beta = 1/8 # 1/(time between contact)

gamma = 1/14 # # 1/(time until removal)

num_days = 4*365



# time range

time_range = [0,num_days]



# initial conditions

init_cond = [1000000,10,0]



# compute the RK45 numerical solution

sol = solve_ivp(SIR_ODE, time_range, init_cond, args=(beta, gamma),

                dense_output=True)



# plot the results

t = np.linspace(0, num_days, 300)

z = sol.sol(t)

plt.plot(t,z[0])

plt.plot(t,z[1])

plt.plot(t,z[2])

plt.xlabel('time (days)')

plt.legend(['Susceptable', 'Infected', 'Removed'], shadow=True)

plt.title('SIR Model')

plt.show()
# Number of Monte Carlo solutions to compute

num_MC_solutions = 250



# paramters for the SIR model

beta_mean = 1/8 # 1/(time between contact)

beta_stdev = 0.05*beta_mean # standard deviation in beta

beta = np.random.normal(beta_mean, beta_stdev, num_MC_solutions)

gamma_mean = 1/14 # # 1/(time until removal)

gamma_stdev = 0.05*gamma_mean # standard deviation in beta

gamma = np.random.normal(gamma_mean, gamma_stdev, num_MC_solutions)

num_days = 4*365



# time range

time_range = [0,num_days]

num_time_iterations = num_days



# initial conditions

init_cond = [1000000,10,0]



t = np.linspace(0, num_days, num_time_iterations)

S = np.zeros([num_MC_solutions,num_time_iterations])

I = np.zeros([num_MC_solutions,num_time_iterations])

R = np.zeros([num_MC_solutions,num_time_iterations])

for idx in range(num_MC_solutions):

    # compute the RK45 numerical solution

    sol = solve_ivp(SIR_ODE, time_range, init_cond, args=(beta[idx], gamma[idx]),

                    dense_output=True)

    z = sol.sol(t)

    S[idx,] = z[0]

    I[idx,] = z[1]

    R[idx,] = z[2]

# Compute uncertainty bounds

S_mean = np.mean(S,0)

S_upper = np.percentile(S, 2, 0)

S_lower = np.percentile(S, 98, 0)

I_mean = np.mean(I,0)

I_upper = np.percentile(I, 2, 0)

I_lower = np.percentile(I, 98, 0)

R_mean = np.mean(R,0)

R_upper = np.percentile(R, 2, 0)

R_lower = np.percentile(R, 98, 0)



plt.plot(t, S_mean, 'k', color='b')

plt.fill_between(t, S_lower, S_upper,

    alpha=0.25, edgecolor='b', facecolor='b',

    linewidth=0)



plt.plot(t, I_mean, 'k', color='orange')

plt.fill_between(t, I_lower, I_upper,

    alpha=0.25, edgecolor='orange', facecolor='orange',

    linewidth=0)



plt.plot(t, R_mean, 'k', color='g')

plt.fill_between(t, R_lower, R_upper,

    alpha=0.25, edgecolor='g', facecolor='g',

    linewidth=0)

plt.xlabel('time (days)')

plt.legend(['Susceptable', 'Infected', 'Removed'], shadow=True)

plt.title('SIR Model')

plt.show()
R0 = beta/gamma # R_0 is the basic reproduction umber (https://en.wikipedia.org/wiki/Basic_reproduction_number)

R_final = R[:,-1] # This is the final value for R.  If the system achieves equlibrium, this is the number (proport) of individuals needed to achieve herd immunity.

plt.plot(R0, R_final, 'k.', markersize=5)

plt.xlabel('$R_0$')

plt.ylabel('Final value of R')

plt.title('Herd Immunity vs. $R_0$')

plt.show()
R0 = beta/gamma # R_0 is the basic reproduction umber (https://en.wikipedia.org/wiki/Basic_reproduction_number)

I_max = np.max(I,1) # The peak value in the infected population.

plt.plot(R0, I_max, 'k.', markersize=5)

plt.xlabel('$R_0$')

plt.ylabel('Peak value in I')

plt.title('Peak value in I vs. $R_0$')

plt.show()