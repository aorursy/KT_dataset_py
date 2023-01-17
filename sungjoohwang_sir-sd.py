# importing "numpy" library for dealing with multi-dimensional arrays and matrices
import numpy as np
# "scipy" is a library for technical computing - "scipy.integrate" is a sub-package for integration technique and "odeint" is a (ordinary differential equation integration) library 
from scipy.integrate import odeint
# "matplotlib" is a plotting library - matplotlib. pyplot is a collection of command style functions that make matplotlib work like MATLAB
import matplotlib.pyplot as plt

# Total population, N.
N = 1000
# Initial number of infected and recovered individuals, I0 and R0.
I0, R0 = 1, 0
# Everyone else, S0, is susceptible to infection initially.
S0 = N - I0 - R0
# Contact rate (effective contact rate considering infection probability), beta, and mean recovery rate, gamma, (in 1/days).
beta, gamma = 0.25, 1./15 
# A grid of time points (in days) linspace (start, stop, num=161) generates linearly spaced arrays
t = np.linspace(0, 160, 161)

# The SIR model differential equations.
def deriv(y, t, N, beta, gamma):
    S, I, R = y
    dSdt = -beta * S * I / N
    dIdt = beta * S * I / N - gamma * I
    dRdt = gamma * I
    return dSdt, dIdt, dRdt

# Initial conditions vector
y0 = S0, I0, R0
# Integrate the SIR equations over the time grid, t.
# scipy.integrate.odeint(func, y0, t, args=(), ...) see https://apmonitor.com/pdc/index.php/Main/SolveDifferentialEquations
ret = odeint(deriv, y0, t, args=(N, beta, gamma))
# The T attribute is the transpose of the array
S, I, R = ret.T

# Plot the data on three separate curves for S(t), I(t) and R(t)
fig = plt.figure(facecolor='w')
ax = fig.add_subplot(111, facecolor='#dddddd', axisbelow=True)
ax.plot(t, S/1000, 'b', alpha=0.5, lw=2, label='Susceptible')
ax.plot(t, I/1000, 'r', alpha=0.5, lw=2, label='Infected')
ax.plot(t, R/1000, 'g', alpha=0.5, lw=2, label='Recovered with immunity')
ax.set_xlabel('Time /days')
ax.set_ylabel('Number (1000s)')
ax.set_ylim(0,1.2)
ax.yaxis.set_tick_params(length=0)
ax.xaxis.set_tick_params(length=0)
ax.grid(b=True, which='major', c='w', lw=2, ls='-')
legend = ax.legend()
legend.get_frame().set_alpha(0.5)
for spine in ('top', 'right', 'bottom', 'left'):
    ax.spines[spine].set_visible(False)
plt.show()

print(t)
print()
print(ret.T)

print()
print(S)
print()
print(I)
print()
print(R)
print()