import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns
x = pd.Series((np.arange(350)-175)/50*np.pi)

fx = abs(((x+np.pi) % (2*np.pi))-np.pi)**3

df = pd.DataFrame({'x': x, 'fx': fx})

df['k0'] = np.pi**3 / 4

df['c0'] = df['k0']



# Calculates l2 error for x, y, and an estimate e.

def l2_error(x, y, e):

    return sum(x.diff()[1:].reset_index(drop=True)

               .multiply(y[:-1].reset_index(drop=True)-e[:-1].reset_index(drop=True))**2)



print("L2 Error: " + str(l2_error(df['x'], df['fx'], df['c0'])))

df[['x', 'fx', 'k0']].plot.line(x='x')
# Calculation of the Fourier coefficients for x^3

def get_x3_Fourier_Coefficient(k):

    return (-1)**k * 6.0 * np.pi / k**2 - 12.0 / (np.pi * k**4) * ((-1)**k - 1)
df['k1'] = get_x3_Fourier_Coefficient(1) * np.cos(x)

df['c1'] = df['k0'] + df['k1']

print("L2 Error: " + str(l2_error(df['x'], df['fx'], df['c1'])))

fig1, ax = plt.subplots(1, 2, figsize=(14,8), sharex=True)

df[['x', 'fx', 'k0', 'k1']].plot.line(x='x', ax=ax[0])

df[['x', 'fx', 'k0', 'c1']].plot.line(x='x', ax=ax[1])
df['k2'] = get_x3_Fourier_Coefficient(2) * np.cos(2 * df['x'])

df['c2'] = df['c1'] + df['k2']

print("L2 Error: " + str(l2_error(df['x'], df['fx'], df['c2'])))

fig1, ax = plt.subplots(1, 2, figsize=(14,8), sharex=True)

df[['x', 'fx', 'k0', 'k1', 'k2']].plot.line(x='x', ax=ax[0])

df[['x', 'fx', 'k0', 'c1', 'c2']].plot.line(x='x', ax=ax[1])
df['k3'] = get_x3_Fourier_Coefficient(3) * np.cos(3 * df['x'])

df['c3'] = df['c2'] + df['k3']

print("L2 Error: " + str(l2_error(df['x'], df['fx'], df['c3'])))

fig1, ax = plt.subplots(1, 2, figsize=(14,8), sharex=True)

df[['x', 'fx', 'k0', 'k1', 'k2', 'k3']].plot.line(x='x', ax=ax[0])

df[['x', 'fx', 'k0', 'c1', 'c2', 'c3']].plot.line(x='x', ax=ax[1])
df['c'] = df['k0']

for i in np.arange(32):

    df['c'] = df['c'] + get_x3_Fourier_Coefficient(i+1) * np.cos((i+1) * df['x'])

print("L2 Error: " + str(l2_error(df['x'], df['fx'], df['c'])))

fig2, ax2 = plt.subplots(1, 1, figsize=(14,8), sharex=True)

df[['x', 'fx', 'c']].plot.line(x='x', ax=ax2)