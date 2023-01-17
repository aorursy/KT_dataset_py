import numpy as np 
import pandas as pd 
import matplotlib as mpl
import matplotlib.pyplot as plt
import math

# Usaremos o pip fbm - https://pypi.org/project/fbm/
from fbm import mbm, mgn, times
from fbm import MBM
# Define a hurst function
def h(t):
    return 0.75 - 0.5 * t

# Generate a mbm realization
mbm_sample = mbm(n=1024, hurst=h, length=1, method='riemannliouville')

# Generate a fGn realization
mgn_sample = mgn(n=1024, hurst=h, length=1, method='riemannliouville')

# Get the times associated with the mBm
t_values = times(n=1024, length=1)

print("total values:", len(t_values), t_values)
print("total mgn sample:", len(mgn_sample), mgn_sample)
print("total mbm sample:", len(mbm_sample), mbm_sample)

plt.style.use('classic')
plt.xlabel('Tempo')
plt.ylabel('Transformação aplicada')
plt.title('Dados resultantes do Hurst function')
plt.plot(mgn_sample)
plt.show()
plt.style.use('classic')
plt.xlabel('Dias úteis')
plt.ylabel('Preço da ação')
plt.title('Gráfico de uma ação')
plt.plot(mbm_sample)
plt.show()
# Cada vez que for executado, o trecho abaixo gera um gráfico diferente.

# Example Hurst function with respect to time.
def h(t):
    return 0.25 * math.sin(20*t) + 0.5

m = MBM(n=1024, hurst=h, length=1, method='riemannliouville') # or m = MBM(1024, h)

# Generate a mBm realization
mbm_sample = m.mbm()

# Generate a mGn realization
mgn_sample = m.mgn()

# Get the times associated with the mBm
t_values = m.times()

plt.style.use('classic')
plt.xlabel('Dias úteis')
plt.ylabel('Preço da ação')
plt.title('Gráfico de uma ação')
plt.plot(mbm_sample)
plt.show()
def h1(t):
    return 0.75 - 0.5 * t
def h2(t):
    return 0.25 * math.sin(20*t) + 0.5
def h3(t):
    return 0.02
def h4(t):
    return 0.02 + math.log(2-t)

# Generate a fGn realization
mgn_sample1 = mgn(n=1024, hurst=h2, length=1, method='riemannliouville')
mbm_sample = m.mbm()

plt.style.use('classic')
plt.title('Gráfico de uma ação')

plt.subplot(2, 1, 1)
plt.xlabel('Dias úteis')
plt.ylabel('Preço da ação')
plt.plot(mbm_sample)

plt.subplot(2, 1, 2)
plt.xlabel('Tempo')
plt.ylabel('Transformação aplicada')
plt.plot(mgn_sample1)

plt.show()