# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from scipy.stats import binom

from scipy.stats import norm



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
m = 10

q = 0.3
# Creamos la distribución de frecuencia

freq_dist = {}

for i in range(m+1):

    freq_dist[i] = binom.pmf(i, m, q)

freq_dist
# Creamos la distribución de severidad

sev_dist = {

    0: 0.05,

    1000: 0.10,

    5000: 0.30,

    10000: 0.20,

    20000: 0.15,

    50000: 0.10,

    100000: 0.05,

    500000: 0.025,

    1000000: 0.025

}
# Creamos la distribución de pérdida agregada

convolutions = {}



# Fijamos la convolución 0

convolutions[0] = {0: 1}



# Fijamos la convolución 1

new_conv = {}

for i in sev_dist:

    new_conv[i] = sev_dist[i]

convolutions[1] = new_conv



# Fijamos las demás convoluciones

for n in range(2,m+1):

    old_conv = new_conv

    new_conv = {}

    for i in sev_dist:

        for j in old_conv:

            if(i+j in new_conv):

                new_conv[i+j] = new_conv[i+j] + old_conv[j] * sev_dist[i]

            else:

                new_conv[i+j] = old_conv[j] * sev_dist[i]

    convolutions[n] = new_conv



convolutions
# Creamos la distribución de la pérdida agregada

aggregate_dist = {}



for i in convolutions:

    for j in convolutions[i]:

        if(j in aggregate_dist):

            aggregate_dist[j] = aggregate_dist[j] + convolutions[i][j]*freq_dist[i]

        else:

            aggregate_dist[j] = convolutions[i][j]*freq_dist[i]



print("El soporte de la distribución tiene ", len(aggregate_dist), " datos.")

aggregate_dist
# Graficamos

import matplotlib.pyplot as plt

plt.plot(*zip(*sorted(aggregate_dist.items())))

plt.show()
# Verificamos que la suma de las masas de probabilidad sea igual a 1

suma = 0

for i in aggregate_dist:

    suma = suma + aggregate_dist[i]

suma
# Calculamos la esperanza

esperanza1 = 0

for i in aggregate_dist:

    esperanza1 = esperanza1 + i * aggregate_dist[i]

print("La esperanza es (método 1): ", esperanza1)



# También podemos calcular la esperanza mediante fórmula

freq_mean = 0

for i in freq_dist:

    freq_mean = freq_mean + i * freq_dist[i]

sev_mean = 0

for i in sev_dist:

    sev_mean = sev_mean + i * sev_dist[i]

esperanza2 = freq_mean * sev_mean 

print("La esperanza es (método 2): ", esperanza2)
# Calculamos el segundo momento

m2 = 0

for i in aggregate_dist:

    m2 = m2 + i**2 * aggregate_dist[i]

print("El segundo momento es: ", m2)



# Calculamos la varianza con el segundo momento

varianza1 = 0

for i in aggregate_dist:

    varianza1 = varianza1 + (i - esperanza1)**2 * aggregate_dist[i]

print("La varianza es (método 1): ", varianza1)



# Calculamos la varianza directamente

varianza2 = m2 - esperanza1**2



print("La varianza es (método 2): ", varianza2)



# Calculamos la varianza por fórmula

freq_var = 0

for i in freq_dist:

    freq_var = freq_var + (i - freq_mean)**2 * freq_dist[i]

sev_var = 0

for i in sev_dist:

    sev_var = sev_var + (i - sev_mean)**2 * sev_dist[i]

varianza3 = sev_mean**2*freq_var + sev_var*freq_mean

print("La varianza calculada por fórmula es: ", varianza3)

    

# Calculamos la asimetría

asimetría = 0

for i in aggregate_dist:

    asimetría = asimetría + (i - esperanza1)**3 * aggregate_dist[i]

print("La asimetría es: ", asimetría)



# Calculamos el coeficiente de asimetría

print("El coeficiente de asimetría es: ", asimetría/varianza1**(0.5*3))
x = 2000000



# Calculamos la distribución acumulada real

cumm_prob = 0

for i in aggregate_dist:

    if(i<=x):

        cumm_prob = cumm_prob + aggregate_dist[i]

print("La probabilidad exacta es: ", cumm_prob)
# Mediante distribución Normal



print("La probabilidad aproximada (normal) es: ", norm.cdf((x - esperanza1)/varianza1**.5))
# Mediante distribución Lognormal

sigma = (np.log(m2) - 2*np.log(esperanza1) )**0.5

mu = np.log(esperanza1) - 0.5*sigma**2

print("La probabilidad aproximada (lognormal) es: ", norm.cdf((np.log(x) - mu)/sigma))
# Calculamos el valor inicial para la recursividad: Pr(S=0)

p0 = (1+q*(sev_dist[0]-1))**m

print("Pr(S=0) = ",p0)
# Calculamos todos los valores restantes

aggregate_dist_recursive = {}

aggregate_dist_recursive[0] = p0

a = -q / (1-q)

b = (m+1)*q/(1-q)

h = 1000

maximo = 10000

for j in range(1,maximo+1):

    aggregate_dist_recursive[j] = 0

    for i in range(1,min(j+1,1000)):

        aggregate_dist_recursive[j] = aggregate_dist_recursive[j] + (a+b*i/j)*sev_dist.get(i*1000, 0)*aggregate_dist_recursive[j-i]/(1-a*sev_dist[0])

aggregate_dist_recursive
print('Pr(S=10,000)')

print(aggregate_dist[10000], ' (Convoluciones)')

print(aggregate_dist_recursive[10], ' (Recursivo)')

print()

print('Pr(S=100,000)')

print(aggregate_dist[100000], ' (Convoluciones)')

print(aggregate_dist_recursive[100], ' (Recursivo)')

print()

print('** A partir de aquí se aprecia más el error de redondeo **')

print()

print('Pr(S=1,000,000)')

print(aggregate_dist[1000000], ' (Convoluciones)')

print(aggregate_dist_recursive[1000], ' (Recursivo)')

print()

print('Pr(S=10,000,000)')

print(aggregate_dist[10000000], ' (Convoluciones)')

print(aggregate_dist_recursive[10000], ' (Recursivo)')

print()

print('Veamos la distribución acumulada')

print()

print('Pr(S<=1,000,000)')

x = 1000000

cumm_prob = 0

for i in aggregate_dist:

    if(i<=x):

        cumm_prob = cumm_prob + aggregate_dist[i]

print(cumm_prob,' (Convoluciones)')

cumm_prob = 0

for i in aggregate_dist_recursive:

    if(i<=x/1000):

        cumm_prob = cumm_prob + aggregate_dist_recursive[i]

print(cumm_prob,' (Recursivo)')

print()

print('Pr(S<=10,000,000)')

x = 10000000

cumm_prob = 0

for i in aggregate_dist:

    if(i<=x):

        cumm_prob = cumm_prob + aggregate_dist[i]

print(cumm_prob,' (Convoluciones)')

cumm_prob = 0

for i in aggregate_dist_recursive:

    if(i<=x/1000):

        cumm_prob = cumm_prob + aggregate_dist_recursive[i]

print(cumm_prob,' (Recursivo)')