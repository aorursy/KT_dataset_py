import numpy as np # linear algebra

import pandas as pd

from scipy.stats import binom
x = np.array([0, 1000, 5000, 10000, 20000, 50000, 100000, 500000, 1000000])

p = np.array([.05,.1,.2,.3,.15,.1,.05,.025,.025])

m = 10
#Determina el soporte de S

soporte = x

for i in range(m-1):

    soporte = np.add.outer(soporte,x)

    soporte = list(set([x1 for x2 in soporte for x1 in x2]))

soporte.sort()

soporte = np.asarray(soporte)
#Vectores de probabilidad conjunta para N = 0,1,...m

v0 = np.eye(1,len(soporte),0) * binom.pmf(0,m,.3)

v0 = v0[0]

v1 = np.array([p[list(x).index(w)] if w in x else 0 for w in soporte]) * binom.pmf(1,m,.3)

dist = pd.DataFrame(np.transpose(v0), index = soporte, columns = ['S'])

datos = {'N0':v0,'N1':v1}

dist = pd.DataFrame(datos, index=soporte)

x2 = x

p2 = p

for n in range(m-1):    

    x2 = np.add.outer(x2,x)

    p2 = np.multiply.outer(p2,p)

    v2 = []

    for i in range(len(soporte)):

        prob = p2[x2 == soporte[i]] 

        if prob.size == 0:

            prob = [0]

        v2.append(np.sum(prob))

    v2 = np.array(v2)

    v2 = v2*binom.pmf(n+2,m,.3)

    dist['N' + str(n+2)] = v2

    #Esta sección solo pretende reducir el costo computacional

    x2f = list(set([w for z in x2 for w in z]))

    x2f = np.array(x2f)

    x2f.sort()

    p2f = []

    for i in range(len(x2f)):

            p2f.append(np.sum(p2[x2 == x2f[i]]))

    x2 = x2f

    p2 = p2f

#Tabla de probabilidades conjuntas

dist
#Momentos

ps = dist['N0'] + dist['N1'] + dist['N2'] + dist['N3'] + dist['N4'] + dist['N5'] + dist['N6'] + dist['N7'] + dist['N8'] + dist['N9'] + dist['N10'] 

media =sum(soporte*ps)

var = sum(soporte*soporte*ps) - media**2

asimetria = sum(soporte**3*ps) - 3*media*var - media**3

coefAsimetria = asimetria/var**(3/2)

kurtosis = sum(soporte**4*ps)-4*media*asimetria+18*media**2*var+7*sum(soporte**4*ps)

coefKurtosis = kurtosis/var**2
print("Media E[N]*E[X]: ", sum(p*x)*m*.3)

print("Media con tabla de distribucion: ", media)

print("Coeficiente de asimetría: ", coefAsimetria)

print("Coeficiente de kurtosis: ", coefKurtosis)