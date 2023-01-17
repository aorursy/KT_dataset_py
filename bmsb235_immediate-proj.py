import matplotlib.pyplot as plt

import math

import random

import statistics as st



#lambdas = random.sample(range(1, 100), 5)

lambdas = [5, 19, 26, 38, 45, 52, 65, 77, 86, 94]

lambdas.sort()



l = 0



cols = ['r--', 'b--', 'g--', 'y--', 'm--', 'c--', 'r-.', 'b-.', 'g-.', 'y-.']

allruns = []

allmed = []

allmean = []

allvar = []

allskew = []

plt.figure(figsize=(10,10))



for k in range(100):

    allruns.append(k)



for li in range(len(lambdas)):

    

    l = lambdas[li]

    

    runs = []

    poisson = []



    for k in range(100):



        x = math.factorial(k)

        f = (l**k)*math.exp(-l)/(x)



        runs.append(k)

        poisson.append(f)

            

    allmed.append(l+(1/3)-(0.02)/l)

    allvar.append(l)

    allmean.append(l)

    allskew.append(l**0.5)

    plt.plot(runs, poisson, cols[li])# label=str(lambdas[li]))

    

    #plt.show()





plt.gca().legend(lambdas)

#plt.show()



print('Lambdas\t','Mean\t','Variance\t','Median\t\t','Skewness\n')

for i in range(len(lambdas)):

    print(lambdas[i], '\t', allmean[i], '\t', allvar[i], '\t', allmed[i], '\t', allskew[i])
