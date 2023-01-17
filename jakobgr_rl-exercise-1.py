# Import libraries

import random as rd

import numpy as np

import matplotlib.pyplot as plt

from math import sqrt

randomSeed = "0"

rd.seed(randomSeed)
class Bandit():

    def __init__(self, distribution_method, lowerLimit, upperLimit):

        self.distribution_methode = distribution_method

        self.lowerLimit= lowerLimit

        self.upperLimit = upperLimit

        self.Q = 0

        self.N = 0

    def get_value(self):

        R = rd.uniform(self.lowerLimit, self.upperLimit)

        #R = rd.gauss(self.lowerLimit, self.upperLimit) #mu, sigma

        #R = rd.expovariate(2/(self.lowerLimit+self.upperLimit))

        self.N += 1

        self.Q += 1/self.N* (R - self.Q)

        return self.Q

    def setQ(self, Q):

        self.Q = Q

    

    def get_N(self):

        return self.N

    def get_averageQ(self):

        return self.Q

        
def epsilonGreedy(eps, n):

    #BanditList = [Bandit(0,-3,5), Bandit(0,2,8), Bandit(0,4,10), Bandit(0,3,7), Bandit(0,-3, 2)]

    BanditList = [Bandit(0,-3,5), Bandit(0,2,8), Bandit(0,4,10), Bandit(0,3,7), Bandit(0,-3, 2)]

    T = []

    M = []

    acc = 0

    for j in range(len(BanditList)):

        T.append(BanditList[j].get_averageQ())

    for i in range(n): 

        p = rd.random()

        if p > eps:

            k = T.index(max(T))

            T[k] = BanditList[k].get_value()

            acc+= T[k]

        else:

            x = rd.randint(0, len(BanditList)-1)

            T[x] = BanditList[x].get_value()

            acc+= T[x]

        M.append(acc/(i+1))

    print(T)

    return M

arg_eps = [0,0.01,0.05,0.2,0.5]

for i in arg_eps:

     rd.seed(randomSeed)

     plt.plot([k for k in range(1000)], epsilonGreedy(i,1000),label=str(i)) 

plt.legend(loc="lower right") 
def optimisticEpsGreedy(eps, n, Q):

    BanditList = [Bandit(0,-3,5), Bandit(0,2,8), Bandit(0,4,10), Bandit(0,3,7), Bandit(0,-3, 2)]

    for i in range(len(BanditList)):

        BanditList[i].setQ(Q)

    T = []

    M = []

    acc = 0

    for j in range(len(BanditList)):

        T.append(BanditList[j].get_averageQ())

    for i in range(n): 

        p = rd.random()

        if p > eps:

            k = T.index(max(T))

            T[k] = BanditList[k].get_value()

            acc+= T[k]

        else:

            x = rd.randint(0, len(BanditList)-1)

            T[x] = BanditList[x].get_value()

            acc+= T[x]

        M.append(acc/(i+1))

    print(T)

    return M
arg_opt_eps = [0,0.01,0.05,0.2,0.5]

for i in arg_opt_eps:

     rd.seed(randomSeed)

     plt.plot([k for k in range(1000)], optimisticEpsGreedy(i,1000, 5),label=str(i)) 

plt.legend(loc="lower right") 
arg_opt_eps = [0,1,2,4,5]

for i in arg_opt_eps:

     rd.seed(randomSeed)

     plt.plot([k for k in range(1000)], optimisticEpsGreedy(0.2,1000, i),label=str(i)) 

plt.legend(loc="lower right") 
def upperConfidenceBound(n, c):

    BanditList = [Bandit(0,-3,5), Bandit(0,2,8), Bandit(0,4,10), Bandit(0,3,7), Bandit(0,-3, 2)]

    T = []

    L = []

    M = []

    acc = 0

    for j in range(len(BanditList)):

        T.append(BanditList[j].get_averageQ())

        L.append(BanditList[j].get_averageQ())

    for i in range(n): 

        for j in range(len(BanditList)):

            if BanditList[j].get_N() == 0:

                L[j]= T[j] + 2**64

            else:

                L[j]= T[j] + c*sqrt(np.log(i+1)/BanditList[j].get_N())

        k = L.index(max(L))

        T[k] = BanditList[k].get_value()

        acc+= T[k]

        M.append(acc/(i+1))

    print(T)

    return M

arg_c = [0,0.01,0.05,2,5]

for i in arg_c:

     rd.seed(randomSeed)

     plt.plot([k for k in range(1000)], upperConfidenceBound(1000, i),label=str(i)) 

plt.legend(loc="lower right") 
# Plot Epsilon Greedy

rd.seed("0")

plt.plot([k for k in range(1000)], epsilonGreedy(0.2,1000),label='Epsilon Greedy') 

# Plot Optimistic Epsilon Greedy 

rd.seed("0")

plt.plot([k for k in range(1000)], optimisticEpsGreedy(0.2,1000, 6),label='Optmistic Epsilon Greedy')     

# Plot Upper Confidence Bound 

rd.seed("0")

plt.plot([k for k in range(1000)], upperConfidenceBound(1000, 0.02),label='UCB')



plt.legend(loc='best')