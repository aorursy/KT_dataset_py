import matplotlib.pyplot as plt

import numpy as np

import random, collections
x=np.arange(0,1,0.01)

plt.plot(x, (1-x)*(x))

plt.xlabel("Team 19's cracking probability")

plt.ylabel("Team 19's victory probability")

plt.xlim(0,1)

plt.rcParams['figure.figsize'] = [10, 10]

plt.show()
x=np.arange(0,1,0.01)

plt.plot(x, (1-x)*(x**2))

plt.xlabel("Team 18's cracking probability")

plt.ylabel("Team 18's victory probability")

plt.xlim(0,1)

plt.rcParams['figure.figsize'] = [10, 10]

plt.show()
x=np.arange(0,1,0.001)

plt.plot(x, (1-x)*(x**19))

plt.xlabel("Team 1's cracking probability")

plt.ylabel("Team 1's victory probability")

plt.xlim(0,1)

#plt.legend(['game %', 'set %', "women's match%", "men's match%", "women's tournament%", "men's tournament%"], loc='upper left')

plt.rcParams['figure.figsize'] = [10, 10]

plt.show()
plotlegend=[]

p=np.arange(0,1,0.01)

for n in range(20):

    plt.plot(x,(1-x)*(x**n))

    plotlegend.append('Team '+str(20-n))

plt.xlabel("Cracking probability")

plt.ylabel("Victory probability")

plt.xlim(0,1)

plt.legend(plotlegend, loc='upper right')

plt.rcParams['figure.figsize'] = [10, 10]

plt.show()
total_trials=1000000

winners=[]

for trial in range(total_trials):

    previous_best=0

    winning_team=0

    for _ in range(20):

        team=_+1

        n=20-team

        target_c_prob=n/(n+1)

        c_prob=max(previous_best,target_c_prob)

        if random.random()>c_prob:

            winning_team=team

            previous_best=c_prob

    winners.append(winning_team)



wincounts=collections.Counter(winners)



vp={}

for _ in range(20):

    team=_+1

    vp[team]=str(100*wincounts[team]/total_trials)+"%"

vp