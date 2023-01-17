# Import the dependencies

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

sns.set_style("white")

print("Import completed")
# Quick approximation of some probability densities by drawing 500.000 samples

sns.distplot(np.random.normal(4,1.5,500000),hist=False, label='first throws, $\mu$=4, $\sigma$=1.5');

sns.distplot(np.random.normal(7,1,500000),hist=False, label='getting better, $\mu$=7, $\sigma$=1');

sns.distplot(np.random.normal(10,0.5,500000),hist=False, label='getting good, $\mu$=10, $\sigma$=0.5');

plt.legend();
#Input parameters

perc1dice=0.40



#Calculate dice throws

N=5000 # amount of dice throws

N1=int(N*perc1dice) # amount of times six sided dice is thrown

N2=int(N*(1-perc1dice))  # amount of times 2 six sided dice is thrown

random_seed=1234567

np.random.seed(random_seed)



#Build result table with some simple headers 

results=np.zeros((3,14), dtype=int) # table where results are stored

for count in range(1, 13):

    results[0,count]=count

results[1,0]=1

results[2,0]=2

results[1,13]=N1

results[2,13]=N2



# Throw N times six sided dice

for count in range(0, N1):

    throw=np.random.randint(1,7)

    results[1,throw] += 1



# Throw M times 2 six sided dice

for count in range(0, N2):

    throw=np.random.randint(1,7)+np.random.randint(1,7)

    results[2,throw] += 1



# Print the results

print(results)

print()

#Calculate P(x=2) given p(y=observation)

for observation in range(1,13):

    nominator=(results[2,observation]/N2)*(N2/N)

    denominator=(results[1,observation]/N1)*(N1/N)+(results[2,observation]/N2)*(N2/N)

    print('Probability of 2 dice given observation ', observation, ' is ' , int(100*nominator/denominator),'%')



# Plot the probability distributions:

fig, axes = plt.subplots(2,2)

plt.subplots_adjust(wspace=0.35, hspace=0.35)

fig.suptitle('Probability distributions')

sns.barplot(results[0, 2:13], results[2,2:13]/N2, color='blue' , ax=axes[0,0])

axes[0,0].set_ylabel('Prob')

axes[0,0].grid(1)

axes[0,0].set_title('$p(Observation \mid dice=2)$')

sns.barplot(results[0, 1:7], results[1,1:7]/N1, color='blue' , ax=axes[0,1])

axes[0,1].set_ylabel('Prob')

axes[0,1].grid(1)

axes[0,1].set_title('$p(Observation \mid dice=1)$')

sns.barplot(results[0, 1:3], [N1/N,N2/N], color='blue' , ax=axes[1,0])

axes[1,0].set_ylabel('Prob')

axes[1,0].grid(1)

axes[1,0].set_title('$p(dice)$')

sns.barplot(results[0, 1:13], (results[1, 1:13]+results[2, 1:13])/N, color='blue' , ax=axes[1,1])

axes[1,1].set_ylabel('Prob')

axes[1,1].grid(1)

axes[1,1].set_title('$p(Observation)$');


