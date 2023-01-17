%matplotlib inline



import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import scipy.stats as stats
uniform_data = stats.uniform.rvs(size=100000,  # Generate 100000 numbers

                                 loc = 0,      # From 0 

                                 scale=10)     # To 10
pd.DataFrame(uniform_data).plot(kind="density",  # Plot the distribution

                               figsize=(9,9),

                               xlim=(-1,11));
stats.uniform.cdf(x=2.5,         # Cutoff value (quantile) to check

                  loc=0,         # Distribution start

                  scale=10)      # Distribution end
stats.uniform.ppf(q=0.4,         # Probability cutoff

                  loc=0,         # Distribution start

                  scale=10)      # Distribution end
for x in range(-1,12,3):

    print("Density at x value " + str(x))

    print( stats.uniform.pdf(x, loc=0, scale=10) )   
import random



random.randint(0,10)     # Get a random integer in the specified range
random.choice([2,4,6,9]) # Get a random element from a sequence
random.random()          # Get a real number between 0 and 1
random.uniform(0,10)     # Get a real in the specified range
random.seed(12)  # Set the seed to an arbitrary value



print([random.uniform(0,10) for x in range(4)])



random.seed(12)  # Set the seed to the same value



print([random.uniform(0,10) for x in range(4)])
prob_under_minus1 = stats.norm.cdf(x= -1,  

                                loc = 0,               

                                scale= 1)     



prob_over_1 = 1 - stats.norm.cdf(x= 1,  

                                loc = 0,               

                                scale= 1) 



between_prob = 1-(prob_under_minus1+prob_over_1)



print(prob_under_minus1, prob_over_1, between_prob)


# Plot normal distribution areas*



plt.rcParams["figure.figsize"] = (9,9)

                                  

plt.fill_between(x=np.arange(-4,-1,0.01), 

                 y1= stats.norm.pdf(np.arange(-4,-1,0.01)) ,

                 facecolor='red',

                 alpha=0.35)



plt.fill_between(x=np.arange(1,4,0.01), 

                 y1= stats.norm.pdf(np.arange(1,4,0.01)) ,

                 facecolor='red',

                 alpha=0.35)



plt.fill_between(x=np.arange(-1,1,0.01), 

                 y1= stats.norm.pdf(np.arange(-1,1,0.01)) ,

                 facecolor='blue',

                 alpha=0.35)



plt.text(x=-1.8, y=0.03, s= round(prob_under_minus1,3))

plt.text(x=-0.2, y=0.1, s= round(between_prob,3))

plt.text(x=1.4, y=0.03, s= round(prob_over_1,3));
print( stats.norm.ppf(q=0.025) ) # Find the quantile for the 2.5% cutoff



print( stats.norm.ppf(q=0.975) ) # Find the quantile for the 97.5% cutoff
fair_coin_flips = stats.binom.rvs(n=10,        # Number of flips per trial

                                  p=0.5,       # Success probability

                                  size=10000)  # Number of trials



print( pd.crosstab(index="counts", columns= fair_coin_flips))



pd.DataFrame(fair_coin_flips).hist(range=(-0.5,10.5), bins=11);
biased_coin_flips = stats.binom.rvs(n=10,      # Number of flips per trial

                                  p=0.8,       # Success probability

                                  size=10000)  # Number of trials



# Print table of counts

print( pd.crosstab(index="counts", columns= biased_coin_flips))



# Plot histogram

pd.DataFrame(biased_coin_flips).hist(range=(-0.5,10.5), bins=11);
stats.binom.cdf(k=5,        # Probability of k = 5 successes or less

                n=10,       # With 10 flips

                p=0.8)      # And success probability 0.8
1 - stats.binom.cdf(k=8,        # Probability of k = 9 successes or more

                    n=10,       # With 10 flips

                    p=0.8)      # And success probability 0.8
stats.binom.pmf(k=5,        # Probability of k = 5 successes

                n=10,       # With 10 flips

                p=0.5)      # And success probability 0.5
stats.binom.pmf(k=8,        # Probability of k = 8 successes

                n=10,       # With 10 flips

                p=0.8)      # And success probability 0.8
random.seed(12)



flips_till_heads = stats.geom.rvs(size=10000,  # Generate geometric data

                                  p=0.5)       # With success prob 0.5



# Print table of counts

print( pd.crosstab(index="counts", columns= flips_till_heads))



# Plot histogram

pd.DataFrame(flips_till_heads).hist(range=(-0.5,max(flips_till_heads)+0.5)

                                    , bins=max(flips_till_heads)+1);
first_five = stats.geom.cdf(k=5,   # Prob of success in first 5 flips

                           p=0.5)



1 - first_five
stats.geom.pmf(k=2,   # Prob of needing exactly 2 flips to get first success

              p=0.5)
# Get the probability of waiting more than 1 time unit before a success



prob_1 = stats.expon.cdf(x=1,        

                         scale=1)  # Arrival rate



1 - prob_1
plt.fill_between(x=np.arange(0,1,0.01), 

                 y1= stats.expon.pdf(np.arange(0,1,0.01)) ,

                 facecolor='blue',

                 alpha=0.35)



plt.fill_between(x=np.arange(1,7,0.01), 

                 y1= stats.expon.pdf(np.arange(1,7,0.01)) ,

                 facecolor='red',

                 alpha=0.35)





plt.text(x=0.3, y=0.2, s= round(prob_1,3))

plt.text(x=1.5, y=0.08, s= round(1 - prob_1,3));
random.seed(12)



arrival_rate_1 = stats.poisson.rvs(size=10000,  # Generate Poisson data

                                   mu=1 )       # Average arrival time 1



# Print table of counts

print( pd.crosstab(index="counts", columns= arrival_rate_1))



# Plot histogram

pd.DataFrame(arrival_rate_1).hist(range=(-0.5,max(arrival_rate_1)+0.5)

                                    , bins=max(arrival_rate_1)+1);
random.seed(12)



arrival_rate_10 = stats.poisson.rvs(size=10000,  # Generate Poisson data

                                   mu=10 )       # Average arrival time 10



# Print table of counts

print( pd.crosstab(index="counts", columns= arrival_rate_10))



# Plot histogram

pd.DataFrame(arrival_rate_10).hist(range=(-0.5,max(arrival_rate_10)+0.5)

                                    , bins=max(arrival_rate_10)+1);
stats.poisson.cdf(k=5,     # Check the probability of 5 arrivals or less

                  mu=10)   # With arrival rate 10
stats.poisson.pmf(k=10,     # Check the prob f exactly 10 arrivals

                  mu=10)    # With arrival rate 10