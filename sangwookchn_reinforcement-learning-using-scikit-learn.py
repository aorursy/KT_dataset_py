#implementing Upper Confidence Bound (UCB)

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math

dataset = pd.read_csv('../input/Ads_CTR_Optimisation.csv')
dataset.head(10)
#UCB needs to be implemented from scratch without using any package, as there is no easy library to use.

N = 10000
d = 10
ads_selected = []
numbers_of_selections = [0] * d
sums_of_rewards = [0] * d
total_reward = 0

for n in range(0, N):
    ad = 0
    max_upper_bound = 0
    for i in range(0, d):
        if numbers_of_selections[i] > 0:
            # 3 lines below is the algorithm shown above
            average_reward = sums_of_rewards[i] / numbers_of_selections[i]
            delta_i = math.sqrt(3/2 * math.log(n + 1) / numbers_of_selections[i])
            upper_bound = average_reward + delta_i
        else:
            upper_bound = 1e400 #makes this large so that the first round gives every category a chance 
        if upper_bound > max_upper_bound:
            ad = i
            max_upper_bound = upper_bound
    ads_selected.append(ad)
    numbers_of_selections[ad] = numbers_of_selections[ad] + 1
    reward = dataset.values[n, ad]
    sums_of_rewards[ad] = sums_of_rewards[ad] + reward
    total_reward = total_reward + reward

#Visualizing the result
plt.hist(ads_selected) #histogram of number of times each ad is clicked
plt.title('Histogram of ads_selected')
plt.xlabel('Ad No')
plt.ylabel('Number of times each add is selected')
plt.show()

#to view the reward
print(total_reward)
            
import random

N = 10000
d = 10
ads_selected = []
numbers_of_rewards_1 = [0] * d
numbers_of_rewards_0 = [0] * d #number of 0 rewards for each ad
total_reward = 0

for n in range(0, N):
    ad = 0
    max_random = 0 #maximum random draw
    for i in range(0, d):
        random_beta = random.betavariate(numbers_of_rewards_1[i] + 1, numbers_of_rewards_0[i] + 1)
        if random_beta > max_random:
            ad = i
            max_random = random_beta
    ads_selected.append(ad)
    reward = dataset.values[n, ad]
    if reward == 1:
        numbers_of_rewards_1[ad] = numbers_of_rewards_1[ad] + 1
    else:
        numbers_of_rewards_0[ad] = numbers_of_rewards_0[ad] + 1
    total_reward = total_reward + reward

plt.hist(ads_selected) #histogram of number of times each ad is clicked
plt.title('Histogram of ads_selected')
plt.xlabel('Ad No')
plt.ylabel('Number of times each add is selected')
plt.show()

#to view the reward
print(total_reward)
