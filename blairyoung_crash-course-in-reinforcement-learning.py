# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import numpy as np

import gym

import random

import math
env = gym.make("Taxi-v2")
env.reset()

env.render()
# Pick a random state and render it

env.env.s = 420

env.render()
# Remember right corresponds to 2

env.step(2)

env.render()
env.step(2)
state, reward, done, info = env.step(2)

print('state: {}'.format(state))

print('reward: {}'.format(reward))

print('done: {}'.format(done))

print('info: {}'.format(info))
state = env.reset()

reward = None

steps = 0



while reward != 20:

    state, reward, done, info = env.step(env.action_space.sample())

    steps += 1



print('Random driving took {} steps to complete a journey'.format(steps))
random_driving_store = []

for i in range(1,100):

    state = env.reset()

    reward = None

    steps = 0



    while reward != 20:

        state, reward, done, info = env.step(env.action_space.sample())

        steps += 1



    random_driving_store.append(steps)

    

from matplotlib import pyplot as plt

import numpy as np



plt.figure()

plt.hlines(0.5,0.5,2)  # Draw a horizontal line

plt.eventplot(random_driving_store, orientation='horizontal', colors='k')



plt.show()

print('Average number of steps for a random drive {}'.format(np.mean(random_driving_store)))
state_size = env.observation_space.n

action_size = env.action_space.n



print('number of possible states: {}'.format(state_size))

print('number of possible actions: {}'.format(action_size))
Q = np.zeros([state_size, action_size])

Q
total_reward = 0

learning_rate = 0.7
done = False

total_reward, reward = 0,0

state = env.reset()

while done != True: # Keeps making actions until episode completes

        action = np.argmax(Q[state]) # Finds the action with the greatest reward. TIP each state is a row in the Q-table, find the best action at this state by finding the max value

        new_state, reward, done, info = env.step(action) #Takes the action with the greatest reward

        Q[state, action] += learning_rate * (reward + np.max(Q[new_state]) - Q[state, action]) # Updates our Q-table based on the state and actions. TIP If your stuck have a look at this pseudo code below

        total_reward += reward # Update our total reward

        state = new_state # Update our current state

#         env.render() # Print the current agent-environment interaction

print('Total reward for this episode: {}'.format(total_reward))





# New Q value = Current Q value + learning rate * (Reward + (maximum value of new state) — Current Q value )
Q = np.zeros([state_size, action_size])



total_reward = 0

learning_rate = 0.7



for episode in range(1,2001):

    done = False

    total_reward, reward = 0,0

    state = env.reset()

    while done != True:

        action = np.argmax(Q[state])

        new_state, reward, done, info = env.step(action)

        Q[state, action] += learning_rate * (reward + np.max(Q[new_state]) - Q[state, action])

        total_reward += reward

        state = new_state   

    if episode % 50 == 0:

        print('Episode {} Total Reward: {}'.format(episode, total_reward))
import numpy as np

import matplotlib.pyplot as plt

import pandas as pd

import random
# Importing the dataset ../input/ad-datacsv/Ads_Optimisation.csv

dataset = pd.read_csv('../input/ad-datacsv/Ads_Optimisation.csv')

print('First user clicked Ad 1, 5 and 9')

dataset.head(1)
number_of_users = 10000

number_of_ads = 10

ads_selected = []

total_reward = 0

for user in range(0, number_of_users):

    ad_picked = random.randrange(number_of_ads)

    ads_selected.append(ad_picked)

    reward = dataset.values[user, ad_picked]

    total_reward = total_reward + reward
total_reward
pd.Series(ads_selected).value_counts(normalize=True)
pd.Series(ads_selected).value_counts(normalize=True).plot(kind='bar')
def UCB_multi_armed_bandit(number_of_users, number_of_ads, dataset):



    ads_selected = []

    numbers_of_selections = [0] * number_of_ads

    sums_of_reward = [0] * number_of_ads

    total_reward = 0



    for user in range(0, number_of_users):

        ad = 0

        max_upper_bound = 0

        for i in range(0, number_of_ads):

            if (numbers_of_selections[i] > 0):

                average_reward = sums_of_reward[i] / numbers_of_selections[i]

                delta_i = math.sqrt(2 * math.log(user+1) / numbers_of_selections[i])

                upper_bound = average_reward + delta_i

            else:

                upper_bound = 1e400

            if upper_bound > max_upper_bound:

                max_upper_bound = upper_bound

                ad = i

        ads_selected.append(ad)

        numbers_of_selections[ad] += 1

        reward = dataset.values[user, ad]

        sums_of_reward[ad] += reward

        total_reward += reward

    return ads_selected, total_reward
ads_selected, total_rewards = UCB_multi_armed_bandit(10000, 10, dataset)
pd.Series(ads_selected).head(1500).value_counts(normalize=True).plot(kind='bar')
total_rewards
dataset.sum()/len(dataset)
total_episodes = 20000        # Total episodes

total_test_episodes = 100     # Total test episodes

max_steps = 99                # Max steps per episode



learning_rate = 0.7           # Learning rate

gamma = 0.618                 # Discounting rate



# Exploration parameters

epsilon = 1.0                 # Exploration rate

max_epsilon = 1.0             # Exploration probability at start

min_epsilon = 0.01            # Minimum exploration probability 

decay_rate = 0.01             # Exponential decay rate for exploration prob





qtable = np.zeros((state_size, action_size))


# List of rewards

rewards = []



# 2 For life or until learning is stopped

for episode in range(total_episodes):

    # Reset the environment

    state = env.reset()

    step = 0

    done = False

    total_rewards = 0

    

    for step in range(max_steps):

        # 3. Choose an action a in the current world state (s)

        ## First we randomize a number

        exp_exp_tradeoff = random.uniform(0, 1)

        

        ## If this number > greater than epsilon --> exploitation (taking the biggest Q value for this state)

        if exp_exp_tradeoff > epsilon:

            action = np.argmax(qtable[state,:])



        # Else doing a random choice --> exploration

        else:

            action = env.action_space.sample()



        # Take the action (a) and observe the outcome state(s') and reward (r)

        new_state, reward, done, info = env.step(action)



        # Update Q(s,a):= Q(s,a) + lr [R(s,a) + gamma * max Q(s',a') - Q(s,a)]

        # qtable[new_state,:] : all the actions we can take from new state

        qtable[state, action] = qtable[state, action] + learning_rate * (reward + gamma * np.max(qtable[new_state, :]) - qtable[state, action])

        

        total_rewards += reward

        

        # Our new state is state

        state = new_state

        

        # If done (if we're dead) : finish episode

        if done == True: 

            break

        

    # Reduce epsilon (because we need less and less exploration)

    epsilon = min_epsilon + (max_epsilon - min_epsilon)*np.exp(-decay_rate*episode) 

    rewards.append(total_rewards)



print ("Score over time: " +  str(sum(rewards)/total_episodes))

print(qtable)
env.reset()



for episode in range(5):

    state = env.reset()

    step = 0

    done = False

    print("****************************************************")

    print("EPISODE ", episode)



    for step in range(max_steps):

        

        # Take the action (index) that have the maximum expected future reward given that state

        action = np.argmax(qtable[state,:])

        

        new_state, reward, done, info = env.step(action)

        

        if done:

            # Here, we decide to only print the last state (to see if our agent is on the goal or fall into an hole)

            env.render()

            

            # We print the number of step it took.

            print("Number of steps", step)

            break

        state = new_state

env.close()
env = gym.make('FrozenLake-v0')
gym.envs.registry.all()