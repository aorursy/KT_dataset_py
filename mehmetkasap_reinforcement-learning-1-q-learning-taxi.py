# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import matplotlib.pyplot as plt

import random

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
# import gym library

# for more information please visit 

# - https://gym.openai.com/envs/Taxi-v3/

# - https://github.com/openai/gym/blob/master/gym/envs/toy_text/taxi.py



import gym
# create taxi environment

env = gym.make('Taxi-v2').env
# taxi environment has been created, to test this:

env
# to see this environment use: .render():

env.render()
env.reset() # reset environment and return random initial state
print('State space: ', env.observation_space) # will show us all possible states
print('Action space: ', env.action_space) # will show us all possible actions
# taxi row, taxi columnn, passenger location, destination respectively

state = env.encode(3,1,2,3) # return state value from 500 state space options

print('state number: ',state)
# lets see this location, we expect that taxi is at location 3x1, 

# passenger is at 2 (namely at Y) and destination is 3 (namely B)

env.s = state

env.render()
env.P[331]
env.reset() # reset first



time_step = 0

total_reward = 0

list_visualize = []



# this while loop is only one episopde

while True:

    

    time_step +=1

    

    # choose action

    action = env.action_space.sample() # take random sample from action space {0,1,2,3,4,5}

    

    # perform action and get reward

    state, reward, done, _ = env.step(action) # here state = next_state

    

    # total reward

    total_reward += reward

    

    # visualize

    list_visualize.append({'frame': env, 

                       'state': state,

                       'action': action,

                       'reward': reward,

                       'Total_Reward': total_reward

                      })

    # visualize all steps

    if time_step %100 == 0:

        env.render()

    

    if done: 

        break

    
print('number of iterations: ', time_step)

print('Total reward: ', total_reward)
# to see slowly how our taxi moves in the environment: 

'''

import time  



for c, value in enumerate(list_visualize):

    print(value["frame"].render())

    print('Time_step: ', c + 1)

    print('Action: ', value["action"])

    print('State: ', value["state"])

    print('Reward: ', value["reward"])

    print('Total_reward: ', value["Total_Reward"])

    time.sleep(1)

'''
# lets make 3 episodes (3 while loops)



for i in range(3):

    

    env.reset()

    new_time_step = 0

    

    while True:

    

        new_time_step +=1



        # choose action

        action = env.action_space.sample() # take random sample from action space {0,1,2,3,4,5}



        # perform action and get reward

        state, reward, done, _ = env.step(action) # here state = next_state



        # total reward

        total_reward += reward



        # visualize

        list_visualize.append({'frame': env, 

                           'state': state,

                           'action': action,

                           'reward': reward,

                           'Total_Reward': total_reward

                          })



        if done: 

            break

    print('number of iterations: ', new_time_step)

    print('Total reward: ', total_reward)

    print('-'*40)
# Q learning template



import gym

import numpy as np

import random

import matplotlib.pyplot as plt



env = gym.make('Taxi-v2').env



# Q table



q_table = np.zeros([env.observation_space.n, env.action_space.n]) # zeros(states, actions) and use .n to make it integer



# hyperparameters: alpha, gamma, epsilon



alpha = 0.1

gamma = 0.9

epsilon = 0.1



# plotting metrix



reward_list = []

dropouts_list = []



episode_number = 1000 # number of trainings 



for i in range(1, episode_number):

    

    # initialize environment

    

    state = env.reset() # For each episode, reset our environment and it returns new starting state 

    

    reward_count = 0

    dropouts = 0

    

    while True:

        

        # exploit OR explore in order to choose action (using The Epsilon-Greedy Algorithm)

        # epsilon = 0.1 means 10% explore and 90% exploit

    

        if random.uniform(0,1) < epsilon:

            action = env.action_space.sample()

        else:

            action = np.argmax(q_table[state]) # let state = 4, so action = argument where 

                                               # q_table[4] has max value

                                               # q_table is 500x5 matrix

                                               # q_table[4] = [0, 12, 32, 2, 5]

                                               # action = in third column corresponding to 32 let say south

        

        # action process and take reward / observation

        next_state, reward, done, _ = env.step(action) # .step(action) performs action and returns 4 parameteres 

                                                       # which are the next state, reward, false or true, end probaility

        

        # Q learning funtion update

        

        # Q(s,a)

        old_q_value = q_table[state,action]

        

        # max Q`(s`, a`)

        next_q_max = np.max(q_table[next_state])

        

        # find new Q value using Q funtion

        next_q_value = (1 - alpha) * old_q_value + alpha * (reward + gamma * next_q_max)

        

        # Q table update

        q_table[state,action] = next_q_value

        

        # update state

        state = next_state

        

        # find wrong drop-outs, no need for this actually, we will use it for visualization purposes 

        if reward == -10:

            dropouts += 1

        

        # find total reward

        reward_count += reward

        

        if done:

            break

    if i%10 == 0:

        dropouts_list.append(dropouts)

        reward_list.append(reward_count)

        print('Episode: {}, reward: {}, wrong dropouts: {}'.format(i, reward_count, dropouts))
fig, (axs1,axs2) = plt.subplots(1,2, figsize=(12, 6)) # create in 1 line 2 plots



axs1.plot(reward_list)

axs1.set_xlabel('episode*10')

axs1.set_ylabel('reward')

axs1.grid(True)



axs2.plot(dropouts_list)

axs2.set_xlabel('episode*10')

axs2.set_ylabel('wrong dropouts')

axs2.grid(True)



plt.show()
# now we have a good Q table that can help me

q_table
env.render() # our environment right now
at_this_state = env.encode(3,0,2,3) # taxi is at location 3x0, passenger is at location 2 and destination is 3

env.s = at_this_state

env.render()
q_table[at_this_state]
# let taxi be at 1x4, passenger in taxi (taxi will be green) (4), destination G (1) 

another_state = env.encode(1,4,4,1)

env.s = another_state

env.render()
q_table[another_state]