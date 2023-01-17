# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import gym # for environment

import random



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
env = gym.make('FrozenLake-v0').env

env.render()
env.observation_space
env.action_space
act = env.action_space.sample()

env.step(act)
# to create deterministic (not slippery) envirenment we can use the codes below

#from gym.envs.registration import register

#register (

#    id = 'FrozenLakeNotSlippery-v0',

#    entry_point = 'gym.envs.toy_text:FrozenLakeEnv', 

#    kwargs = {'map_name' : '4x4', 'is slippery' : False},

#    max_episode_steps = 100,

#    reward_threshold = 0.78, # optimum .8196

#)
# hyperparameters

alpha = 0.08 # learning rate

gamma = 0.95 # discount rate

epsilon = 0.13 # 



# initialization

time_step = 0

episodes = 50000



# plotting metrix

reward_list = []



# initialize Q table

q_table = np.zeros([env.observation_space.n, env.action_space.n])



for i in range(1,episodes):

    

    state = env.reset() # reset environment for each episode

    time_step += 1 

    reward_count = 0

    

    # update Q funtion

    while True: 

        

        # Choose action using epsilon greedy

        if random.uniform(0,1) < epsilon:

            action = env.action_space.sample()

        else:

            action = np.argmax(q_table[state])

            

        # action process and take reward / observation

        next_state, reward, done, _ = env.step(action)

        

        old_q_value = q_table[state, action]

        next_max_q_value = np.max(q_table[next_state])

        

        # update Q value

        new_q_value = (1 - alpha) * old_q_value + alpha * (reward + gamma * next_max_q_value)

        

        # update Q table

        q_table[state, action] = new_q_value

        

        # update state

        state = next_state

        

        # we will use it for visualization purposes 

        if reward == -10:

            dropouts += 1

        

        # find total reward

        reward_count += reward

        

        if done:

            break

    

    if i%100 == 0:

        reward_list.append(reward_count)

        print('Episode: {}, Reward: {}'.format(i, reward_count) )

        
import matplotlib.pyplot as plt

plt.scatter(np.arange(len(reward_list)), reward_list)