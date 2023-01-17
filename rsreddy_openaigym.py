# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))
import gym
import numpy as np


# Any results you write to the current directory are saved as output.
env = gym.make("FrozenLake-v0")
#Left, Right, Up, Down
env.action_space.n
# An observation is a way to descirbe the current state of the env of 16 tiles (S-Start, F-Frozen, H-Hole, G-Goal)
env.observation_space
# Initialize our Learning rate
alpha = 0.4
# Initalize our Discout factor for future rewards
gamma = 0.999
# Initialize our Q-table with 16 possible states in each we have an option to right, left, up, down
q_table = dict([(x, [1, 1, 1, 1]) for x in range(16)])
q_table

# observation will be the one with the maximum Q-value
def choose_action(observ):
    return np.argmax(q_table[observ])
# MDP to learn our envionrment
# running 10000 episodes
# An episode is complete if one of the following conditions are met:
    # 1. we reached G 
    # 2. we exhausted the 2500 iterations
    # 3. we fall into a hole
for i in range(10000):
    observ = env.reset()
    action = choose_action(observ)
    prev_observ = None
    prev_action = None
    
    t = 0
    # run 2500 time steps in each episode
    for t in range(2500):
        env.render()
        # excutes our action and return four different values 
        # observ: the next state because of our action
        # reward of that action
        # Boolean done - current episode is complete
        # info - contains more information 
        observ, reward, done, info = env.step(action)
        action = choose_action(observ)
        # once we executed the current action using env.step we can update the Q-values in our q_table provided we had a previous state 
        # retrieve a Q_value for the previous state and the previous action combination.  
        if not prev_observ is None:
            q_old = q_table[prev_observ][prev_action]
            q_new = q_old
            # calculate new Q-value for the prvious state-action combo which only contains the reward for the current action
            if done:
                q_new += alpha * (reward - q_old)
            else:
            # comput new Q-value : the value for the prvious state-action combo
                q_new += alpha * (reward + gamma + q_table[observ][action] - q_old)
            
            #update the Q-values for the previous state-action combo
            new_table = q_table[prev_observ]
            new_table[prev_action] = q_new
            
            q_table[prev_observ] = new_table
        # the current state now becomes the previous state and current action becomes previous action 
        prev_observ = observ
        prev_action = action
        
        # r=1 implies that the agent reached the goal tile.
        if done:
            print("Episode {} finished after {} timesteps with r={}.".format(i, t, reward))
            break
        
        
        
new_table
q_table