# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
#Install the OpenAI Gym
#!pip install cmake 'gym[atari]' scipy
!pip install gym[atari]
!apt-get install -y python-opengl

import gym #Import the gym module
env = gym.make("FrozenLake-v0").env #We load the Taxi game environment.
#We are using the .env on the end of make to avoid training stopping at 200 iterations, which is the default

env.reset() # reset environment to a new, random state
env.render() #Renders one frame of the environment

"""
0 = south
1 = north
2 = east
3 = west
4 = pickup
5 = dropoff
"""
#Create a Q table with rows = number of states, columns = number of actions, and initialise all values to zero
q_table = np.zeros([env.observation_space.n, env.action_space.n])


import random
from IPython.display import clear_output

# Hyperparameters
alpha = 0.6
gamma = 0.8
epsilon = 0.1

# For plotting metrics
all_epochs = []
all_penalties = []

#Run 100,000 episodes
for i in range(1, 100001):
    state = env.reset() #At the start of each episode we reset the taxi enviroment and pick a random state to start in.

    epochs, penalties, reward, = 0, 0, 0
    done = False
    
    while not done:
        if random.uniform(0, 1) < epsilon:
            action = env.action_space.sample() # Explore action space
        else:
            action = np.argmax(q_table[state]) # Exploit learned values

        next_state, reward, done, info = env.step(action) 
        
        old_value = q_table[state, action]
        next_max = np.max(q_table[next_state])
        
        new_value = (1 - alpha) * old_value + alpha * (reward + gamma * next_max)
        q_table[state, action] = new_value

        if reward == -20:
            penalties += 1

        state = next_state
        epochs += 1
        
    if i % 100 == 0:
        clear_output(wait=True)
        print(f"Episode: {i}")

print("Training finished.\n")
#View an arbitrary state in the state space to see the updated q-values for each action
q_table[15]
"""Evaluate agent's performance after Q-learning"""

total_epochs, total_penalties = 0, 0
episodes = 100

for _ in range(episodes):
    state = env.reset()
    epochs, penalties, reward = 0, 0, 0
    
    done = False
    
    while not done:
        action = np.argmax(q_table[state])
        state, reward, done, info = env.step(action)

        if reward == -20:
            penalties += 1

        epochs += 1

    total_penalties += penalties
    total_epochs += epochs

print(f"Results after {episodes} episodes:")
print(f"Average timesteps per episode: {total_epochs / episodes}")
print(f"Average penalties per episode: {total_penalties / episodes}")