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
!pip install atari_py
import gym

from IPython.display import clear_output

import time

import matplotlib.pyplot as plt

env = gym.make("Taxi-v2")
current_state = env.reset()

print (current_state)

# print ('**************************************')

env.render()
env.action_space.n
env.env.s = 111

env.render()
state, reward, done, info = env.step(1)

env.render()

print(f"state={state}, reward={reward}, done={done}, info={info}")
state = env.reset()

counter = 0

reward = None

env.render()

total_reward = 0



while reward != 20:

    state, reward, done, info = env.step(env.action_space.sample())

    total_reward += reward

    counter += 1



env.render()



print(f'steps={counter},reward={total_reward}')
# from IPython.display import    

state = env.reset()

counter = 0

reward = 0

total_reward = 0



while reward != 20:

    state, reward, done, info = env.step(env.action_space.sample())

    total_reward += reward

    counter += 1

    

    clear_output(wait=True)

    env.render()

    print(f'steps={counter},reward={total_reward}')

    time.sleep(0.01)
Q = np.zeros([env.observation_space.n, env.action_space.n])

G = 0

alpha = 0.618
# init_state = env.reset()

for episode in range(1,10001):

    done = False

    G, reward = 0,0

#     state = init_state

    state = env.reset()

    steps = 0

    while done != True:

            steps += 1

            action = np.argmax(Q[state]) #1

            state2, reward, done, info = env.step(action) #2

            Q[state,action] += alpha * (reward + np.max(Q[state2]) - Q[state,action]) #3

            G += reward

            state = state2   

    if episode % 50 == 0:

        clear_output(True)

        print(f'Episode={episode}, Steps={steps}, Total Reward={G}')

        time.sleep(0.1)

# Q = np.zeros([env.observation_space.n, env.action_space.n])

G=0

state = env.reset()

env.render()

print(state)

steps = 0

# input()

done = False

# done = True

while done != True:

    action = np.argmax(Q[state]) #1

    state2, reward, done, info = env.step(action) #2

    Q[state,action] += alpha * (reward + np.max(Q[state2]) - Q[state,action]) #3

    G += reward

#     input()

    state = state2 

    clear_output(True)

    env.render()

    time.sleep(0.5)

    steps += 1

    

print(f'steps={steps}, reward={G}')

# print(env.registry.all())
env = gym.make("MsPacman-v0")

state = env.reset()

print(state.shape)

print(env.action_space.n)



for x in range(1,1000):

#     clear_output(True)

    env.step(env.action_space.sample())



plt.imshow(env.render('rgb_array'))
