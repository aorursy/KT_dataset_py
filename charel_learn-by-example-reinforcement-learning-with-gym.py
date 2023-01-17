import gym # openAi gym

from gym import envs

import numpy as np 

import datetime

import keras 

import tensorflow as tf

import matplotlib.pyplot as plt

%matplotlib inline

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from time import sleep



from rl.agents.dqn import DQNAgent

from rl.policy import LinearAnnealedPolicy, BoltzmannQPolicy, EpsGreedyQPolicy

from rl.memory import SequentialMemory

from rl.core import Processor

from rl.callbacks import FileLogger, ModelIntervalCheckpoint

print("OK")
print(envs.registry.all())

env = gym.make('Taxi-v2')

env.reset()

env.render()
# Let's first do some random steps in the game so you see how the game looks like



rew_tot=0

obs= env.reset()

env.render()

for _ in range(6):

    action = env.action_space.sample() #take step using random action from possible actions (actio_space)

    obs, rew, done, info = env.step(action) 

    rew_tot = rew_tot + rew

    env.render()

#Print the reward of these random action

print("Reward: %r" % rew_tot)    

    
# action space has 6 possible actions, the meaning of the actions is nice to know for us humans but the neural network will figure it out

print(env.action_space)

NUM_ACTIONS = env.action_space.n

print("Possible actions: [0..%a]" % (NUM_ACTIONS-1))



print(env.observation_space)

print()

env.env.s=42 # some random number, you might recognize it

env.render()

env.env.s = 222 # and some other

env.render()
# Value iteration algorithem

NUM_ACTIONS = env.action_space.n

NUM_STATES = env.observation_space.n

V = np.zeros([NUM_STATES]) # The Value for each state

Pi = np.zeros([NUM_STATES], dtype=int)  # Our policy with we keep updating to get the optimal policy

gamma = 0.9 # discount factor

significant_improvement = 0.01



def best_action_value(s):

    # finds the highest value action (max_a) in state s

    best_a = None

    best_value = float('-inf')



    # loop through all possible actions to find the best current action

    for a in range (0, NUM_ACTIONS):

        env.env.s = s

        s_new, rew, done, info = env.step(a) #take the action

        v = rew + gamma * V[s_new]

        if v > best_value:

            best_value = v

            best_a = a

    return best_a



iteration = 0

while True:

    # biggest_change is referred to by the mathematical symbol delta in equations

    biggest_change = 0

    for s in range (0, NUM_STATES):

        old_v = V[s]

        action = best_action_value(s) #choosing an action with the highest future reward

        env.env.s = s # goto the state

        s_new, rew, done, info = env.step(action) #take the action

        V[s] = rew + gamma * V[s_new] #Update Value for the state using Bellman equation

        Pi[s] = action

        biggest_change = max(biggest_change, np.abs(old_v - V[s]))

    iteration += 1

    if biggest_change < significant_improvement:

        print (iteration,' iterations done')

        break
# Let's see how the algorithm solves the taxi game

rew_tot=0

obs= env.reset()

env.render()

done=False

while done != True: 

    action = Pi[obs]

    obs, rew, done, info = env.step(action) #take step using selected action

    rew_tot = rew_tot + rew

    env.render()

#Print the reward of these actions

print("Reward: %r" % rew_tot)  
NUM_ACTIONS = env.action_space.n

NUM_STATES = env.observation_space.n

Q = np.zeros([NUM_STATES, NUM_ACTIONS]) #You could also make this dynamic if you don't know all games states upfront

gamma = 0.9 # discount factor

alpha = 0.9 # learning rate

for episode in range(1,1001):

    done = False

    rew_tot = 0

    obs = env.reset()

    while done != True:

            action = np.argmax(Q[obs]) #choosing the action with the highest Q value 

            obs2, rew, done, info = env.step(action) #take the action

            Q[obs,action] += alpha * (rew + gamma * np.max(Q[obs2]) - Q[obs,action]) #Update Q-marix using Bellman equation

            #Q[obs,action] = rew + gamma * np.max(Q[obs2]) # same equation but with learning rate = 1 returns the basic Bellman equation

            rew_tot = rew_tot + rew

            obs = obs2   

    if episode % 50 == 0:

        print('Episode {} Total Reward: {}'.format(episode,rew_tot))
# Let's see how the algorithm solves the taxi game by following the policy to take actions delivering max value



rew_tot=0

obs= env.reset()

env.render()

done=False

while done != True: 

    action = np.argmax(Q[obs])

    obs, rew, done, info = env.step(action) #take step using selected action

    rew_tot = rew_tot + rew

    env.render()

#Print the reward of these actions

print("Reward: %r" % rew_tot)  
# Let's show the game layout



env = gym.make('FrozenLake-v0')

rew_tot=0

obs= env.reset()

env.render()

env = gym.make('FrozenLake-v0')

env.reset()

NUM_ACTIONS = env.action_space.n

NUM_STATES = env.observation_space.n

Q = np.zeros([NUM_STATES, NUM_ACTIONS]) #You could also make this dynamic if you don't know all games states upfront

gamma = 0.95 # discount factor

alpha = 0.01 # learning rate

epsilon = 0.1 #

for episode in range(1,500001):

    done = False

    obs = env.reset()

    while done != True:

        if np.random.rand(1) < epsilon:

            # exploration with a new option with probability epsilon, the epsilon greedy approach

            action = env.action_space.sample()

        else:

            # exploitation

            action = np.argmax(Q[obs])

        obs2, rew, done, info = env.step(action) #take the action

        Q[obs,action] += alpha * (rew + gamma * np.max(Q[obs2]) - Q[obs,action]) #Update Q-marix using Bellman equation

        obs = obs2   

        

    if episode % 5000 == 0:

        #report every 5000 steps, test 100 games to get avarage point score for statistics and verify if it is solved

        rew_average = 0.

        for i in range(100):

            obs= env.reset()

            done=False

            while done != True: 

                action = np.argmax(Q[obs])

                obs, rew, done, info = env.step(action) #take step using selected action

                rew_average += rew

        rew_average=rew_average/100

        print('Episode {} avarage reward: {}'.format(episode,rew_average))

        

        if rew_average > 0.8:

            # FrozenLake-v0 defines "solving" as getting average reward of 0.78 over 100 consecutive trials.

            # Test it on 0.8 so it is not a one-off lucky shot solving it

            print("Frozen lake solved")

            break

 
# Let's see how the algorithm solves the frozen-lakes game



rew_tot=0.

obs= env.reset()

done=False

while done != True: 

    action = np.argmax(Q[obs])

    obs, rew, done, info = env.step(action) #take step using selected action

    rew_tot += rew

    env.render()



print("Reward:", rew_tot)  