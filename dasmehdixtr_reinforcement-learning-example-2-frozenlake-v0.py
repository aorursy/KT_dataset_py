import gym

import numpy as np

import random

import matplotlib.pyplot as plt
env = gym.make('FrozenLake-v0')



from gym.envs.registration import register

# To make enviroment deterministic uncomment this area

#register(

#        id='FrozenLakeNotSlippery-v0',

#        entry_point='gym.envs.toy_text:FrozenLakeEnv',

#        kwargs={'map_name' : '4x4', 'is_slippery':False},

#        max_episode_steps = 100,

#        reward_threshold = 0.78)



q_table = np.zeros([env.observation_space.n, env.action_space.n])
alpha = 0.1

gamma = 0.9 

epsilon = 0.1

# plotting metric

reward_list = []

dropout_list = []
episode_number = 30000



for i in range(1,episode_number):

    

    # init enviroment

    state = env.reset()

    

    reward_count = 0

    dropouts = 0

    

    while True:

        

        # exploit vs explore to find action epsilon 0.1 => %10 explore %90 explotit

        if random.uniform(0,1) < epsilon:

            action = env.action_space.sample()

        else:

            action = np.argmax(q_table[state])

            

        # action process and take reward/ take observation

        next_state, reward, done, _ = env.step(action)

        

        # q learning funt

        

        old_value = q_table[state, action]  #old value

        next_max = np.max(q_table[next_state]) #next max

        

        next_value = (1-alpha)*old_value + alpha*(reward + gamma*next_max)

        # q table update

        q_table[state,action] = next_value

        

        # update state

        state = next_state

        

        reward_count  += reward

    

        if done:

            break

        

    if i%10 == 0:

        dropout_list.append(dropouts)

        reward_list.append(reward_count)

        print("Episode: {}, reward {}".format(i, reward_count))
plt.plot(reward_list)