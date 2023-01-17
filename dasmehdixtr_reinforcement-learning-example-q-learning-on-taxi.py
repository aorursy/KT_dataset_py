import gym

import numpy as np

import matplotlib.pyplot as plt

import random
env = gym.make("Taxi-v3").env



# Q table 500 sample(observation space = 5*5*5*4 = 500) - 6 action (left,right, up, down, pickup, dropout)

q_table = np.zeros([env.observation_space.n,env.action_space.n])


alpha = 0.1

gamma = 0.9 

epsilon = 0.1

# plotting metric

reward_list = []

dropout_list = []
episode_number = 10000



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

            

        # action process and take reward / take observation

        next_state, reward, done, _ = env.step(action)

        

        # q learning funct

        

        old_value = q_table[state, action]  #old value

        next_max = np.max(q_table[next_state]) #next max

        

        next_value = (1-alpha)*old_value + alpha*(reward + gamma*next_max)

        # q table update

        q_table[state,action] = next_value

        

        # update state

        state = next_state

        

        # find wrong dropout 

        if reward == -10:

            dropouts += 1

            

        if done:

            break

        

        reward_count  += reward

    if i%10 == 0:

        

        dropout_list.append(dropouts)

        reward_list.append(reward_count)

        print("Episode: {}, reward {}, wrong dropout {}".format(i, reward_count,dropouts))
fig, axs = plt.subplots(1,2)



axs[0].plot(reward_list)

axs[0].set_xlabel("episode")

axs[0].set_ylabel("reward")



axs[1].plot(dropout_list)

axs[1].set_xlabel("episode")

axs[1].set_ylabel("wrong dropout")



axs[0].grid(True)

axs[1].grid(True)

plt.show()
env.s = env.encode(0,0,3,4)

env.render()
