import gym

import time

from IPython.display import clear_output



env = gym.make("FrozenLake-v0", is_slippery=False).env

# env.s = 10

env.render()



# for i in range(0,100):

#     clear_output(wait=True)

#     env.reset()

#     env.render()

#     time.sleep(0.5)

    
def print_frames(frames):

    for i, frame in enumerate(frames):

        clear_output(wait=True)

        print(frame['frame'])

        print(f"Episode: {frame['episode']}")

        print(f"Timestep: {i + 1}")

        print(f"State: {frame['state']}")

        print(f"Action: {frame['action']}")

        print(f"Reward: {frame['reward']}")

        time.sleep(1)
import numpy as np

q_table = np.zeros([env.observation_space.n, env.action_space.n])
state = env.reset()

env.s = 14

env.render()

print(env.step(2))

time.sleep(5)

clear_output(wait=True)

env.render()

%%time

"""Training the agent"""



import random

from IPython.display import clear_output



# Hyperparameters

alpha = 0.8

gamma = 0.1

epsilon = 0.2



# For plotting metrics

all_epochs = []

# all_penalties = []



for i in range(1, 100000):

    state = env.reset()



    epochs,  reward, = 0, 0

    done = False

    

    while not done:

        explore_eploit = random.uniform(0, 1)

        if  explore_eploit < epsilon:

            action = env.action_space.sample() # Explore action space

        else:

            action = np.argmax(q_table[state]) # Exploit learned values



        next_state, reward, done, info = env.step(action) 

        

        old_value = q_table[state, action]

        next_max = np.max(q_table[next_state])

        

        new_value = (1 - alpha) * old_value + alpha * (reward + gamma * next_max)

        q_table[state, action] = new_value

        

        state = next_state

        epochs += 1

        

    if i % 100 == 0:

        clear_output(wait=True)

        print(f"Episode: {i}")



print("Training finished.\n")
total_epochs = 0

episodes = 100

frames = []

tot_reward = 0

from random import randint



for ep in range(episodes):

#     state = env.reset()

    env.s = randint(0, 13)

    epochs, reward =  0, 0

    

    done = False

    

    while not done:

        action = np.argmax(q_table[state])

        state, reward, done, info = env.step(action)



        

        # Put each rendered frame into dict for animation

        frames.append({

            'frame': env.render(mode='ansi'),

            'episode': ep, 

            'state': state,

            'action': action,

            'reward': reward

            }

        )

        epochs += 1



    total_epochs += epochs

    tot_reward += reward



print(f"Results after {episodes} episodes:")

print(f"Average timesteps per episode: {total_epochs / episodes}")

print(f"Total Rewards {tot_reward}")
print_frames(frames)
q_table