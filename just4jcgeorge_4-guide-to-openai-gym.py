import gym # openAi gym

from gym import envs

from IPython.display import Image

import os

Image("/kaggle/input/week9dataset/Guide of OpenAI Gym1.png")
Image("/kaggle/input/week9dataset/Quality_Based_Reinforcement_Learning_Methods2.jpeg")
import gym

env = gym.make('FrozenLake-v0')

env.action_space# View the number of actions available in the environment
env.reset() # Reset the environment

env.render() # Render the environment



action = env.action_space.sample() # Use sample to indicate the random actions

env.step(action) # Take random actions
observation = env.reset() 



for t in range(20):

    env.render()

    action = env.action_space.sample()

    observation, reward, done, info = env.step(action) 

    print(observation, reward, done) # Output

    if done: # If the game is terminated

        print("Episode finished after {} timesteps".format(t+1)) # Print the timesteps

        break
from IPython import display

import time



observation = env.reset() 

for t in range(20):

    env.render() 

    action = env.action_space.sample()

    observation, reward, done, info = env.step(action) 

    print(observation, reward, done)

    display.clear_output(wait=True) # Clear the Output

    time.sleep(1) # Delay 1s to execute

    print("step:", t)

    if done:

        print("Episode finished after {} timesteps".format(t+1))

        break
!pip install gym[atari] 

from matplotlib import pyplot as plt

%matplotlib inline



env = gym.make("Breakout-v0")

rgb_array = env.render(mode='rgb_array') # Render as arrays

plt.imshow(rgb_array) # Display
env.reset()

for t in range(100): # Set maximum timesteps as 100

    plt.imshow(env.render(mode='rgb_array'))

    display.display(plt.gcf())

    action = env.action_space.sample()

    observation, reward, done, info = env.step(action) # Take random actions

    print(reward, done)

    display.clear_output(wait=True)