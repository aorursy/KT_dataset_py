!pip install gym 

!pip install 'gym[box2d]'

!pip install atari_py
import gym

from gym import wrappers

from gym import envs

import numpy as np 

#import datetime

import matplotlib.pyplot as plt

%matplotlib inline

#import time

import os
env = gym.make("BreakoutNoFrameskip-v4")

plt.imshow(env.render('rgb_array'))

plt.grid(False)

print("Observation space:", env.observation_space)

print("Action space:", env.action_space)

env = gym.make("MsPacmanNoFrameskip-v4")

plt.imshow(env.render('rgb_array'))

plt.grid(False)

print("Observation space:", env.observation_space)

print("Action space:", env.action_space)
env = gym.make("SpaceInvadersNoFrameskip-v4")

plt.imshow(env.render('rgb_array'))

plt.grid(False)

print("Observation space:", env.observation_space)

print("Action space:", env.action_space)
# Set-up the virtual display environment

!apt-get update

!apt-get install python-opengl -y

!apt install xvfb -y

!pip install pyvirtualdisplay

!pip install piglet

!apt-get install ffmpeg -y
# Start the virtual monitor

from pyvirtualdisplay import Display

display = Display(visible=0, size=(1400, 900))

display.start()
env = gym.make("BreakoutNoFrameskip-v4")

# plt.imshow(env.render('rgb_array'))

# plt.grid(False)

# print("Observation space:", env.observation_space)

# print("Action space:", env.action_space)



# play a random game and create video

# env = gym.make("MsPacmanNoFrameskip-v4")

monitor_dir = os.getcwd()



#Setup a wrapper to be able to record a video of the game

record_video = True

should_record = lambda i: record_video

env = wrappers.Monitor(env, monitor_dir, video_callable=should_record, force=True)



#Play a random game

state = env.reset()

done = False

while not done:

  action = env.action_space.sample() #random action, replace by the prediction of the model

  state, reward, done, _ = env.step(action)



record_video = False

env.close() 



# download videos

#from google.colab import files

#import glob

os.chdir(monitor_dir) # change directory to get the files

!pwd #show file path

!ls # show directory content



from IPython.display import FileLink



monitor_dir = os.getcwd()

FileLink(r'openaigym.video.5.15.video000000.mp4')