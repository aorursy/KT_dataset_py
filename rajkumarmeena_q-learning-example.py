# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np
import pandas as pd

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print("This notebook is to practice q learning example")

# Any results you write to the current directory are saved as output.
# the problem starts from here. This problem of finding the best path that will work where we have
# no idea about the sourroundings.
import pylab as plt

# R matrix
npIntialMatrix = [ [-1,-1,-1,-1,0,-1],[-1,-1,-1,0,-1,100],[-1,-1,-1,0,-1,-1],[-1,0,0,-1,0,-1],
[-1,0,0,-1,-1,100],[-1,0,-1,-1,0,100] ]
R = np.matrix(npIntialMatrix)
Q = np.matrix(np.zeros([6,6]))
print(Q)
print(R)
gamma = 0.9
# initial state of the player is started with postion 1 although it can be started from any
# where we want.
initial_state = 1
current_state_row = R[1,]
def avaliable_actions(state):
    current_state_row = R[state,]
    
print(current_state_row)
import numpy as np

# R matrix
R = np.matrix([ [-1,-1,-1,-1,0,-1],
		[-1,-1,-1,0,-1,100],
		[-1,-1,-1,0,-1,-1],
		[-1,0,0,-1,0,-1],
		[-1,0,0,-1,-1,100],
		[-1,0,-1,-1,0,100] ])

# Q matrix
Q = np.matrix(np.zeros([6,6]))

# Gamma (learning parameter).
gamma = 0.8

# Initial state. (Usually to be chosen at random)
initial_state = 1

# This function returns all available actions in the state given as an argument
def available_actions(state):
    current_state_row = R[state,]
    av_act = np.where(current_state_row >= 0)[1]
    return av_act

# Get available actions in the current state
available_act = available_actions(initial_state) 

# This function chooses at random which action to be performed within the range 
# of all the available actions.
def sample_next_action(available_actions_range):
    next_action = int(np.random.choice(available_act,1))
    return next_action

# Sample next action to be performed
action = sample_next_action(available_act)

# This function updates the Q matrix according to the path selected and the Q 
# learning algorithm
def update(current_state, action, gamma):
    
    max_index = np.where(Q[action,] == np.max(Q[action,]))[1]

    if max_index.shape[0] > 1:
        max_index = int(np.random.choice(max_index, size = 1))
    else:
        max_index = int(max_index)
    max_value = Q[action, max_index]
    
    # Q learning formula
    Q[current_state, action] = R[current_state, action] + gamma * max_value

# Update Q matrix
update(initial_state,action,gamma)

#-------------------------------------------------------------------------------
# Training

# Train over 10 000 iterations. (Re-iterate the process above).
for i in range(10000):
    current_state = np.random.randint(0, int(Q.shape[0]))
    available_act = available_actions(current_state)
    action = sample_next_action(available_act)
    update(current_state,action,gamma)
    
# Normalize the "trained" Q matrix
print("Trained Q matrix:")
print(Q/np.max(Q)*100)

#-------------------------------------------------------------------------------
# Testing

# Goal state = 5
# Best sequence path starting from 2 -> 2, 3, 1, 5

current_state = 2
steps = [current_state]

while current_state != 5:

    next_step_index = np.where(Q[current_state,] == np.max(Q[current_state,]))[1]
    
    if next_step_index.shape[0] > 1:
        next_step_index = int(np.random.choice(next_step_index, size = 1))
    else:
        next_step_index = int(next_step_index)
    
    steps.append(next_step_index)
    current_state = next_step_index

# Print selected sequence of steps
print("Selected path:")
print(steps)
import numpy as np
import pandas as pd

# R matrix for immidiat rewards
R = np.matrix([[-1,-1,-1,-1,0,-1],
               [-1,-1,-1,0,-1,100],
               [-1,-1,-1,0,-1,-1],
               [-1,0,0,-1,0,-1],
               [-1,0,0,-1,-1,100],
               [-1,0,-1,-1,0,100] ])
# Q matrix for the optimal path
Q = np.matrix(np.zeros([6,6]))

print(Q)
print(R)
# discount factor for future rewards. I am keeping it high for future rewards
gamma = .8
# now we have to choose what is the intial state of the Robot in the environment
initial_state = 1

# Now we need to find all the possible move the robot can take. So writing a function
# the is going to provide the next move value

def all_possible_move(state):
    current_state_row = R[state,]
    av_move = np.where(current_state_row>0)[1]
    return av_move

# get the possible action using above method
available_action = all_possible_move(initial_state)
# now chosse any random possible move at once. The below function is for that.

def next_action_at_random(available_action):
    next_action = int(np.random.choice(available_action,1))
    return next_action

actual_action = next_action_at_random(available_action)
