import numpy as np

from numpy.random import random

state_low = 0

state_high = 1

reward_search = 1

reward_waiting = 0

reward_recharge = 0

reward_search_OutBat = -1 #penanty when try to search while low battery and come back to recharge

# probability while search and remain:

p_high = 0.5 # remain high

p_low = 0.5 # remain low



# actions

# High state actions: wait,search

# Low state actions: wait,search, recharge

action_probs = [np.array([0.5, 0.5]), np.array([1.0/3.0, 1.0/3.0, 1.0/3.0])]

state_actions = [2,3]

discount = 0.5

# There should not be any forced knowledge for the agent, just let it learn.

def next_state(state, action):

    if state == state_high: # high state

        if action == 0: # wait

            n_state = state_high

            reward = reward_waiting

        else: # search

            n_state = state_high if random()<=p_high else state_low

            reward = reward_search

    else: # low state

        if action == 0: # wait

            n_state = state_low

            reward = reward_waiting

        elif action ==1: # search

            n_state = state_low if random()<=p_low else state_high

            reward = reward_search if n_state == state_low else reward_search_OutBat

        else: #recharge

            n_state = state_high

            reward = reward_recharge

                

    return n_state, reward



# doc test, pytest

import doctest

doctest.testmod(name='next_state', verbose=False)



def main():

    v = np.zeros(2)

    num_iter = 1000

    for i in range(num_iter):

        new_v = np.zeros_like(v)

        for state in [0,1]:

            for a in range(state_actions[state]): # go through action for high and low state

                action_prob = action_probs[state][a]

                # take action

                n_state, reward = next_state(state, a)

                # update v* 

                new_v[state] += action_prob*(reward + discount*v[n_state])

        v = new_v

    print(v)



main()
reward_search = 1

reward_waiting = 0

reward_recharge = 0

reward_search_OutBat = -3 #penanty when try to search while low battery and come back to recharge

main()
reward_search = 3

reward_waiting = 0

reward_recharge = 0

reward_search_OutBat = 0 #penanty when try to search while low battery and come back to recharge

main()