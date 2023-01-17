import gym
# Importing libraries
import numpy as np
import random
import math
from collections import deque
import collections
import pickle

#for text processing
import re
import pandas as pd
env = gym.make("Taxi-v3").env

env.render()
city_df = pd.read_csv("city.csv")
all_cities = city_df['location'].tolist()

all_cities
from nltk.tokenize import word_tokenize as tk

def fetch_pickup_drop(text):
    city_df = pd.read_csv("city.csv")
    all_cities = city_df['location'].tolist()
    
    city_names_in_sms = []
    origin = ""
    destination = ""
    time_of_pickup = ""
    
    for city in all_cities:
        if city in text:
            city_names_in_sms.append(city)
            
    if len(city_names_in_sms) == 2:
        for city in city_names_in_sms:
            
            #1st case city to city
            orig_dest_match = re.findall(city_names_in_sms[1]+' to '+city_names_in_sms[0], text)
            if len(orig_dest_match):
                origin = city_names_in_sms[1]
                destination = city_names_in_sms[0]
                break
            orig_dest_match = re.findall(city_names_in_sms[0]+' to '+city_names_in_sms[1], text)
            if len(orig_dest_match):
                origin = city_names_in_sms[0]
                destination = city_names_in_sms[1]
                break
            
            #2nd case [from, to , for] - city
            dest_match = re.findall('to '+city, text)
            if len(dest_match)<1:
                dest_match = re.findall('for '+city, text)
            if len(dest_match):
                destination = city
            orig_match = re.findall('from '+city,text)
            if len(orig_match):
                origin = city
                
    time = re.findall(r'[\d+]+ PM', text)
    if len(time):
        time_of_pickup = time[0]
    else:
        time = re.findall(r'[\d+]+ AM', text)
        if len(time):
            time_of_pickup = time[0]
  
    return [origin, destination, time_of_pickup]
                 
env.reset() # reset environment to a new, random state
env.render()

print("Action Space {}".format(env.action_space))
print("State Space {}".format(env.observation_space))
#Initialize Q_table
import numpy as np

state = env.encode(4, 1, 2, 3) # (taxi row, taxi column, passenger index, destination index)
print("State:", state)
env.s = state
env.render()

q_table = np.zeros([env.observation_space.n, env.action_space.n])
sms = pd.read_csv('sms.txt', names = ['text'])
sms.iloc[2]['text']
%%time
import random
from IPython.display import clear_output

# Create Hyperparameters for our Q-learning algorithm
total_epochs = 50000           # total episodes
total_test_episodes = 100        # total test episodes
max_steps = 99                   # Max steps per episode

learning_rate = 0.7              # Learning rate
gamma = 0.610                      # Discounting rate, gamma
alpha = 0.3
# Exploration parameters
epsilon = 1.0                    # Exploration rate
max_epsilon = 1.0                # Exploration probability at the start
min_epsilon = 0.01               # Minimum exploration probability
decay_rate = 0.01                # Exponential decay rate for exploration probability
"""Training the agent"""
for epoch in range(total_epochs):
    # Reset environment
    state = env.reset()
    step = 0
    done = False
    
    for step in range(max_steps):
        # choose an action in the current state space
        exp_expl_tradeoff = random.uniform(0,1)
        
        # if this number > epsilon then we start exploitation(taking the biggest q-value for the state)
        if exp_expl_tradeoff > epsilon:
            action = np.argmax(q_table[state,:])
        # else we start exploration:
        else:
            action = env.action_space.sample()
            
        # observe the outcome state(s`) and reward(r)
        new_state, reward, done, info = env.step(action) #run one timestep 
        
        # Update Q(s,a):= Q(s,a) + lr [R(s,a) + gamma * max Q(s',a') - Q(s,a)]
        q_table[state, action] = q_table[state, action] + learning_rate * (reward + gamma * 
                                    np.max(q_table[new_state, :]) - q_table[state, action])
        # current state = new state 
        state = new_state
        
        # if done finish episode
        if done == True:
            break
        
    epoch += 1
        
    # Reduce epsilon beacuse we need to explore lesser as we progress
    epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(-decay_rate*epoch)#1+e^(-0.01*1)


np.save("./q_table.npy", q_table)
#Load trained q_table for evaluation

q_table = np.load("./q_table.npy")
def create_loc_dict(city_df):
    loc_dict = {'dwarka sector 23': 0, 'dwarka sector 21': 1, 'hauz khaas':2, 'airport':3}
    ## Create dictionary example, loc_dict['dwarka sector 23] = 0
        
    return loc_dict
orig_df = pd.read_csv(r"C:\Users\DAVE\Documents\Notebooks\Mid-Project II\org_df.csv") 
def check_pick_up_drop_correction(pick_up, drop, line_num):
    org_list = orig_df.iloc[line_num].tolist()
    original_origin = org_list[0]
    original_destination = org_list[1]
    if original_origin == pick_up and original_destination == drop:
        return True
    else:
        return False

    
"""Evaluate agent's performance after Q-learning"""

# 1) We need to take text drom "sms.txt" and fetch pickup and drop from it.
# 2) Generate the random state from an enviroment and change the pick up and drop as the fetched one
# 3) Evaluate you q_table performance on all the texts given in sms.txt.
# 4) Have a check if the fetched pickup, drop is not matching with original pickup, drop using orig.csv
# 5) If fetched pickup or/and drop does not match with the original, add penality and reward -10
# 6) Calculate the Total reward, penalities, Wrong pickup/drop predicted and Average time steps per episode.

total_epochs, total_penalties,wrong_predictions = 0, 0, 0
total_reward = []


count = 0
time_list = []
f = open("./sms.txt", "r")
num_of_lines = 1000
episode = 0
city = pd.read_csv("./city.csv")

loc_dict = create_loc_dict(city)
line_num = 0
rewards = 0
frames=[]
for line in f:
    done = False
    pickup, drop, time = fetch_pickup_drop(line)
    rewards = 0
    state = env.reset()
    state = env.encode(random.randint(0,4), random.randint(0,4) , loc_dict[pickup], loc_dict[drop]) # (taxi row, taxi column, passenger index, destination index)
    env.s = state
    
    print("********************************************")
    print("Episode", episode)
    
    for step in range(200):
        action = np.argmax(q_table[state,:])
        
        new_state, reward, done, info = env.step(action)
        
        rewards += reward
        state = new_state
        
        frames.append({
        'episode': episode,
        'origin': pickup,
        'destination': drop,
        'frame': env.render(mode='ansi'),
        'state': new_state,
        'action': action,
        'reward': reward
        }
        )
        
        if done:
            total_reward.append(rewards)
            print("Score:", rewards)
            total_epochs += step
            break
    if (check_pick_up_drop_correction(pickup, drop, line_num)):
        pass
    else:
        total_penalties+=1
        wrong_predictions+=1
        total_reward[line_num] -= 10
        
    
    line_num+=1
    episode+=1
     
    
total_rewards = 0
for i in total_reward:
    total_rewards += i
total_rewards = total_rewards/num_of_lines


print(f"Results after {num_of_lines} episodes:")
print(f"Average timesteps per episode: {total_epochs / num_of_lines}")
print(f"Average penalties per episode: {total_penalties / num_of_lines}")
print(f"Total number of wrong predictions", wrong_predictions)
print()
print("Total Reward is", total_rewards)
from IPython.display import clear_output
from time import sleep

def print_frames(frames):
    print(frames)
    for i, frame in enumerate(frames):
        clear_output(wait=True)
        print(frame['frame'])
        print(f"Episode: {frame['episode']}, origin: {frame['origin']}, destination: {frame['destination']}")
        print(f"Timestep: {i + 1}")
        print(f"State: {frame['state']}")
        print(f"Action: {frame['action']}")
        print(f"Reward: {frame['reward']}")
        sleep(.5)
        
print_frames(frames)
print(frames[9]['episode'])
