!pip install kaggle-environments --upgrade
from __future__ import absolute_import

from __future__ import division

from __future__ import print_function



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

import sys

import PIL.Image



import tensorflow as tf

import logging



from sklearn import preprocessing

import random

import matplotlib.pyplot as plt

import seaborn as sns



from kaggle_environments import evaluate, make

from kaggle_environments.envs.halite.helpers import *

seed=123

tf.compat.v1.set_random_seed(seed)

session_conf = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)

sess = tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph(), config=session_conf)

tf.compat.v1.keras.backend.set_session(sess)

logging.disable(sys.maxsize)

global ship_
env = make("halite", debug=True)

env.run(["random"])

env.render(mode="ipython",width=800, height=600)
env.configuration
env.specification
env.specification.reward
env.specification.action
env.specification.observation
def getDirTo(fromPos, toPos, size):

    fromX, fromY = divmod(fromPos[0],size), divmod(fromPos[1],size)

    toX, toY = divmod(toPos[0],size), divmod(toPos[1],size)

    if fromY < toY: return ShipAction.NORTH

    if fromY > toY: return ShipAction.SOUTH

    if fromX < toX: return ShipAction.EAST

    if fromX > toX: return ShipAction.WEST



# Directions a ship can move

directions = [ShipAction.NORTH, ShipAction.EAST, ShipAction.SOUTH, ShipAction.WEST]



# Will keep track of whether a ship is collecting halite or carrying cargo to a shipyard

ship_states = {}



# Returns the commands we send to our ships and shipyards

def simple_agent(obs, config):

    size = config.size

    board = Board(obs, config)

    me = board.current_player

    # If there are no ships, use first shipyard to spawn a ship.

    if len(me.ships) == 0 and len(me.shipyards) > 0:

        me.shipyards[0].next_action = ShipyardAction.SPAWN



    # If there are no shipyards, convert first ship into shipyard.

    if len(me.shipyards) == 0 and len(me.ships) > 0:

        me.ships[0].next_action = ShipAction.CONVERT

    

    for ship in me.ships:

        if ship.next_action == None:

            

            ### Part 1: Set the ship's state 

            if ship.halite < 200: # If cargo is too low, collect halite

                ship_states[ship.id] = "COLLECT"

            if ship.halite > 500: # If cargo gets very big, deposit halite

                ship_states[ship.id] = "DEPOSIT"

                

            ### Part 2: Use the ship's state to select an action

            if ship_states[ship.id] == "COLLECT":

                # If halite at current location running low, 

                # move to the adjacent square containing the most halite

                if ship.cell.halite < 100:

                    neighbors = [ship.cell.north.halite, ship.cell.east.halite, 

                                 ship.cell.south.halite, ship.cell.west.halite]

                    best = max(range(len(neighbors)), key=neighbors.__getitem__)

                    ship.next_action = directions[best]

            if ship_states[ship.id] == "DEPOSIT":

                # Move towards shipyard to deposit cargo

                direction = getDirTo(ship.position, me.shipyards[0].position, size)

                if direction: ship.next_action = direction

                

    return me.next_actions
trainer = env.train([None, "random"])

observation = trainer.reset()

while not env.done:

    my_action = simple_agent(observation, env.configuration)

    print("My Action", my_action)

    observation = trainer.step(my_action)[0]

    print("Reward gained",observation.players[0][0])
env.render(mode="ipython",width=800, height=600)
def ActorModel(num_actions,in_):

    common = tf.keras.layers.Dense(128, activation='tanh')(in_)

    common = tf.keras.layers.Dense(32, activation='tanh')(common)

    common = tf.keras.layers.Dense(num_actions, activation='softmax')(common)

    

    return common
def CriticModel(in_):

    common = tf.keras.layers.Dense(128)(in_)

    common = tf.keras.layers.ReLU()(common)

    common = tf.keras.layers.Dense(32)(common)

    common = tf.keras.layers.ReLU()(common)

    common = tf.keras.layers.Dense(1)(common)

    

    return common
input_ = tf.keras.layers.Input(shape=[441,])

model = tf.keras.Model(inputs=input_, outputs=[ActorModel(5,input_),CriticModel(input_)])

model.summary()
optimizer = tf.keras.optimizers.Adam(lr=7e-4)
huber_loss = tf.keras.losses.Huber()

action_probs_history = []

critic_value_history = []

rewards_history = []

running_reward = 0

episode_count = 0

num_actions = 5

eps = np.finfo(np.float32).eps.item()

gamma = 0.99  # Discount factor for past rewards

env = make("halite", debug=True)

trainer = env.train([None,"random"])
le = preprocessing.LabelEncoder()

label_encoded = le.fit_transform(['NORTH', 'SOUTH', 'EAST', 'WEST', 'CONVERT'])

label_encoded
def getDirTo(fromPos, toPos, size):

    fromX, fromY = divmod(fromPos[0],size), divmod(fromPos[1],size)

    toX, toY = divmod(toPos[0],size), divmod(toPos[1],size)

    if fromY < toY: return ShipAction.NORTH

    if fromY > toY: return ShipAction.SOUTH

    if fromX < toX: return ShipAction.EAST

    if fromX > toX: return ShipAction.WEST



# Directions a ship can move

directions = [ShipAction.NORTH, ShipAction.EAST, ShipAction.SOUTH, ShipAction.WEST]

   

def decodeDir(act_):

    if act_ == 'NORTH':return directions[0]

    if act_ == 'EAST':return directions[1]

    if act_ == 'SOUTH':return directions[2]

    if act_ == 'WEST':return directions[3]

    

# Will keep track of whether a ship is collecting halite or carrying cargo to a shipyard

ship_states = {}

ship_ = 0

def update_L1():

    ship_+=1

# Returns the commands we send to our ships and shipyards

def advanced_agent(obs, config, action):

    size = config.size

    board = Board(obs, config)

    me = board.current_player 

    act = le.inverse_transform([action])[0]

    global ship_

    

   # If there are no ships, use first shipyard to spawn a ship.

    if len(me.ships) == 0 and len(me.shipyards) > 0:

        me.shipyards[ship_-1].next_action = ShipyardAction.SPAWN



    # If there are no shipyards, convert first ship into shipyard.

    if len(me.shipyards) == 0 and len(me.ships) > 0 and ship_==0:

        me.ships[0].next_action = ShipAction.CONVERT   

    try: 

        if act=='CONVERT':

            me.ships[0].next_action = ShipAction.CONVERT

            update_L1()

            if len(me.ships)==0 and len(me.shipyards) > 0:

                me.shipyards[ship_-1].next_action = ShipyardAction.SPAWN

        if me.ships[0].halite < 200:

            ship_states[me.ships[0].id] = 'COLLECT'

        if me.ships[0].halite > 800:

            ship_states[me.ships[0].id] = 'DEPOSIT' 



        if ship_states[me.ships[0].id] == 'COLLECT': 

            if me.ships[0].cell.halite < 100:

                me.ships[0].next_action = decodeDir(act)

        if ship_states[me.ships[0].id] == 'DEPOSIT':

            # Move towards shipyard to deposit cargo

            direction = getDirTo(me.ships[0].position, me.shipyards[ship_-1].position, size)

            if direction: me.ships[0].next_action = direction

    except:

        pass

                

    return me.next_actions
while not env.done:    

    state = trainer.reset()

    episode_reward = 0

    with tf.GradientTape() as tape:

        for timestep in range(1,env.configuration.episodeSteps+200):

            # of the agent in a pop up window.

            state_ = tf.convert_to_tensor(state.halite)

            state_ = tf.expand_dims(state_, 0)

            # Predict action probabilities and estimated future rewards

            # from environment state

            action_probs, critic_value = model(state_)

            critic_value_history.append(critic_value[0, 0])

            

            # Sample action from action probability distribution

            action = np.random.choice(num_actions, p=np.squeeze(action_probs))

            action_probs_history.append(tf.math.log(action_probs[0, action]))

            

            # Apply the sampled action in our environment

            action = advanced_agent(state, env.configuration, action)

            state = trainer.step(action)[0]

            gain=state.players[0][0]/5000

            rewards_history.append(gain)

            episode_reward += gain

            

            if env.done:

                state = trainer.reset() 

        # Update running reward to check condition for solving

        running_reward = 0.05 * episode_reward + (1 - 0.05) * running_reward



        # Calculate expected value from rewards

        # - At each timestep what was the total reward received after that timestep

        # - Rewards in the past are discounted by multiplying them with gamma

        # - These are the labels for our critic

        returns = []

        discounted_sum = 0

        for r in rewards_history[::-1]:

            discounted_sum = r + gamma * discounted_sum

            returns.insert(0, discounted_sum)

        # Normalize

        returns = np.array(returns)

        returns = (returns - np.mean(returns)) / (np.std(returns) + eps)

        returns = returns.tolist()

        # Calculating loss values to update our network

        history = zip(action_probs_history, critic_value_history, returns)

        actor_losses = []

        critic_losses = []

        for log_prob, value, ret in history:

            # At this point in history, the critic estimated that we would get a

            # total reward = `value` in the future. We took an action with log probability

            # of `log_prob` and ended up recieving a total reward = `ret`.

            # The actor must be updated so that it predicts an action that leads to

            # high rewards (compared to critic's estimate) with high probability.

            diff = ret - value

            actor_losses.append(-log_prob * diff)  # actor loss



            # The critic must be updated so that it predicts a better estimate of

            # the future rewards.

            critic_losses.append(

                huber_loss(tf.expand_dims(value, 0), tf.expand_dims(ret, 0))

            )

        # Backpropagation

        loss_value = sum(actor_losses) + sum(critic_losses)

        grads = tape.gradient(loss_value, model.trainable_variables)

        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        

        # Clear the loss and reward history

        action_probs_history.clear()

        critic_value_history.clear()

        rewards_history.clear()

        

    # Log details

    episode_count += 1

    if episode_count % 10 == 0:

        template = "running reward: {:.2f} at episode {}"

        print(template.format(running_reward, episode_count))



    if running_reward > 550:  # Condition to consider the task solved

        print("Solved at episode {}!".format(episode_count))

        break
while not env.done:

    state_ = tf.convert_to_tensor(state.halite)

    state_ = tf.expand_dims(state_, 0)

    action_probs, critic_value = model(state_)

    critic_value_history.append(critic_value[0, 0])

    action = np.random.choice(num_actions, p=np.squeeze(action_probs))

    action_probs_history.append(tf.math.log(action_probs[0, action]))

    action = advanced_agent(state, env.configuration, action)

    state = trainer.step(action)[0]
env.render(mode="ipython",width=800, height=600)