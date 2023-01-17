# 1. Enable Internet in the Kernel (Settings side pane)



# 2. Curl cache may need purged if v0.1.6 cannot be found (uncomment if needed). 

# !curl -X PURGE https://pypi.org/simple/kaggle-environments



# ConnectX environment was defined in v0.1.6

!pip install 'kaggle-environments>=0.1.6'
from kaggle_environments import evaluate, make, utils



env = make("connectx", debug=True)
import random

import numpy as np

from matplotlib import pyplot as plt



env.configuration.rows = 3

env.configuration.columns = 4

env.configuration.inarow = 3

#env.specification
# rows is state

# columns are actions

q_table = np.zeros([3**(env.configuration.rows * env.configuration.columns),env.configuration.columns])
# Hyperparameters

alpha = 0.9    # Learning rate

gamma = 0.1    # Discount factor (0 = only this step matters)

# epsilon is determined each step



all_epochs = []

all_penalties = []



NB_STEPS = 10000

NB_STEPS_RANDOM = 1000

EPSILON_0 = 1

EPSILON_END = 0.1

NB_STEPS_EPSILON = 1000

trainer = env.train([None, "random"])

episodes = 0

progression = []

step = 0

random.seed(10)

for _ in range(NB_STEPS):

    observation = trainer.reset()

  

    # Init Vars

    total_reward = 0

    done = False

    episode_step = 0  

    while not done:

        # This part computes the epsilon that will be used later on

        if step < NB_STEPS_RANDOM:

            epsilon = 1

        elif step < NB_STEPS_RANDOM + NB_STEPS_EPSILON:

            epsilon = EPSILON_0 - (EPSILON_0-EPSILON_END) * ((step-NB_STEPS_RANDOM) / NB_STEPS_EPSILON)

        else:

            epsilon = EPSILON_END

        

        # Sometimes, you cannot play a move because the column is full, so you take actions among the possible columns

        possible_action = observation.board[0:env.configuration.columns]

        possible_action = [i for i in range(len(possible_action)) if possible_action[i] == 0]

        # This next line converts the state to an integer using ternary converter.

        # For example, if the state is 0,1,2; the row will be 1*3 + 2 = 5. I will look the fifth row and take the action that yields the max value

        obs_value = int("".join(str(i) for i in observation.board),3)

        if random.uniform(0, 1) < epsilon:

            # Check the action space and choose a random action

            action = random.choice(possible_action)

        else:

            # Check the learned values

            action = possible_action[int(np.argmax(q_table[obs_value][possible_action],axis=0))]

        

        next_observation, reward, done, _ = trainer.step(action)

        # 1 for a win, 0 for a draw and -1 for a lose.

        if done :

            if reward == None:

                reward = -1

        if reward == 0.5:

            reward = 0

        elif reward == 0:

            reward = -1        

          

        # Old Q-table value

        old_value = q_table[obs_value, action]

        next_max = np.max(q_table[obs_value])



        # Update the new value

        # Bellman equation !

        new_value = (1 - alpha) * old_value + alpha * (reward + gamma * next_max)

        q_table[obs_value, action] = new_value



        observation = next_observation

        

        total_reward += reward

        step += 1               

        episode_step += 1



        if done:

            progression.append(total_reward)

            if episodes % 100 == 0:

                print(episodes, epsilon, sum(progression[-101:-1]),sum(sum(q_table)))

            episodes += 1   



# Lets look the performance of the agent for 100 games interval

chunks = 100

list_progression = [progression[i:i + chunks] for i in range(0, len(progression), chunks)]

list_progression = [sum(i)/len(i) for i in list_progression]

plt.plot(list_progression)

plt.show()
for _ in range(1000):

    observation = trainer.reset()

  

    # Init Vars

    total_reward = 0

    done = False

    episode_step = 0  

    while not done:

        

        possible_action = observation.board[0:env.configuration.columns]

        possible_action = [i for i in range(len(possible_action)) if possible_action[i] == 0]



        # Check the learned values

        obs_value = int("".join(str(i) for i in observation.board),3)

        action = possible_action[int(np.argmax(q_table[obs_value][possible_action],axis=0))]

        

        next_observation, reward, done, _ = trainer.step(action)

        if done :

            if reward == None:

                reward = -1

        

        if reward == 0.5:

            reward = 0

        elif reward == 0:

            reward = -1        



        observation = next_observation

        

        total_reward += reward

        step += 1               

        episode_step += 1



        if done:

            progression.append(total_reward)

            if episodes % 10 == 0:

                print(episodes, epsilon, sum(progression[-11:-1]),sum(sum(q_table)))

            episodes += 1