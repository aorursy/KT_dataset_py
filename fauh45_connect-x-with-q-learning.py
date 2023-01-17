# Downloading kaggle env

!pip install 'kaggle-environments>=0.1.6'
# Make Kaggle Env

from kaggle_environments import evaluate, make, utils



env = make("connectx", debug=True)
# Test Out the environment

env.render(mode="ipython", width=500, height=450)
# Creating Agents

def agent(observation, configuration):

    from random import choice

    return choice([c for c in range(configuration.columns) if observation.board[c] == 0])
env.reset()

env.run([agent, 'random'])

env.render(mode="ipython", width=500, height=450)
from collections import defaultdict

from tqdm.notebook import tqdm

import numpy as np

import random
LEARNING_RATE = .1

DISCOUNT_FACTOR = .6

EPSILON = .99

SWITCH_PROBABILITY = .5



PAIR = [None, 'negamax']



episode = 10000



MIN_EPSILON = .1

EPSILON_DECAY = .9999

LR_DECAY = .9

LR_DECAY_STEP = 1000
def epsilon_greedy(n_action, QTable, state):

    global EPSILON

    

    r = random.uniform(0, 1)

    if r >= EPSILON:

        curr_state = tuple(state['board'])

        return np.argmax([QTable[curr_state][c] if state['board'][c] == 0 else -1e9 for c in range(n_action)])

    else:

        return random.choice([c for c in range(n_action) if state['board'][c] == 0])
Q_table = defaultdict(lambda: np.zeros(env.configuration.columns))
# History

total_reward_per_episode = []

epoch_per_episode = []

q_table_row = []
trainer = env.train(PAIR)
for i in tqdm(range(episode)):

    # Do random change of enemy agent

    if random.uniform(0, 1) < SWITCH_PROBABILITY:

        PAIR = PAIR[::-1]

        trainer = env.train(PAIR)

    

    EPSILON = max(MIN_EPSILON, EPSILON * EPSILON_DECAY)

    state = trainer.reset()

    done = False

    

    epoch, total_reward = 0, 0

    

    while not done:

        action = int(epsilon_greedy(env.configuration.columns, Q_table, state))

        next_state, reward, done, info = trainer.step(action)

        

        if done:

            if reward == 1:

                reward = 20

            elif reward == 0:

                reward = -20

            else:

                reward = 10

        else:

            reward = -0.05

            

        old_value = Q_table[tuple(state['board'])][action]

        next_max = np.max(Q_table[tuple(next_state['board'])])

        

        new_value = old_value + LEARNING_RATE * (reward + DISCOUNT_FACTOR * next_max - old_value)

        state = next_state

        

        Q_table[tuple(state['board'])][action] = new_value

        

        total_reward += reward

        epoch += 1

    

    total_reward_per_episode.append(total_reward)

    epoch_per_episode.append(epoch)

    q_table_row.append(len(Q_table))

    

    if (i + 1) % LR_DECAY_STEP == 0:

        LEARNING_RATE *= LR_DECAY
len(Q_table)
import matplotlib.pyplot as plt



plt.plot(total_reward_per_episode)

plt.xlabel('Episode')

plt.ylabel('Total Rewards')

plt.show()
plt.plot(epoch_per_episode)

plt.xlabel('Episode')

plt.ylabel('Epoch')

plt.show()
plt.plot(q_table_row)

plt.xlabel('Episode')

plt.ylabel('Q-Table Row')

plt.show()
# Extract only the action of the QTable

tmp_Q_table = Q_table.copy()

action_dict = dict()



for a in tmp_Q_table:

    if np.count_nonzero(tmp_Q_table) > 0:

        action_dict[a] = int(np.argmax(tmp_Q_table[a]))
agent_file = """def my_agent(observation, configuration):

    from random import choice

    

    q_table = """ + str(action_dict).replace(' ', '') + """

    

    board = observation.board[:]

    

    if tuple(board) not in q_table:

        return choice([c for c in range(configuration.columns) if observation.board[c] == 0])

        

    action = q_table[tuple(board)]

    

    if observation.board[action] == 0:

        return choice([c for c in range(configuration.columns) if observation.board[c] == 0])

        

    return action

    """
with open('submission.py', 'w') as sf:

    sf.write(agent_file)