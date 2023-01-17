# 1. Enable Internet in the Kernel (Settings side pane)



# 2. Curl cache may need purged if v0.1.4 cannot be found (uncomment if needed). 

# !curl -X PURGE https://pypi.org/simple/kaggle-environments



# ConnectX environment was defined in v0.1.4

!pip install 'kaggle-environments>=0.1.4'
from kaggle_environments import evaluate, make



env = make("connectx", debug=True)

env.render()
# This agent random chooses a non-empty column.

def my_agent(observation, configuration):

#     reward = calc_reward(observation, configuration)

    choice = 3

    print("config")

    print(str(configuration))

    print(configuration.rows)

    print(observation)

    print("choice = %d" % (choice))

    return choice
def reward_vertical(matrix, rows_count, columns_count, value, goal_len = 4, empty = 0):

    max_length = 0

    for col in range(columns_count):

        length = 0

        for row in range(rows_count - 1, 0, -1):

            if (matrix[row][col] == empty):

                break

            elif ((matrix[row][col] not in [value, empty]) & (length > 0)):

                length = 0

                break

            elif (length + row + 1 < goal_len):

                length = 0

                break

            elif (matrix[row][col] == value):

                length += 1

        # new col

        max_length = max(max_length, length)

    return max_length



def calc_reward(observation, configuration, done, agent_mark = 1, enemy_mark = 2):

    board = np.array(observation.board).reshape([configuration.rows, configuration.columns])

    

    agent_vertical_reward = reward_vertical(board, configuration.rows, configuration.columns, agent_mark)

    agent_reward = agent_vertical_reward

#     print("agent: vertical_reward = {}".format(agent_vertical_reward))

    

    enemy_vertical_reward = reward_vertical(board, configuration.rows, configuration.columns, enemy_mark)

    enemy_reward = enemy_vertical_reward

#     print("enemy: vertical_reward = {}".format(enemy_vertical_reward))

    

    reward = 0

    if (agent_reward == 4):

        reward = 1.0

    elif (agent_reward == enemy_reward):

        reward = 0.25

    elif (agent_reward - enemy_reward == 1):

        reward = 0.5

    elif (agent_reward - enemy_reward == 2):

        reward = 0.75

#     print("agent_reward = {}, enemy_reward = {}, result reward = {}".format(agent_reward, enemy_reward, reward))

    return reward
from kaggle_environments import Environment



class ConnectTrainer():

    """Connect Trainer

    """

    def __init__(self, env, configuration):

        self.env =  env

        self.trainer = self.env.train([None, "random"])

        self.configuration = configuration



    def step(self, action):

        action += 1

        observation, reward, done, info = self.trainer.step(action.item())

        new_reward = calc_reward(observation, self.configuration, done)

#         print("reward = {}, new_reward = {}, done = {}".format(reward, new_reward, done))

        observation = observation.board

#         print(np.array(observation).reshape(self.configuration.rows, self.configuration.columns))

        return observation, new_reward, done, info



    def reset(self):

        observation = self.trainer.reset()

        observation = observation.board

        return observation
import numpy as np

from matplotlib import pyplot



from keras.models import Sequential

from keras.layers import Dense, Activation, Flatten

from keras.optimizers import Adam



from rl.agents import SARSAAgent

from rl.policy import BoltzmannQPolicy



ENV_NAME = 'ConnectX-v0'



# Get the environment and extract the number of actions.

# Play as first position against random agent.

wrap_env = ConnectTrainer(env, env.configuration)

observation = wrap_env.reset()



nb_actions = env.configuration.columns - 1

input_array_size = env.configuration.columns * env.configuration.rows



# Next, we build a very simple model.

model = Sequential()

model.add(Flatten(input_shape=(1, input_array_size)))

model.add(Dense(512))

model.add(Activation('relu'))

model.add(Dense(512))

model.add(Activation('relu'))

model.add(Dense(512))

model.add(Activation('relu'))

model.add(Dense(nb_actions))

model.add(Activation('linear'))

print(model.summary())



# Init memory and model agent

policy = BoltzmannQPolicy()

sarsa = SARSAAgent(model=model, nb_actions=nb_actions, nb_steps_warmup=10, policy=policy)

sarsa.compile(Adam(lr=1e-3), metrics=['mae'])



# Train

train_history = sarsa.fit(wrap_env, nb_steps=50, visualize=False, verbose=2)



# Plot history

train_rewards = train_history.history['episode_reward']

pyplot.plot(train_rewards)



# After training is done, we save the final weights.

sarsa.save_weights('sarsa_{}_weights.h5f'.format(ENV_NAME), overwrite=True)
env.reset()

# Play as the first agent against default "random" agent.

env.run([my_agent, "random"])

env.render(mode="ipython", width=500, height=450)
# Play as first position against random agent.

trainer = env.train([None, "random"])



observation = trainer.reset()



while not env.done:

    my_action = my_agent(observation, env.configuration)

    print("My Action", my_action)

    observation, reward, done, info = trainer.step(my_action)

    print("observ = {}, reward = {}, done = {}, info = {}".format(observation, reward, done, info))

    # env.render(mode="ipython", width=100, height=90, header=False, controls=False)

env.render()
def mean_reward(rewards):

    return sum(r[0] for r in rewards) / sum(r[0] + r[1] for r in rewards)



# Run multiple episodes to estimate it's performance.

print("My Agent vs Random Agent:", mean_reward(evaluate("connectx", [my_agent, "random"], num_episodes=10)))

print("My Agent vs Negamax Agent:", mean_reward(evaluate("connectx", [my_agent, "negamax"], num_episodes=10)))
import inspect

import os



def write_agent_to_file(function, file):

    with open(file, "a" if os.path.exists(file) else "w") as f:

        f.write(inspect.getsource(function))

        print(function, "written to", file)



write_agent_to_file(my_agent, "submission.py")