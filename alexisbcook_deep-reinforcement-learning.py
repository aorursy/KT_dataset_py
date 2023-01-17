

import random

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline


!pip install 'tensorflow==1.15.0'
# Check version of tensorflow

import tensorflow as tf

tf.__version__
from kaggle_environments import make, evaluate

from gym import spaces



class ConnectFourGym:

    def __init__(self, agent2="random"):

        ks_env = make("connectx", debug=True)

        self.env = ks_env.train([None, agent2])

        self.rows = ks_env.configuration.rows

        self.columns = ks_env.configuration.columns

        # Learn about spaces here: http://gym.openai.com/docs/#spaces

        self.action_space = spaces.Discrete(self.columns)

        self.observation_space = spaces.Box(low=0, high=2, 

                                            shape=(self.rows,self.columns,1), dtype=np.int)

        # Tuple corresponding to the min and max possible rewards

        self.reward_range = (-10, 1)

        # StableBaselines throws error if these are not defined

        self.spec = None

        self.metadata = None

    def reset(self):

        self.obs = self.env.reset()

        return np.array(self.obs['board']).reshape(self.rows,self.columns,1)

    def change_reward(self, old_reward, done):

        if old_reward == 1: # The agent won the game

            return 1

        elif done: # The opponent won the game

            return -1

        else: # Reward 1/42

            return 1/(self.rows*self.columns)

    def step(self, action):

        # Check if agent's move is valid

        is_valid = (self.obs['board'][int(action)] == 0)

        if is_valid: # Play the move

            self.obs, old_reward, done, _ = self.env.step(int(action))

            reward = self.change_reward(old_reward, done)

        else: # End the game and penalize agent

            reward, done, _ = -10, True, {}

        return np.array(self.obs['board']).reshape(self.rows,self.columns,1), reward, done, _
# Create ConnectFour environment

env = ConnectFourGym(agent2="random")


!apt-get update

!apt-get install -y cmake libopenmpi-dev python3-dev zlib1g-dev

!pip install "stable-baselines[mpi]==2.9.0"
import os

from stable_baselines.bench import Monitor 

from stable_baselines.common.vec_env import DummyVecEnv



# Create directory for logging training information

log_dir = "ppo/"

os.makedirs(log_dir, exist_ok=True)



# Logging progress

monitor_env = Monitor(env, log_dir, allow_early_resets=True)



# Create a vectorized environment

vec_env = DummyVecEnv([lambda: monitor_env])
from stable_baselines import PPO1 

from stable_baselines.a2c.utils import conv, linear, conv_to_fc

from stable_baselines.common.policies import CnnPolicy



# Neural network for predicting action values

def modified_cnn(scaled_images, **kwargs):

    activ = tf.nn.relu

    layer_1 = activ(conv(scaled_images, 'c1', n_filters=32, filter_size=3, stride=1, 

                         init_scale=np.sqrt(2), **kwargs))

    layer_2 = activ(conv(layer_1, 'c2', n_filters=64, filter_size=3, stride=1, 

                         init_scale=np.sqrt(2), **kwargs))

    layer_2 = conv_to_fc(layer_2)

    return activ(linear(layer_2, 'fc1', n_hidden=512, init_scale=np.sqrt(2)))  



class CustomCnnPolicy(CnnPolicy):

    def __init__(self, *args, **kwargs):

        super(CustomCnnPolicy, self).__init__(*args, **kwargs, cnn_extractor=modified_cnn)

        

# Initialize agent

model = PPO1(CustomCnnPolicy, vec_env, verbose=0)
# Train agent

model.learn(total_timesteps=100000)



# Plot cumulative reward

with open(os.path.join(log_dir, "monitor.csv"), 'rt') as fh:    

    firstline = fh.readline()

    assert firstline[0] == '#'

    df = pd.read_csv(fh, index_col=None)['r']

df.rolling(window=1000).mean().plot()

plt.show()
def agent1(obs, config):

    # Use the best model to select a column

    col, _ = model.predict(np.array(obs['board']).reshape(6,7,1))

    # Check if selected column is valid

    is_valid = (obs['board'][int(col)] == 0)

    # If not valid, select random move. 

    if is_valid:

        return int(col)

    else:

        return random.choice([col for col in range(config.columns) if obs.board[int(col)] == 0])
# Create the game environment

env = make("connectx")



# Two random agents play one game round

env.run([agent1, "random"])



# Show the game

env.render(mode="ipython")


def get_win_percentages(agent1, agent2, n_rounds=100):

    # Use default Connect Four setup

    config = {'rows': 6, 'columns': 7, 'inarow': 4}

    # Agent 1 goes first (roughly) half the time          

    outcomes = evaluate("connectx", [agent1, agent2], config, [], n_rounds//2)

    # Agent 2 goes first (roughly) half the time      

    outcomes += [[b,a] for [a,b] in evaluate("connectx", [agent2, agent1], config, [], n_rounds-n_rounds//2)]

    print("Agent 1 Win Percentage:", np.round(outcomes.count([1,-1])/len(outcomes), 2))

    print("Agent 2 Win Percentage:", np.round(outcomes.count([-1,1])/len(outcomes), 2))

    print("Number of Invalid Plays by Agent 1:", outcomes.count([None, 0]))

    print("Number of Invalid Plays by Agent 2:", outcomes.count([0, None]))
get_win_percentages(agent1=agent1, agent2="random")