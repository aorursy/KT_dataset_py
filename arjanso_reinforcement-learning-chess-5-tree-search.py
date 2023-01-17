%load_ext autoreload

%autoreload 2

!pip install python-chess  # Python-Chess is the Python Chess Package that handles the chess environment

!pip install --upgrade git+https://github.com/arjangroen/RLC.git  # RLC is the Reinforcement Learning package

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)import os

import inspect

import matplotlib.pyplot as plt
from keras.models import load_model
import os

os.listdir('../input')
from RLC.real_chess import agent, environment, learn, tree

import chess

from chess.pgn import Game





opponent = agent.GreedyAgent()

env = environment.Board(opponent, FEN=None)

player = agent.Agent(lr=0.0005,network='big')

learner = learn.TD_search(env, player,gamma=0.9,search_time=0.9)

node = tree.Node(learner.env.board, gamma=learner.gamma)

player.model.summary()
n_iters = 10000  # maximum number of iterations

timelimit = 25000 # maximum time for learning

network_replacement_interval = 10  # For the stability of the nearal network updates, the network is not continuously replaced
learner.learn(iters=n_iters,timelimit_seconds=timelimit,c=network_replacement_interval) 
reward_smooth = pd.DataFrame(learner.reward_trace)

reward_smooth.rolling(window=1000,min_periods=1000).mean().plot(figsize=(16,9),title='average reward over the last 1000 steps')

plt.show()
reward_smooth = pd.DataFrame(learner.piece_balance_trace)

reward_smooth.rolling(window=50,min_periods=50).mean().plot(figsize=(16,9),title='average piece balance over the last 50 episodes')

plt.show()
learner.env.reset()

learner.search_time = 60

learner.temperature = 1/3
learner.play_game(n_iters,maxiter=128)
pgn = Game.from_board(learner.env.board)

with open("rlc_pgn","w") as log:

    log.write(str(pgn))
learner.agent.model.save('RLC_model.h5')