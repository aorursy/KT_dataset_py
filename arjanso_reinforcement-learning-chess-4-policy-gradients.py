%load_ext autoreload

%autoreload 2

!pip install python-chess  # Python-Chess is the Python Chess Package that handles the chess environment

!pip install --upgrade git+https://github.com/arjangroen/RLC.git  # RLC is the Reinforcement Learning package

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)import os

import inspect
import chess

from chess.pgn import Game

board = chess.Board()

import RLC

from RLC.capture_chess.environment import Board

from RLC.capture_chess.learn import Reinforce, ActorCritic

from RLC.capture_chess.agent import Agent, policy_gradient_loss

board = Board()

agent = Agent(network='conv_pg',lr=0.3)



R = Reinforce(agent,board)
print(inspect.getsource(policy_gradient_loss))
pgn = R.learn(iters=2000)
with open("final_game.pgn","w") as log:

    log.write(str(pgn))

    
board = Board()



critic = Agent(network='conv',lr=0.1)

critic.fix_model()

actor = Agent(network='conv_pg',lr=0.3)



R = ActorCritic(actor, critic,board)
pgn = R.learn(iters=1000)
reward_smooth = pd.DataFrame(R.reward_trace)

reward_smooth.rolling(window=100,min_periods=0).mean().plot(figsize=(16,9))