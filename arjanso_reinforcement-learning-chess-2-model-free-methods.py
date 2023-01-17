%load_ext autoreload

%autoreload 2
import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import inspect
!pip install --upgrade git+https://github.com/arjangroen/RLC.git  # RLC is the Reinforcement Learning package
from RLC.move_chess.environment import Board

from RLC.move_chess.agent import Piece

from RLC.move_chess.learn import Reinforce
env = Board()

env.render()

env.visual_board
p = Piece(piece='king')
r = Reinforce(p,env)
print(inspect.getsource(r.monte_carlo_learning))
for k in range(100):

    eps = 0.5

    r.monte_carlo_learning(epsilon=eps)
r.visualize_policy()
r.agent.action_function.max(axis=2).astype(int)
print(inspect.getsource(r.sarsa_td))
p = Piece(piece='king')

env = Board()

r = Reinforce(p,env)

r.sarsa_td(n_episodes=10000,alpha=0.2,gamma=0.9)
r.visualize_policy()
print(inspect.getsource(r.sarsa_lambda))
p = Piece(piece='king')

env = Board()

r = Reinforce(p,env)

r.sarsa_lambda(n_episodes=10000,alpha=0.2,gamma=0.9)
r.visualize_policy()
print(inspect.getsource(r.sarsa_lambda))
p = Piece(piece='king')

env = Board()

r = Reinforce(p,env)

r.q_learning(n_episodes=1000,alpha=0.2,gamma=0.9)
r.visualize_policy()
r.agent.action_function.max(axis=2).round().astype(int)