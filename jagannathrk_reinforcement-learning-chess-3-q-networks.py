import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

import inspect
%load_ext autoreload

%autoreload 2
!pip install python-chess  # Python-Chess is the Python Chess Package that handles the chess environment

!pip install --upgrade git+https://github.com/arjangroen/RLC.git  # RLC is the Reinforcement Learning package
import chess

from chess.pgn import Game

import RLC

from RLC.capture_chess.environment import Board

from RLC.capture_chess.learn import Q_learning

from RLC.capture_chess.agent import Agent
board = Board()

board.board
board.layer_board[0,::-1,:].astype(int)
board = Board()

agent = Agent(network='conv',gamma=0.1,lr=0.07)

R = Q_learning(agent,board)

R.agent.fix_model()

R.agent.model.summary()
print(inspect.getsource(agent.network_update))
print(inspect.getsource(R.play_game))
pgn = R.learn(iters=750)
reward_smooth = pd.DataFrame(R.reward_trace)

reward_smooth.rolling(window=125,min_periods=0).mean().plot(figsize=(16,9),title='average performance over the last 125 steps')
with open("final_game.pgn","w") as log:

    log.write(str(pgn))
board.reset()

bl = board.layer_board

bl[6,:,:] = 1/10  # Assume we are in move 10

av = R.agent.get_action_values(np.expand_dims(bl,axis=0))



av = av.reshape((64,64))



p = board.board.piece_at(20)#.symbol()





white_pieces = ['P','N','B','R','Q','K']

black_piece = ['_','p','n','b','r','q','k']



df = pd.DataFrame(np.zeros((6,7)))



df.index = white_pieces

df.columns = black_piece



for from_square in range(16):

    for to_square in range(30,64):

        from_piece = board.board.piece_at(from_square).symbol()

        to_piece = board.board.piece_at(to_square)

        if to_piece:

            to_piece = to_piece.symbol()

        else:

            to_piece = '_'

        df.loc[from_piece,to_piece] = av[from_square,to_square]

        

        
df[['_','p','n','b','r','q']]