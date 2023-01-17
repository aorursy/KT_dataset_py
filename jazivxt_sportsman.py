!pip install kaggle-environments
from kaggle_environments import make

from IPython.core.display import HTML



env = make("connectx")

print(env.name, env.version)

print("Default Agents: ", *env.agents)
%%writefile submission.py

import numpy as np

player_one = True



def find_winnig_moves(board):

    moves = []

    return moves



def find_defending_moves(board):

    moves = []

    return moves



def agent(observation, configuration):

    global player_one

    moves = []

    board = observation.board

    board = np.array(board).reshape(configuration.rows,configuration.columns)

    

    if np.sum(board) == 1: player_one = False

    if np.sum(board) < 2: print('Player 1:', player_one)

    

    if np.sum(board) < 7: #Random Start

        moves += [int(np.random.choice(np.arange(configuration.columns).astype(int)))]

    else:

        moves = find_winnig_moves(board)

        moves += find_defending_moves(board)



    for i in range(configuration.columns): #Add Any Open

        if np.sum([1 for m in board[:,i] if m ==0])>0:

            moves +=[i]

    return moves[0]
%run submission.py
env = make("connectx", debug=True)

env.run([agent, "random"])

HTML(env.render(mode="ipython", width=600, height=500, header=False))
env.run(["random", agent])

HTML(env.render(mode="ipython", width=600, height=500, header=False))