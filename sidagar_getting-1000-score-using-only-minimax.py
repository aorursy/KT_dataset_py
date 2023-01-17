# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import random

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
def get_heuristic(grid, mark, config):

    score = 0

    for i in range(config.inarow):

        num  = count_windows (grid,i+1,mark,config)

        score += (4**(i+1))*num

    for i in range(config.inarow):

        num_opp = count_windows (grid,i+1,mark%2+1,config)

        score-= (2**((2*i)+3))*num_opp

    return score
def count_windows(grid, num_discs, piece, config):

    num_windows = 0

    # horizontal

    for row in range(config.rows):

        for col in range(config.columns-(config.inarow-1)):

            window = list(grid[row, col:col+config.inarow])

            if check_window(window, num_discs, piece, config):

                num_windows += 1

    # vertical

    for row in range(config.rows-(config.inarow-1)):

        for col in range(config.columns):

            window = list(grid[row:row+config.inarow, col])

            if check_window(window, num_discs, piece, config):

                num_windows += 1

    # positive diagonal

    for row in range(config.rows-(config.inarow-1)):

        for col in range(config.columns-(config.inarow-1)):

            window = list(grid[range(row, row+config.inarow), range(col, col+config.inarow)])

            if check_window(window, num_discs, piece, config):

                num_windows += 1

    # negative diagonal

    for row in range(config.inarow-1, config.rows):

        for col in range(config.columns-(config.inarow-1)):

            window = list(grid[range(row, row-config.inarow, -1), range(col, col+config.inarow)])

            if check_window(window, num_discs, piece, config):

                num_windows += 1

    return num_windows



def drop_piece(grid, col, mark, config):

    next_grid = grid.copy()

    for row in range(config.rows-1, -1, -1):

        if next_grid[row][col] == 0:

            break

    next_grid[row][col] = mark

    return next_grid



def check_window(window, num_discs, piece, config):

    return (window.count(piece) == num_discs and window.count(0) == config.inarow-num_discs)
def score_move_a(grid, col, mark, config,n_steps=1):

    next_grid = drop_piece(grid, col, mark, config)

    valid_moves = [col for col in range (config.columns) if next_grid[0][col]==0]

    if len(valid_moves)==0 or n_steps ==0:

        score = get_heuristic(next_grid, mark, config)

        return score

    else :

        scores = [score_move_b(next_grid,col,mark,config,n_steps-1) for col in valid_moves]

        score = min(scores)

    return score



def score_move_b(grid, col, mark, config,n_steps):

    next_grid = drop_piece(grid,col,(mark%2)+1,config)

    valid_moves = [col for col in range (config.columns) if next_grid[0][col]==0]

    if len(valid_moves)==0 or n_steps ==0:

        score = get_heuristic(next_grid, mark, config)

        return score

    else :

        scores = [score_move_a(next_grid,col,mark,config,n_steps-1) for col in valid_moves]

        score = max(scores)

    return score
def agent(obs, config):

    valid_moves = [c for c in range(config.columns) if obs.board[c] == 0]

    grid = np.asarray(obs.board).reshape(config.rows, config.columns)

    scores = dict(zip(valid_moves, [score_move_a(grid, col, obs.mark, config,1) for col in valid_moves]))

    max_cols = [key for key in scores.keys() if scores[key] == max(scores.values())]

    return random.choice(max_cols)
from kaggle_environments import make, evaluate
env = make("connectx", debug=True)

env.run([agent,"negamax"])

env.render(mode="ipython")
def get_win_percentages(agent1, agent2, n_rounds=100):

    config = {'rows': 10, 'columns': 7, 'inarow': 4}

    outcomes = evaluate("connectx", [agent1, agent2], config, [], n_rounds//2)

    outcomes += [[b,a] for [a,b] in evaluate("connectx", [agent2, agent1], config, [], n_rounds-n_rounds//2)]

    print("Agent 1 Win Percentage:", np.round(outcomes.count([1,-1])/len(outcomes), 2))

    print("Agent 2 Win Percentage:", np.round(outcomes.count([-1,1])/len(outcomes), 2))

    print("Number of Invalid Plays by Agent 1:", outcomes.count([None, 0]))

    print("Number of Invalid Plays by Agent 2:", outcomes.count([0, None]))
get_win_percentages(agent1=agent, agent2="negamax",n_rounds = 100)
def get_heuristic(grid, mark, config):

    score = 0

    for i in range(config.inarow):

        num  = count_windows (grid,i+1,mark,config)

        if (i==(config.inarow-1) and num >= 1):

            return float("inf")

        score += (4**(i+1))*num

    for i in range(config.inarow):

        num_opp = count_windows (grid,i+1,mark%2+1,config)

        if (i==(config.inarow-1) and num_opp >= 1):

            return float ("-inf")

        score-= (2**((2*i)+3))*num_opp

    return score
def score_move_a(grid, col, mark, config,n_steps=1):

    next_grid = drop_piece(grid, col, mark, config)

    valid_moves = [col for col in range (config.columns) if next_grid[0][col]==0]

    score = get_heuristic(next_grid, mark, config)

    #Since we have just dropped our piece there is only the possibility of us getting 4 in a row and not the opponent.

    #Thus score can only be +infinity.

    if len(valid_moves)==0 or n_steps ==0 or score == float("inf"):

        return score

    else :

        scores = [score_move_b(next_grid,col,mark,config,n_steps-1) for col in valid_moves]

        score = min(scores)

    return score



def score_move_b(grid, col, mark, config,n_steps):

    next_grid = drop_piece(grid,col,(mark%2)+1,config)

    valid_moves = [col for col in range (config.columns) if next_grid[0][col]==0]

    score = get_heuristic(next_grid, mark, config)

    #The converse is true here.

    #Since we have just dropped opponent piece there is only the possibility of opponent getting 4 in a row and not us.

    #Thus score can only be -infinity.

    if len(valid_moves)==0 or n_steps ==0 or score == float ("-inf"):

        return score

    else :

        scores = [score_move_a(next_grid,col,mark,config,n_steps-1) for col in valid_moves]

        score = max(scores)

    return score
def agent(obs, config):

    valid_moves = [c for c in range(config.columns) if obs.board[c] == 0]

    grid = np.asarray(obs.board).reshape(config.rows, config.columns)

    scores = dict(zip(valid_moves, [score_move_a(grid, col, obs.mark, config,1) for col in valid_moves]))

    max_cols = [key for key in scores.keys() if scores[key] == max(scores.values())]

    return random.choice(max_cols)
env = make("connectx", debug=True)

env.run([agent,"negamax"])

env.render(mode="ipython")
get_win_percentages(agent1=agent, agent2="negamax",n_rounds = 100)