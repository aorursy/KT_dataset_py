# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



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
import numpy as np

import random

def score_move_a(grid, col, mark, config, start_score, n_steps):

    next_grid, pos = drop_piece(grid, col, mark, config)

    row, col = pos

    score = get_heuristic_optimised(grid,next_grid,mark,config, row, col,start_score)

    valid_moves = [col for col in range (config.columns) if next_grid[0][col]==0]

    #Since we have just dropped our piece there is only the possibility of us getting 4 in a row and not the opponent.

    #Thus score can only be +infinity.

    scores = []

    if len(valid_moves)==0 or n_steps ==0 or score == float("inf"):

        return score

    else :

        for col in valid_moves:

            current = score_move_b(next_grid,col,mark,config,score,n_steps-1)

            scores.append(current)

        score = min(scores)

    return score



def score_move_b(grid, col, mark, config, start_score, n_steps):

    next_grid, pos = drop_piece(grid,col,(mark%2)+1,config)

    row, col = pos

    score = get_heuristic_optimised(grid,next_grid,mark,config, row, col,start_score)

    valid_moves = [col for col in range (config.columns) if next_grid[0][col]==0]

    

    #The converse is true here.

    #Since we have just dropped opponent piece there is only the possibility of opponent getting 4 in a row and not us.

    #Thus score can only be -infinity.

    scores = []

    if len(valid_moves)==0 or n_steps ==0 or score == float ("-inf"):

        return score

    else :

        for col in valid_moves:

            current = score_move_a (next_grid,col,mark,config,score,n_steps-1)

            scores.append(current)

        score = max(scores)

    return score



def drop_piece(grid, col, mark, config):

    next_grid = grid.copy()

    for row in range(config.rows-1, -1, -1):

        if next_grid[row][col] == 0:

            break

    next_grid[row][col] = mark

    return next_grid,(row,col)



def get_heuristic(grid, mark, config):

    score = 0

    num = count_windows(grid,mark,config)

    for i in range(config.inarow):

        #num  = count_windows (grid,i+1,mark,config)

        if (i==(config.inarow-1) and num[i+1] >= 1):

            return float("inf")

        score += (4**(i))*num[i+1]

    num_opp = count_windows (grid,mark%2+1,config)

    for i in range(config.inarow):

        if (i==(config.inarow-1) and num_opp[i+1] >= 1):

            return float ("-inf")

        score-= (2**((2*i)+1))*num_opp[i+1]

    return score



def get_heuristic_optimised(grid, next_grid, mark, config, row, col, start_score):

    score = 0

    num1 = count_windows_optimised(grid,mark,config,row,col)

    num2 = count_windows_optimised(next_grid,mark,config,row,col)

    for i in range(config.inarow):

        if (i==(config.inarow-1) and (num2[i+1]-num1[i+1]) >= 1):

            return float("inf")

        score += (4**(i))*(num2[i+1]-num1[i+1])

    num1_opp = count_windows_optimised(grid,mark%2+1,config,row,col)

    num2_opp = count_windows_optimised(next_grid,mark%2+1,config,row,col)

    for i in range(config.inarow): 

        if (i==(config.inarow-1) and num2_opp[i+1]-num1_opp[i+1]  >= 1):

            return float ("-inf")     

        score-= (2**((2*i)+1))*(num2_opp[i+1]-num1_opp[i+1])

    score+= start_score

    return score



def check_window(window, piece, config):

    if window.count((piece%2)+1)==0:

        return window.count(piece)

    else:

        return -1



def count_windows(grid, piece, config):

    num_windows = np.zeros(config.inarow+1)

    # horizontal

    for row in range(config.rows):

        for col in range(config.columns-(config.inarow-1)):

            window = list(grid[row, col:col+config.inarow])

            type_window = check_window(window, piece, config)

            if type_window != -1:

                num_windows[type_window] += 1

    # vertical

    for row in range(config.rows-(config.inarow-1)):

        for col in range(config.columns):

            window = list(grid[row:row+config.inarow, col])

            type_window = check_window(window, piece, config)

            if type_window != -1:

                num_windows[type_window] += 1

    # positive diagonal

    for row in range(config.rows-(config.inarow-1)):

        for col in range(config.columns-(config.inarow-1)):

            window = list(grid[range(row, row+config.inarow), range(col, col+config.inarow)])

            type_window = check_window(window, piece, config)

            if type_window != -1:

                num_windows[type_window] += 1

    # negative diagonal

    for row in range(config.inarow-1, config.rows):

        for col in range(config.columns-(config.inarow-1)):

            window = list(grid[range(row, row-config.inarow, -1), range(col, col+config.inarow)])

            type_window = check_window(window, piece, config)

            if type_window != -1:

                num_windows[type_window] += 1

    return num_windows



def count_windows_optimised(grid, piece, config, row, col):

    num_windows = np.zeros(config.inarow+1)

    # horizontal

    for acol in range(max(0,col-(config.inarow-1)),min(col+1,(config.columns-(config.inarow-1)))):

        window = list(grid[row, acol:acol+config.inarow])

        type_window = check_window(window, piece, config)

        if type_window != -1:

            num_windows[type_window] += 1

    # vertical

    for arow in range(max(0,row-(config.inarow-1)),min(row+1,(config.rows-(config.inarow-1)))):

        window = list(grid[arow:arow+config.inarow, col])

        type_window = check_window(window, piece, config)

        if type_window != -1:

            num_windows[type_window] += 1

    # positive diagonal

    for arow, acol in zip(range(row-(config.inarow-1),row+1),range(col-(config.inarow-1),col+1)):

        if (arow>=0 and acol>=0 and arow<=(config.rows-config.inarow) and acol<=(config.columns-config.inarow)):

            window = list(grid[range(arow, arow+config.inarow), range(acol, acol+config.inarow)])

            type_window = check_window(window, piece, config)

            if type_window != -1:

                num_windows[type_window] += 1

    # negative diagonal

    for arow,acol in zip(range(row,row+config.inarow),range(col,col-config.inarow,-1)):

        if (arow >= (config.inarow-1) and acol >=0 and arow <= (config.rows-1) and acol <= (config.columns-config.inarow)):

            window = list(grid[range(arow, arow-config.inarow, -1), range(acol, acol+config.inarow)])

            type_window = check_window(window, piece, config)

            if type_window != -1:

                num_windows[type_window] += 1

    return num_windows



def agent(obs, config):

    grid = np.asarray(obs.board).reshape(config.rows, config.columns)

    valid_moves = [c for c in range(config.columns) if grid[0][c] == 0]

    scores = {}

    start_score = get_heuristic(grid, obs.mark, config)

    for col in valid_moves:

        scores[col] = score_move_a(grid, col, obs.mark, config,start_score,2)

    print ("2 Step lookahead agent:",scores)

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