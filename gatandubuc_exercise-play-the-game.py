from learntools.core import binder

binder.bind(globals())

from learntools.game_ai.ex1 import *

from learntools.game_ai.ex1 import MyBoard

import random
# Lines below will give you a hint or solution code

#q_1.hint()

q_1.solution()
import numpy as np



# Gets board at next step if agent drops piece in selected column

def drop_piece(grid, col, piece, config):

    next_grid = grid.copy()

    for row in range(config.rows-1, -1, -1):

        if next_grid[row][col] == 0:

            break

    next_grid[row][col] = piece

    return next_grid



# Returns True if dropping piece in column results in game win

def check_winning_move2(obs, config, col, piece, inarow=4):

    # Convert the board to a 2D grid

    grid = np.asarray(obs.board).reshape(config.rows, config.columns)

    next_grid = drop_piece(grid, col, piece, config)

    if inarow == 1:

        return True

    else:

        # horizontal

        for row in range(config.rows):

            for col in range(config.columns-(inarow-1)):

                window = list(next_grid[row,col:col+inarow])

                if window.count(piece) == inarow:

                    if inarow in [2,3] and col in [0,6]:

                        return False

                    return True

        # vertical

        for row in range(config.rows-(inarow-1)):

            for col in range(config.columns):

                window = list(next_grid[row:row+inarow,col])

                if window.count(piece) == inarow:

                    if inarow in [2,3] and np.sum(next_grid[:,col]==0) in [2,3]:

                        return False

                    return True

        # positive diagonal

        for row in range(config.rows-(inarow-1)):

            for col in range(config.columns-(inarow-1)):

                window = list(next_grid[range(row, row+inarow), range(col, col+inarow)])

                if window.count(piece) == inarow:

                    return True

        # negative diagonal

        for row in range(config.inarow-1, config.rows):

            for col in range(config.columns-(inarow-1)):

                window = list(next_grid[range(row, row-inarow, -1), range(col, col+inarow)])

                if window.count(piece) == inarow:

                    return True

    return False
def agent_q2(obs, config):

    valid_moves = [col for col in range(config.columns) if obs.board[col] == 0]

    for col in valid_moves:

        if check_winning_move2(obs, config, col, obs.mark):

            return col

    for col in valid_moves:

        if check_winning_move2(obs, config, col, obs.mark%2+1):

            return col

    return random.choice(valid_moves)

    

q_2.check()
# Lines below will give you a hint or solution code

#q_2.hint()

q_2.solution()
q_3.hint()
# Check your answer (Run this code cell to receive credit!)

q_3.solution()
def my_agent(obs, config):

    # Your code here: Amend the agent!

    import random

    import numpy as np

    import pandas as pd

    

    def get_heuristic(grid, mark, config):

        A = 100000

        B = 50

        C = 1

        D = -2

        E = -100

        F = -1000000

        num_twos = count_windows(grid, 2, mark, config)

        num_threes = count_windows(grid, 3, mark, config)

        num_fours = count_windows(grid, 4, mark, config)

        num_twos_opp = count_windows(grid, 2, mark%2+1, config)

        num_threes_opp = count_windows(grid, 3, mark%2+1, config)

        num_four_opp = count_windows(grid, 4, mark%2+1, config)

        score = A*num_fours + B*num_threes + C*num_twos + D*num_twos_opp + E*num_threes_opp + F*num_four_opp

        return score

    

    # Calculates score if agent drops piece in selected column

    def score_move(grid, col, mark, config):

        next_grid = drop_piece(grid, col, mark, config)

        score = get_heuristic(next_grid, mark=mark, config=config)

        return score



    # Helper function for score_move: gets board at next step if agent drops piece in selected column

    def drop_piece(grid, col, mark, config):

        next_grid = grid.copy()

        for row in range(config.rows-1, -1, -1):

            if next_grid[row][col] == 0:

                break

        next_grid[row][col] = mark

        return next_grid

    

    # Helper function for get_heuristic: checks if window satisfies heuristic conditions

    def check_window(window, num_discs, piece, config):

        return (window.count(piece) == num_discs and window.count(0) == config.inarow-num_discs)



    # Helper function for get_heuristic: counts number of windows satisfying specified heuristic conditions

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

    

    # Get list of valid moves

    valid_moves = [c for c in range(config.columns) if obs.board[c] == 0]

    # Convert the board to a 2D grid

    grid = np.asarray(obs.board).reshape(config.rows, config.columns)

    # Use the heuristic to assign a score to each possible board in the next turn

    scores = dict(zip(valid_moves, [score_move(grid, col, obs.mark, config) for col in valid_moves]))

    # Get a list of columns (moves) that maximize the heuristic

    max_cols = [key for key in scores.keys() if scores[key] == max(scores.values())]

    # Select at random from the maximizing columns

    return random.choice(max_cols)
import pandas as pd



def sort_prior(liste):



    result = pd.Series(liste).value_counts()

    if result.nunique != 1:

        return int(result.index[0])

    else:

        return random.choice(result.index)

        

    

    

def my_agent2(obs, config):

    

    dico_valid_moves = {'valid_moves_'+str(moves):[col for col in range(config.columns) if obs.board[col] == 0 and check_winning_move2(obs=obs,

                                                                                                      config=config,

                                                                                                      col=col,

                                                                                                      piece=obs.mark,

                                                                                                      inarow=moves

                                                                                                     )] for moves in range(1,5)}

    dico_opponent_moves = {'opponent_moves_'+str(moves):[col for col in range(config.columns) if obs.board[col] == 0 and check_winning_move2(obs=obs,

                                                                                                      config=config,

                                                                                                      col=col,

                                                                                                      piece=obs.mark%2+1,

                                                                                                      inarow=moves

                                                                                                     )] for moves in range(1,5)}

    i = 4

    if len(dico_valid_moves['valid_moves_'+str(i)])==0:

        if len(dico_opponent_moves['opponent_moves_'+str(i)])>0:

            return sort_prior(dico_opponent_moves['opponent_moves_'+str(i)])

    else:

        return sort_prior(dico_valid_moves['valid_moves_'+str(i)])

    i = 3

    while i!=1:   

        if len(dico_opponent_moves['opponent_moves_'+str(i)])>0:

            for col in dico_opponent_moves['opponent_moves_'+str(i)]:

                if col in dico_valid_moves['valid_moves_'+str(i)]:

                    return col

            return sort_prior(dico_opponent_moves['opponent_moves_'+str(i)])

        elif len(dico_valid_moves['valid_moves_'+str(i)])>0:

            return sort_prior(dico_valid_moves['valid_moves_'+str(i)])

        i = i-1



    return random.choice(dico_valid_moves['valid_moves_1'])
# Run this code cell to get credit for creating an agent

q_4.check()
from kaggle_environments import evaluate, make, utils



env = make("connectx", debug=True)

env.play([my_agent, None], width=500, height=450)
import inspect

import os



def write_agent_to_file(function, file):

    with open(file, "a" if os.path.exists(file) else "w") as f:

        f.write(inspect.getsource(function))

        print(function, "written to", file)



write_agent_to_file(my_agent, "submission.py")



# Check that submission file was created

q_5.check()