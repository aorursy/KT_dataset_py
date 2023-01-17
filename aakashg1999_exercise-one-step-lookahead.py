from learntools.core import binder

binder.bind(globals())

from learntools.game_ai.ex2 import *
# TODO: Assign your values here

A = 10000000

B = 10000

C = 100

D = -1000

E = -100000



# Check your answer (this will take a few seconds to run!)

q_1.check()
# Lines below will give you a hint or solution code

#q_1.hint()

#q_1.solution()
#q_2.hint()
# Check your answer (Run this code cell to receive credit!)

q_2.solution()
import random



def score_move(grid, col, mark, config):

    next_grid = drop_piece(grid, col, mark, config)

    score = get_heuristic(next_grid, mark, config)

    return score



# Helper function for score_move: gets board at next step if agent drops piece in selected column

def drop_piece(grid, col, mark, config):

    next_grid = grid.copy()

    for row in range(config.rows-1, -1, -1):

        if next_grid[row][col] == 0:

            break

    next_grid[row][col] = mark

    return next_grid



# Helper function for score_move: calculates value of heuristic for grid



def get_heuristic(grid, col, mark, config):

    num_twos = count_windows(grid, 2, mark, config)

    num_threes = count_windows(grid, 3, mark, config)

    num_fours = count_windows(grid, 4, mark, config)

    num_twos_opp = count_windows(grid, 2, mark%2+1, config)

    num_threes_opp = count_windows(grid, 3, mark%2+1, config)

    score = 100000*num_fours + 1000*num_threes + 10*num_twos - 100*num_twos_opp - 10000*num_threes_opp

    return score



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



def score_move(grid, col, mark, config):

    next_grid = drop_piece(grid, col, mark, config)

    score = get_heuristic(next_grid, mark, config)

    return score



# Helper function for score_move: gets board at next step if agent drops piece in selected column

def drop_piece(grid, col, mark, config):

    next_grid = grid.copy()

    for row in range(config.rows-1, -1, -1):

        if next_grid[row][col] == 0:

            break

    next_grid[row][col] = mark

    return next_grid



# Helper function for score_move: calculates value of heuristic for grid



def get_heuristic(grid, col, mark, config):

    num_twos = count_windows(grid, 2, mark, config)

    num_threes = count_windows(grid, 3, mark, config)

    num_fours = count_windows(grid, 4, mark, config)

    num_twos_opp = count_windows(grid, 2, mark%2+1, config)

    num_threes_opp = count_windows(grid, 3, mark%2+1, config)

    score = 100000*num_fours + 1000*num_threes + 10*num_twos - 100*num_twos_opp - 10000*num_threes_opp

    return score



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



def my_agent(obs, config):

    # Your code here: Amend the agent!

    valid_moves = [col for col in range(config.columns) if obs.board[col] == 0]

    grid = np.asarray(obs.board).reshape(config.rows, config.columns)

    scores=dict(zip(valid_moves,[score_move(grid, col, obs.mark, config) for col in valid_moves]))

    max_cols=[keys for keys in scores.keys() if scores[keys]==max(scores.value())]

    return max_cols[-1]
# Run this code cell to get credit for creating an agent

q_3.check()
import inspect

import os



def write_agent_to_file(function, file):

    with open(file, "a" if os.path.exists(file) else "w") as f:

        f.write(inspect.getsource(function))

        print(function, "written to", file)



write_agent_to_file(my_agent, "submission.py")