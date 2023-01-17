from learntools.core import binder

binder.bind(globals())

from learntools.game_ai.ex1 import *
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

def check_winning_move(obs, config, col, piece):

    # Convert the board to a 2D grid

    grid = np.asarray(obs.board).reshape(config.rows, config.columns)

    next_grid = drop_piece(grid, col, piece, config)

    # horizontal

    for row in range(config.rows):

        for col in range(config.columns-(config.inarow-1)):

            window = list(next_grid[row,col:col+config.inarow])

            if window.count(piece) == config.inarow:

                return True

    # vertical

    for row in range(config.rows-(config.inarow-1)):

        for col in range(config.columns):

            window = list(next_grid[row:row+config.inarow,col])

            if window.count(piece) == config.inarow:

                return True

    # positive diagonal

    for row in range(config.rows-(config.inarow-1)):

        for col in range(config.columns-(config.inarow-1)):

            window = list(next_grid[range(row, row+config.inarow), range(col, col+config.inarow)])

            if window.count(piece) == config.inarow:

                return True

    # negative diagonal

    for row in range(config.inarow-1, config.rows):

        for col in range(config.columns-(config.inarow-1)):

            window = list(next_grid[range(row, row-config.inarow, -1), range(col, col+config.inarow)])

            if window.count(piece) == config.inarow:

                return True

    return False
import random



def agent_q1(obs, config):

    valid_moves = [col for col in range(config.columns) if obs.board[col] == 0]

    for m in valid_moves :

        if check_winning_move(obs, config, m, obs.mark):

            return m

    return random.choice(valid_moves)

    

# Check your answer

q_1.check()
# Lines below will give you a hint or solution code

q_1.hint()

q_1.solution()
def agent_q2(obs, config):

    valid_moves = [col for col in range(config.columns) if obs.board[col] == 0]

    for m in valid_moves :

        if check_winning_move(obs, config, m, obs.mark):

            return m

    for m in valid_moves :

        if check_winning_move(obs, config, m, 3-obs.mark):

            return m

    return random.choice(valid_moves) 



# Check your answer

q_2.check()
# Lines below will give you a hint or solution code

q_2.hint()

q_2.solution()
q_3.hint()
# Check your answer (Run this code cell to receive credit!)

q_3.solution()
def my_agent(obs, config):

    # Imports and helper functions

    import numpy as np

    import random

    

    # Gets board at next step if agent drops piece in selected column

    def drop_piece(grid, col, piece, config):

        next_grid = grid.copy()

        for row in range(config.rows-1, -1, -1):

            if next_grid[row][col] == 0:

                break

        next_grid[row][col] = piece

        return next_grid

    

    # Calculates score if agent drops piece in selected column

    def score_move(grid, col, mark, config):

        next_grid = drop_piece(grid, col, mark, config)

        score = get_heuristic(next_grid, mark, config)

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



    # get heuristics for a given grid

    def get_heuristic(grid, mark, config):

        A = 1000000

        B = 110

        C = 20

        D = -20

        C = -1000

        num_twos = count_windows(grid, 2, mark, config)

        num_threes = count_windows(grid, 3, mark, config)

        num_fours = count_windows(grid, 4, mark, config)

        num_twos_opp = count_windows(grid, 2, mark%2+1, config)

        num_threes_opp = count_windows(grid, 3, mark%2+1, config)

        score = A*num_fours + B*num_threes + C*num_twos + D*num_twos_opp + E*num_threes_opp

        return score  

    

    # Agent makes selection          

    valid_moves = [col for col in range(config.columns) if obs.board[col] == 0]

       

    # Convert board to a 2D grid

    grid = np.asarray(obs.board).reshape(config.rows, config.columns)

    

    # Use the heuristic to assign a score to each possible board in the next turn

    scores = dict(zip(valid_moves, [score_move(grid, col, obs.mark, config) for col in valid_moves]))

    

    # Get a list of columns (moves) that maximize the heuristic

    max_cols = [key for key in scores.keys() if scores[key] == max(scores.values())]

    

    # Return a random move from the maximizing columns

    return random.choice(max_cols)
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