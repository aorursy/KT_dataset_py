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

    # Your code here: Amend the agent!

    for col in valid_moves:

        if check_winning_move(obs, config, col, obs.mark):

            return col

    return random.choice(valid_moves)

    

# Check your answer

q_1.check()
# Lines below will give you a hint or solution code

# q_1.hint()

# q_1.solution()
def agent_q2(obs, config):

    # Your code here: Amend the agent!

    valid_moves = [col for col in range(config.columns) if obs.board[col] == 0]

    for col in valid_moves:

        if check_winning_move(obs, config, col, obs.mark):

            return col

    for col in valid_moves:

        if check_winning_move(obs, config, col, obs.mark%2+1):

            return col

    return 0 



# Check your answer

q_2.check()
# Lines below will give you a hint or solution code

# q_2.hint()

# q_2.solution()
# q_3.hint()
# Check your answer (Run this code cell to receive credit!)

q_3.solution()
def my_agent(obs, config):

    # Your code here: Amend the agent!

    import numpy as np

    import random

    

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



    valid_moves = [col for col in range(config.columns) if obs.board[col] == 0]

    

    for col in valid_moves:

        if check_winning_move(obs, config, col, obs.mark):

            return col

    for col in valid_moves:

        if check_winning_move(obs, config, col, obs.mark%2+1):

            return col

    

    return random.choice(valid_moves)
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