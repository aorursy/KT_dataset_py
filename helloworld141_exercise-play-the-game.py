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

#q_1.hint()

#q_1.solution()
def agent_q2(obs, config):

    # Your code here: Amend the agent!

    valid_moves = [col for col in range(config.columns) if obs.board[col] == 0] # first config.cols columns are the top of the board

    for move in valid_moves:

        if check_winning_move(obs, config, move, obs.mark):

            return move

    for move in valid_moves:

        if check_winning_move(obs, config, move, 3-obs.mark):

            return move # block oponent

    return random.choice(valid_moves)



# Check your answer

q_2.check()
# Lines below will give you a hint or solution code

#q_2.hint()

#q_2.solution()
#q_3.hint()
# Check your answer (Run this code cell to receive credit!)

q_3.solution()
def my_agent(obs, config):

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

    # Your code here: Amend the agent!

    valid_moves = [col for col in range(config.columns) if obs.board[col] == 0] # first config.cols columns are the top of the board

    for move in valid_moves:

        if check_winning_move(obs, config, move, obs.mark):

            return move

    for move in valid_moves:

        if check_winning_move(obs, config, move, 3-obs.mark):

            return move # block oponent

    return random.choice(valid_moves)
# Run this code cell to get credit for creating an agent

q_4.check()
from kaggle_environments import evaluate, make, utils



env = make("connectx", debug=True)

env.play([my_agent, None], width=500, height=450)
def get_win_percentages(agent1, agent2, n_rounds=100):

    # Use default Connect Four setup

    config = {'rows': 6, 'columns': 7, 'inarow': 4}

    # Agent 1 goes first (roughly) half the time          

    outcomes = evaluate("connectx", [agent1, agent2], config, [], n_rounds//2)

    # Agent 2 goes first (roughly) half the time      

    outcomes += [[b,a] for [a,b] in evaluate("connectx", [agent2, agent1], config, [], n_rounds-n_rounds//2)]

    print("Agent 1 Win Percentage:", np.round(outcomes.count([1,0])/len(outcomes), 2))

    print("Agent 2 Win Percentage:", np.round(outcomes.count([0,1])/len(outcomes), 2))

    print("Number of Invalid Plays by Agent 1:", outcomes.count([None, 0.5]))

    print("Number of Invalid Plays by Agent 2:", outcomes.count([0.5, None]))

    print("Number of Draws (in {} game rounds):".format(n_rounds), outcomes.count([0.5, 0.5]))

    

def agent_random(obs, config):

    valid_moves = [col for col in range(config.columns) if obs.board[col] == 0]

    return random.choice(valid_moves)
get_win_percentages(agent1=my_agent, agent2=agent_random)
import inspect

import os



def write_agent_to_file(function, file):

    with open(file, "a" if os.path.exists(file) else "w") as f:

        f.write(inspect.getsource(function))

        print(function, "written to", file)



write_agent_to_file(my_agent, "submission.py")



# Check that submission file was created

q_5.check()