

import random

import numpy as np
# Calculates score if agent drops piece in selected column

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

def get_heuristic(grid, mark, config):

    num_threes = count_windows(grid, 3, mark, config)

    num_fours = count_windows(grid, 4, mark, config)

    num_threes_opp = count_windows(grid, 3, mark%2+1, config)

    score = num_threes - 1e2*num_threes_opp + 1e6*num_fours

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
# The agent is always implemented as a Python function that accepts two arguments: obs and config

def agent(obs, config):

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
from kaggle_environments import make, evaluate



# Create the game environment

env = make("connectx")



# Two random agents play one game round

env.run([agent, "random"])



# Show the game

env.render(mode="ipython")


def get_win_percentages(agent1, agent2, n_rounds=100):

    # Use default Connect Four setup

    config = {'rows': 6, 'columns': 7, 'inarow': 4}

    # Agent 1 goes first (roughly) half the time          

    outcomes = evaluate("connectx", [agent1, agent2], config, [], n_rounds//2)

    # Agent 2 goes first (roughly) half the time      

    outcomes += [[b,a] for [a,b] in evaluate("connectx", [agent2, agent1], config, [], n_rounds-n_rounds//2)]

    print("Agent 1 Win Percentage:", np.round(outcomes.count([1,-1])/len(outcomes), 2))

    print("Agent 2 Win Percentage:", np.round(outcomes.count([-1,1])/len(outcomes), 2))

    print("Number of Invalid Plays by Agent 1:", outcomes.count([None, 0]))

    print("Number of Invalid Plays by Agent 2:", outcomes.count([0, None]))
get_win_percentages(agent1=agent, agent2="random")