from learntools.core import binder

binder.bind(globals())

from learntools.game_ai.ex3 import *
q_1.hint()
# Check your answer (Run this code cell to receive credit!)

q_1.solution()
# Fill in the blank

num_leaves = 7*7*7



# Check your answer

q_2.check()
# Lines below will give you a hint or solution code

q_2.hint()

q_2.solution()
# Fill in the blank

selected_move = 3



# Check your answer

q_3.check()
# Lines below will give you a hint or solution code

q_3.hint()

q_3.solution()
#q_4.hint()
# Check your answer (Run this code cell to receive credit!)

q_4.solution()
def my_agent(obs, config):

    # Your code here: Amend the agent!

    

    import random

    import numpy as np

    

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

    

     # Helper function for score_move: gets board at next step if agent drops piece in selected column

    def drop_piece(grid, col, mark, config):

        next_grid = grid.copy()

        for row in range(config.rows-1, -1, -1):

            if next_grid[row][col] == 0:

                break

        next_grid[row][col] = mark

        return next_grid

    

    # Helper function for minimax: calculates value of heuristic for grid

    #def count_windows ?

    def get_heuristic(grid, mark, config):

        num_threes = count_windows(grid, 3, mark, config)

        num_fours = count_windows(grid, 4, mark, config)

        num_threes_opp = count_windows(grid, 3, mark%2+1, config)

        num_fours_opp = count_windows(grid, 4, mark%2+1, config)

        score = num_threes - 1e2*num_threes_opp - 1e4*num_fours_opp + 1e6*num_fours

        return score

    

    # Helper function for minimax: checks if agent or opponent has four in a row in the window

    def is_terminal_window(window, config):

        return window.count(1) == config.inarow or window.count(2) == config.inarow



    # Helper function for minimax: checks if game has ended

    #def is_terminal_window

    def is_terminal_node(grid, config):

        # Check for draw 

        if list(grid[0, :]).count(0) == 0:

            return True

        # Check for win: horizontal, vertical, or diagonal

        # horizontal 

        for row in range(config.rows):

            for col in range(config.columns-(config.inarow-1)):

                window = list(grid[row, col:col+config.inarow])

                if is_terminal_window(window, config):

                    return True

        # vertical

        for row in range(config.rows-(config.inarow-1)):

            for col in range(config.columns):

                window = list(grid[row:row+config.inarow, col])

                if is_terminal_window(window, config):

                    return True

        # positive diagonal

        for row in range(config.rows-(config.inarow-1)):

            for col in range(config.columns-(config.inarow-1)):

                window = list(grid[range(row, row+config.inarow), range(col, col+config.inarow)])

                if is_terminal_window(window, config):

                    return True

        # negative diagonal

        for row in range(config.inarow-1, config.rows):

            for col in range(config.columns-(config.inarow-1)):

                window = list(grid[range(row, row-config.inarow, -1), range(col, col+config.inarow)])

                if is_terminal_window(window, config):

                    return True

        return False

    

    

    # Minimax implementation

    def minimax(node, depth, maximizingPlayer, mark, config):

        #def is_terminal_node ?

        is_terminal = is_terminal_node(node, config)

        valid_moves = [c for c in range(config.columns) if node[0][c] == 0]

        if depth == 0 or is_terminal:

            #def get_heuristic ?

            return get_heuristic(node, mark, config)

        if maximizingPlayer:

            value = -np.Inf

            for col in valid_moves:

                child = drop_piece(node, col, mark, config)

                value = max(value, minimax(child, depth-1, False, mark, config))

            return value

        else:

            value = np.Inf

            for col in valid_moves:

                child = drop_piece(node, col, mark%2+1, config)

                value = min(value, minimax(child, depth-1, True, mark, config))

            return value

    

    # Uses minimax to calculate value of dropping piece in selected column

    def score_move(grid, col, mark, config, nsteps):

        #def drop_piece ?

        next_grid = drop_piece(grid, col, mark, config)

        #def minimax ?

        score = minimax(next_grid, nsteps-1, False, mark, config)

        return score

    

    N_STEPS=2

    

    # Get list of valid moves

    valid_moves = [c for c in range(config.columns) if obs.board[c] == 0]

    # Convert the board to a 2D grid

    grid = np.asarray(obs.board).reshape(config.rows, config.columns)

    # Use the heuristic to assign a score to each possible board in the next step

    # def score_move ?

    scores = dict(zip(valid_moves, [score_move(grid, col, obs.mark, config, N_STEPS) for col in valid_moves]))

    # Get a list of columns (moves) that maximize the heuristic

    max_cols = [key for key in scores.keys() if scores[key] == max(scores.values())]

    

    return random.choice(max_cols)
# Run this code cell to get credit for creating an agent

q_5.check()
from kaggle_environments import evaluate, make, utils

env = make("connectx", debug=True)



# Agents play one game round

env.run([my_agent,"negamax"])



# Show the game

env.render(mode="ipython")
# Link: https://www.kaggle.com/alexisbcook/play-the-game

def get_win_percentages(agent1, agent2, n_rounds=10):

    import numpy as np

    # Use default Connect Four setup

    config = {'rows': 6, 'columns': 7, 'inarow': 4}

    # Agent 1 goes first (roughly) half the time          

    outcomes = evaluate("connectx", [agent1, agent2], config, [], n_rounds//2)

    # Agent 2 goes first (roughly) half the time      

    outcomes += [[b,a] for [a,b] in evaluate("connectx", [agent2, agent1], config, [], n_rounds-n_rounds//2)]

    

    #these codes are not work because of the change in kaggle environment

    #you can fix this by downgrading the kaggle_environments

    #!pip install kaggle_environments==0.1.6 use this before any other code.

    

    #print("Agent 1 Win Percentage:", np.round(outcomes.count([1,0])/len(outcomes), 2))

    #print("Agent 2 Win Percentage:", np.round(outcomes.count([0,1])/len(outcomes), 2))

    #print("Number of Invalid Plays by Agent 1:", outcomes.count([None, 0.5]))

    #print("Number of Invalid Plays by Agent 2:", outcomes.count([0.5, None]))

    #print("Number of Draws (in {} game rounds):".format(n_rounds), outcomes.count([0.5, 0.5]))

    

    print("Agent 1 Win Percentage:", np.round(outcomes.count([1,-1])/len(outcomes), 2))

    print("Agent 2 Win Percentage:", np.round(outcomes.count([-1,1])/len(outcomes), 2))

    print("Number of Invalid Plays by Agent 1:", outcomes.count([None, 0]))

    print("Number of Invalid Plays by Agent 2:", outcomes.count([0, None]))

    print("Number of Draws (in {} game rounds):".format(n_rounds), outcomes.count([0, 0]))
#get_win_percentages(agent1=my_agent, agent2=my_agent)
import inspect

import os



def write_agent_to_file(function, file):

    with open(file, "a" if os.path.exists(file) else "w") as f:

        f.write(inspect.getsource(function))

        print(function, "written to", file)



write_agent_to_file(my_agent, "submission.py")