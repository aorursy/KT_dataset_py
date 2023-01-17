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
# Uses minimax to calculate value of dropping piece in selected column + pruning

def score_move_prune(grid, col, mark, config, nsteps):

    next_grid = drop_piece(grid, col, mark, config)

    score = minimax_prune(next_grid, nsteps-1, False, mark, config)

    return score



# Minimax implementation + pruning

def minimax_prune(node, depth, maximizingPlayer, mark, config,a ,b ):

    is_terminal = is_terminal_node(node, config)

    valid_moves = [c for c in range(config.columns) if node[0][c] == 0]

    if depth == 0 or is_terminal:

        return get_heuristic(node, mark, config)

    if maximizingPlayer:

        value = -np.Inf

        for col in valid_moves:

            child = drop_piece(node, col, mark, config)

            value = max(value, minimax(child, depth-1, False, mark, config, a, b))

            a = max(a, value)

            #prune

            if a >= b : 

                break

        return value

    else:

        value = np.Inf

        for col in valid_moves:

            child = drop_piece(node, col, mark%2+1, config)

            value = min(value, minimax(child, depth-1, True, mark, config, a, b))

            b = min (b, value)

            #prune

            if a >= b : 

                break

        return value



def my_agent(obs, config):

    import random

    # Your code here: Amend the agent!

    valid_moves = [col for col in range(config.columns) if obs.board[col] == 0]

    # Convert the board to a 2D grid

    grid = np.asarray(obs.board).reshape(config.rows, config.columns)

    # Use the heuristic to assign a score to each possible board in the next step

    scores = dict(zip(valid_moves, [score_move_prune(grid, col, obs.mark, config, N_STEPS) for col in valid_moves]))

    # Get a list of columns (moves) that maximize the heuristic

    max_cols = [key for key in scores.keys() if scores[key] == max(scores.values())]

    # Select at random from the maximizing columns

    return random.choice(valid_moves)
# Run this code cell to get credit for creating an agent

q_5.check()
import inspect

import os



def write_agent_to_file(function, file):

    with open(file, "a" if os.path.exists(file) else "w") as f:

        f.write(inspect.getsource(function))

        print(function, "written to", file)



write_agent_to_file(my_agent, "submission.py")