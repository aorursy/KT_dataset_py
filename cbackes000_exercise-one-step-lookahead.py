from learntools.core import binder

binder.bind(globals())

from learntools.game_ai.ex2 import *
# TODO: Assign your values here

A = 10000

B = 100

C = 1

D = -100

E = -10000



# Check your answer (this will take a few seconds to run!)

q_1.check()
# Lines below will give you a hint or solution code

#q_1.hint()

#q_1.solution()
#q_2.hint()
# Check your answer (Run this code cell to receive credit!)

# The agent from the tutorial will choose randomly.

q_2.solution()
def my_agent(obs, config):

    # Your code here: Amend the agent!

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
# Run this code cell to get credit for creating an agent

q_3.check()
import inspect

import os



def write_agent_to_file(function, file):

    with open(file, "a" if os.path.exists(file) else "w") as f:

        f.write(inspect.getsource(function))

        print(function, "written to", file)



write_agent_to_file(my_agent, "submission.py")