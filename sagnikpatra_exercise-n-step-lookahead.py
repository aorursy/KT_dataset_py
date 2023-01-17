from learntools.core import binder

binder.bind(globals())

from learntools.game_ai.ex3 import *
#q_1.hint()
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

#q_3.hint()

#q_3.solution()
#q_4.hint()
# Check your answer (Run this code cell to receive credit!)

q_4.solution()
def my_agent(obs, config):

    # Your code here: Amend the agent!

    valid_moves = [col for col in range(config.columns) if obs.board[col] == 0]

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