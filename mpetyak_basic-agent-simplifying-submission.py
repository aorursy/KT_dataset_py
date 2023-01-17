# 1. Enable Internet in the Kernel (Settings side pane)



# 2. Curl cache may need purged if v0.1.4 cannot be found (uncomment if needed). 

# !curl -X PURGE https://pypi.org/simple/kaggle-environments



# ConnectX environment was defined in v0.1.4

!pip install 'kaggle-environments>=0.1.4'
from kaggle_environments import evaluate, make



env = make("connectx", debug=True)

env.render()
def is_winning_board(board, configuration):

    """

    Check if a given board has 4 connected

    """

    move_deltas = [(1, 0), # horizontal

                   (0, 1), # vertical

                   (1, 1), # diagonal top-right

                   (-1, 1) # diagonal bottom-left

                  ]

    

    rows, columns = configuration.rows, configuration.columns

    for r in range(rows):

        for c in range(columns):

            # ignore empty cells

            if board[r][c] == 0:

                continue

                

            for dr, dc in move_deltas:

                # check we don't leave the board

                if not (0 < r + dr*3 < rows and 0 < c + dc*3 < columns):

                    continue

                # finally check for 4 in a row

                if board[r][c] == board[r+dr][c+dc] == board[r+dr*2][c+dc*2] == board[r+dr*3][c+dc*3]:

                    return True

        

    return False
def make_move(board, column, marker, configuration):

    """

    Returns a new board with a chip dropped at provided column

    """

    import copy

    

    board = copy.deepcopy(board) # avoid modifying the original board

    row = 0

    max_row = configuration.rows

    

    # find lowest unfilled

    while row < max_row and board[row][column] == 0:

        row += 1

    

    board[row - 1][column] = marker

    

    return board
# This agent random chooses a non-empty column.

def my_agent(observation, configuration):

    from random import choice

    import numpy

    

    rows, columns = configuration.rows, configuration.columns

    allowable_moves = [c for c in range(columns) if observation.board[c] == 0]

    

    # transform into 2D board

    board = numpy.array(observation.board)

    board.resize(rows, columns)

    

    for move in allowable_moves:

        for marker in ["1", "0"]: # first try our moves then opponent's

            new_board = make_move(board, move, marker, configuration)

            if is_winning_board(new_board, configuration):

                return move

        

    return choice(allowable_moves)
env.reset()

# Play as the first agent against default "random" agent.

env.run([my_agent, "random"])

env.render(mode="ipython", width=500, height=450)
# Play as first position against random agent.

trainer = env.train([None, "random"])



observation = trainer.reset()



while not env.done:

    my_action = my_agent(observation, env.configuration)

    print("My Action", my_action)

    observation, reward, done, info = trainer.step(my_action)

    # env.render(mode="ipython", width=100, height=90, header=False, controls=False)

env.render()
def mean_reward(rewards):

    return sum(r[0] for r in rewards) / sum(r[0] + r[1] for r in rewards)



# Run multiple episodes to estimate it's performance.

print("My Agent vs Random Agent:", mean_reward(evaluate("connectx", [my_agent, "random"], num_episodes=10)))

print("My Agent vs Negamax Agent:", mean_reward(evaluate("connectx", [my_agent, "negamax"], num_episodes=10)))
import inspect

import os



# can get function reference through 'globals()[func_name]'

import_functions = [

    is_winning_board,

    make_move

]



def write_agent_to_file(function, file, import_functions=[]):

    # get source and transform into list of lines

    function_source = inspect.getsource(my_agent)

    function_source = function_source.split("\n")



    for func in import_functions:

        import_source = inspect.getsource(func)

        # add tab after every new line

        import_source = import_source.split("\n")

        import_source = ["    " + line for line in import_source]

        # insert new function after function definition

        function_source.insert(1, "\n".join(import_source))

    

    function_source = "\n".join(function_source)

    

    with open(file, "a" if os.path.exists(file) else "w") as f:

        f.write(function_source)

        print(function_source) # print written code



write_agent_to_file(my_agent, "submission.py", import_functions=import_functions)