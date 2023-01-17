from kaggle_environments import make, evaluate



# Create the game environment

# Set debug=True to see the errors if your agent refuses to run

env = make("connectx", debug=True)
def my_agent(obs, config):

    

    ################################

    # Imports and helper functions #

    ################################

    

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

    

    #########################

    # Agent makes selection #

    #########################

    

    valid_moves = [col for col in range(config.columns) if obs.board[col] == 0]

    for col in valid_moves:

        if check_winning_move(obs, config, col, obs.mark):

            return col

    return random.choice(valid_moves)
import inspect

import os



def write_agent_to_file(function, file):

    with open(file, "a" if os.path.exists(file) else "w") as f:

        f.write(inspect.getsource(function))

        print(function, "written to", file)



write_agent_to_file(my_agent, "submission.py")
import sys

from kaggle_environments import utils



out = sys.stdout

submission = utils.read_file("/kaggle/working/submission.py")

agent = utils.get_last_callable(submission)

sys.stdout = out



env = make("connectx", debug=True)

env.run([agent, agent])

print("Success!" if env.state[0].status == env.state[1].status == "DONE" else "Failed...")