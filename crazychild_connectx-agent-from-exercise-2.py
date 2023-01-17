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

    

    class MyBoard(object):

        def __init__(self, board, mark):

            self.board = board

            self.mark = mark

        

    valid_moves = [col for col in range(config.columns) if obs.board[col] == 0]

    

    for move in valid_moves:

        if check_winning_move(obs, config, move, obs.mark):

            return move

        

    for move1 in valid_moves:

        

        grid = np.asarray(obs.board).reshape(config.rows, config.columns)

        board = list(drop_piece(grid, move1, obs.mark, config).flatten())

        mark = obs.mark

        next_obs = MyBoard(board, mark)

        for move2 in range(config.columns):

            if next_obs.board[move2]==0 and check_winning_move(next_obs, config, move2, obs.mark%2+1):

                if move1 == move2:

                    valid_moves.remove(move1)

                else:

                    return move2

                

    return random.choice(valid_moves) if len(valid_moves)>0 else 0
env.run([my_agent, 'random'])
env.render(mode='ipython', width=500, height=500)
import numpy as np



def get_win_percentages(agent1, agent2, n_rounds=200):

    # Use default Connect Four setup

    config = {'rows': 6, 'columns': 7, 'inarow': 4}

    # Agent 1 goes first (roughly) half the time          

    outcomes = evaluate("connectx", [agent1, agent2], config, [], n_rounds//2)

    # Agent 2 goes first (roughly) half the time      

    outcomes += [[b,a] for [a,b] in evaluate("connectx", [agent2, agent1], config, [], n_rounds-n_rounds//2)]

#     print(outcomes)

    print("Agent 1 Win Percentage:", np.round(outcomes.count([1,0])/len(outcomes), 2))

    print("Agent 2 Win Percentage:", np.round(outcomes.count([0,1])/len(outcomes), 2))

    print("Number of Invalid Plays by Agent 1:", outcomes.count([None, 0]))

    print("Number of Invalid Plays by Agent 2:", outcomes.count([0, None]))
get_win_percentages(agent1=my_agent, agent2='random')
import inspect

import os



def write_agent_to_file(function, file):

    with open(file, "w") as f:

        f.write(inspect.getsource(function))

        print(function, "written to", file)

file_name = "submission_ver_1"

write_agent_to_file(my_agent, f"{file_name}.py")
import sys

from kaggle_environments import utils



out = sys.stdout

submission = utils.read_file(f"/kaggle/working/{file_name}.py")

agent = utils.get_last_callable(submission)

sys.stdout = out



env = make("connectx", debug=True)

env.run([agent, agent])

print("Success!" if env.state[0].status == env.state[1].status == "DONE" else "Failed...")