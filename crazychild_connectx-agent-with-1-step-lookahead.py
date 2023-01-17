from kaggle_environments import make, evaluate

# Create the game environment
# Set debug=True to see the errors if your agent refuses to run
env = make("connectx", debug=True)
# The agent is always implemented as a Python function that accepts two arguments: obs and config
def my_agent(obs, config):
    
    ################################
    # Imports and helper functions #
    ################################
    
    import numpy as np
    import random


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
        A = 1e6
        B = 1e2
        C = 1
        D = -1e1
        E = -1e4
        num_twos = count_windows(grid, 2, mark, config)
        num_threes = count_windows(grid, 3, mark, config)
        num_fours = count_windows(grid, 4, mark, config)
        num_twos_opp = count_windows(grid, 2, mark%2+1, config)
        num_threes_opp = count_windows(grid, 3, mark%2+1, config)
        score = A*num_fours + B*num_threes + C*num_twos + D*num_twos_opp + E*num_threes_opp
        return score

    # Helper function for get_heuristic: checks if window satisfies heuristic conditions
    def check_window(window, num_discs, piece, config):
        return (window.count(piece) == num_discs and window.count(0) == config.inarow-num_discs)

    # Helper function for get_heuristic: counts number of windows satisfying specified heuristic conditions
    def count_windows(grid, num_discs, piece, config):
        num_windows = 0
        row_limit = config.rows-(config.inarow-1)
        col_limit = config.columns-(config.inarow-1)

        for row in range(config.rows):
            for col in range(config.columns):
                end_row = row+config.inarow
                end_col = col+config.inarow

                # horizontal
                if end_col < col_limit:
                    window = list(grid[row, col:end_col])
                    if check_window(window, num_discs, piece, config):
                        num_windows += 1
                # vertical
                if end_row < row_limit:
                    window = list(grid[row:end_row, col])
                    if check_window(window, num_discs, piece, config):
                        num_windows += 1

                # positive diagonal
                if (end_row < row_limit) and (end_col < col_limit):
                    window = list(grid[range(row, end_row), range(col, end_col)])
                    if check_window(window, num_discs, piece, config):
                        num_windows += 1

                # negative diagonal
                if (row >= (config.inarow-1)) and (end_col < col_limit):
                    window = list(grid[range(row, row-config.inarow, -1), range(col, end_col)])
                    if check_window(window, num_discs, piece, config):
                        num_windows += 1

        return num_windows
    
    #########################
    # Agent makes selection #
    #########################
    
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
env.run([my_agent, 'random'])
env.render(mode='ipython', width=500, height=500)
import numpy as np

def get_win_percentages(agent1, agent2, n_rounds=10):
    # Use default Connect Four setup
    config = {'rows': 6, 'columns': 7, 'inarow': 4}
    # Agent 1 goes first (roughly) half the time          
    outcomes = evaluate("connectx", [agent1, agent2], config, [], n_rounds//2)
    # Agent 2 goes first (roughly) half the time      
    outcomes += [[b,a] for [a,b] in evaluate("connectx", [agent2, agent1], config, [], n_rounds-n_rounds//2)]
    print(outcomes)
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
file_name = "submission_ver_1_look_ahead"
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