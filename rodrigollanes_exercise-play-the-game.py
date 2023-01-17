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

    oponent = 2 if(obs.mark == 1) else 1

    for i in valid_moves:

        if(check_winning_move(obs, config, i, obs.mark)):

            return i

    return random.choice(valid_moves)

    

# Check your answer

q_1.check()
# Lines below will give you a hint or solution code

#q_1.hint()

#q_1.solution()
def agent_q2(obs, config):

    valid_moves = [col for col in range(config.columns) if obs.board[col] == 0]

    # Your code here: Amend the agent!

    oponent = 2 if(obs.mark == 1) else 1

    for i in valid_moves:

        if(check_winning_move(obs, config, i, obs.mark)):

            return i

    for i in valid_moves:

        if(check_winning_move(obs, config, i, oponent)):

            return i

    return random.choice(valid_moves)

# Check your answer

q_2.check()
# Lines below will give you a hint or solution code

#q_2.hint()

#q_2.solution()
#q_3.hint()
# Check your answer (Run this code cell to receive credit!)

q_3.solution()
def check_move(obs, config, col, you, piece):

    # Convert the board to a 2D grid

    grid = np.asarray(obs.board).reshape(config.rows, config.columns)

    next_grid = drop_piece(grid, col, you, config)

    next_grid = drop_piece(next_grid, col, piece, config)

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
def my_agent(obs, config):

    rows = [obs.board[i:i+config.columns] for i in range(0, config.rows*config.columns-1, config.columns)]

    cols = [[row[i] for row in rows] for i in range(len(rows[0]))]

    valid_moves = [col for col in range(config.columns) if obs.board[col] == 0]

    oponent = 1

    if(obs.mark == 1):

        oponent = 2

    

    for i in valid_moves:

        if(check_winning_move(obs, config, i, obs.mark)):

            return i

    for i in valid_moves:

        if(check_winning_move(obs, config, i, oponent)):

            return i

    

    aux = []

    for i in valid_moves:

        if(not check_move(obs, config, i, obs.mark, oponent)):

            aux.append(i)

    valid_moves = aux

    

    for i in valid_moves:

        for j in cols[i]:

            if(j != 0):

                if(j == obs.mark):

                    return i

                else:

                    break

    for i in valid_moves:

        h = 0

        for j in cols[i]:

            if(j == 0):

                h += 1

            else:

                break

        if(h >= config.inarow):

            return i

        

    return random.choice(valid_moves)
# Two random agents play one game round

env.run([my_agent, "random"])



# Show the game

env.render(mode="ipython")
# Run this code cell to get credit for creating an agent

q_4.check()
from kaggle_environments import evaluate, make, utils



env = make("connectx", debug=True)

env.play([my_agent, None], width=500, height=450)
import inspect

import os



def write_agent_to_file(function, file):

    with open(file, "a" if os.path.exists(file) else "w") as f:

        f.write(inspect.getsource(function))

        print(function, "written to", file)



if(os.path.exists("submission.py")):

    os.remove('submission.py') 

    

write_agent_to_file(drop_piece, "submission.py")

write_agent_to_file(check_move, "submission.py")

write_agent_to_file(check_winning_move, "submission.py")

write_agent_to_file(my_agent, "submission.py")



# Check that submission file was created

q_5.check()
# Note: Stdout replacement is a temporary workaround.

import sys

out = sys.stdout

submission = utils.read_file("/kaggle/working/submission.py")

agent = utils.get_last_callable(submission)

sys.stdout = out



env = make("connectx", debug=True)

env.run([agent, agent])

print("Success!" if env.state[0].status == env.state[1].status == "DONE" else "Failed...")