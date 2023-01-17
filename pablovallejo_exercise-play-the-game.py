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

    for move in valid_moves:

        if(check_winning_move(obs, config, move, piece=obs.mark) == True):

            return move

    

    return random.choice(valid_moves)

    

# Check your answer

q_1.check()
# Lines below will give you a hint or solution code

q_1.hint()

q_1.solution()
def agent_q2(obs, config):

    valid_moves = [col for col in range(config.columns) if obs.board[col] == 0]

    piece = obs.mark

    for move in valid_moves:

        if(check_winning_move(obs, config, move, piece) == True):

            return move

        if piece == 1:

            piece = 2

        elif piece == 2:

            piece = 1

        if(check_winning_move(obs, config, move, piece) == True):

            return move

    return random.choice(valid_moves) 



# Check your answer

q_2.check()
# Lines below will give you a hint or solution code

q_2.hint()

q_2.solution()
q_3.hint()
# Check your answer (Run this code cell to receive credit!)

q_3.solution()
def my_agent(obs, config):

    valid_moves = [col for col in range(config.columns) if obs.board[col] == 0]

    piece = obs.mark

    probabilidad = 0

    probabilidad1 = 0

    probabilidad2 = 0

    probabilidad3 = 0

    probabilidad4 = 0

    probabilidad5 = 0

    mov_correcto = [0,0,0,0,0,0,0]

    maximo = 0

    pos = 0

    b = 0

    for a in range(42):

        if obs.board[a] != 0:

            b = b+1            

    if b == 0:

        return random.choice(valid_moves)

    for move in valid_moves:

        probabilidad = 0

        if(check_winning_move(obs, config, move, piece) == True):

            return move

        if(check_winning_move(obs, config, move, obs.mark%2+1) == True):

            return move

        if obs.board[move] == obs.board[move+1]:

            probabilidad1 = probabilidad1 +1

            if obs.board[move] == obs.board[move+2]:

                probabilidad1 = probabilidad1 +1

                if obs.board[move] == obs.board[move+3]:

                    probabilidad1 = probabilidad1 +1

        if obs.board[move] == obs.board[move-1]:

            probabilidad2 = probabilidad2 +1

            if obs.board[move] == obs.board[move-2]:

                probabilidad2 = probabilidad2 +1

                if obs.board[move] == obs.board[move-3]:

                    probabilidad2 = probabilidad2 +1

        if obs.board[move] == obs.board[move+7]:

            probabilidad3 = probabilidad3 +1

            if obs.board[move] == obs.board[move+14]:

                probabilidad3 = probabilidad3 +1

                if obs.board[move] == obs.board[move+21]:

                    probabilidad3 = probabilidad3 +1

        if obs.board[move] == obs.board[move+8]:

            probabilidad4 = probabilidad4 +1

            if obs.board[move] == obs.board[move+16]:

                probabilidad4 = probabilidad4 +1

                if obs.board[move] == obs.board[move+24]:

                    probabilidad4 = probabilidad4 +1

        if obs.board[move] == obs.board[move+6]:

            probabilidad5 = probabilidad5 +1

            if obs.board[move] == obs.board[move+12]:

                probabilidad5 = probabilidad5 +1

                if obs.board[move] == obs.board[move+18]:

                    probabilidad5 = probabilidad5 +1

        if probabilidad1 >= probabilidad:

            probabilidad= probabilidad1

        if probabilidad2 >= probabilidad:

            probabilidad= probabilidad2

        if probabilidad3 >= probabilidad:

            probabilidad= probabilidad3

        if probabilidad4 >= probabilidad:

            probabilidad= probabilidad4

        if probabilidad5 >= probabilidad:

            probabilidad= probabilidad5 

           

        mov_correcto[move] = probabilidad

    for move in valid_moves:

        if mov_correcto[move] >= maximo:

            maximo = mov_correcto[move]

            pos = move

    return pos
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



write_agent_to_file(my_agent, "submission.py")



# Check that submission file was created

q_5.check()