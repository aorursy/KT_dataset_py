!pip install kaggle-environments>=0.1.6
from kaggle_environments import make, evaluate, utils, agent

import random

import numpy as np

import os

import inspect
env=make("connectx",debug=True)

env.render()
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
def my_agent(obs, config):

    opponent_piece = 1 if obs.mark == 2 else 2

    choice = []

    for col in range(config.columns):

        if check_winning_move(obs,config,col,obs.mark):

            return col

        elif check_winning_move(obs,config,col,opponent_piece):

            choice.append(col)

    if len(choice):

        return random.choice(choice)

    valid_moves = [col for col in range(config.columns) if obs.board[col] == 0]

    return random.choice(valid_moves)
env.reset()

env.run([my_agent,"random"])

env.render(mode="ipython",width=500,height=500)
env.reset()

env.run(["random",my_agent])

env.render(mode="ipython",width=500,height=500)
env.reset()

env.run([my_agent,"negamax"])

env.render(mode="ipython",width=500,height=500)
env.reset()

env.run(["negamax",my_agent])

env.render(mode="ipython",width=500,height=500)
def win_percentage(player,opponent,num_episodes=10):

    episodes = num_episodes//2

    outcomes = evaluate("connectx",[player,opponent],num_episodes=episodes)

    outcomes += [[b,a] for [a,b] in evaluate("connectx",[opponent,player],num_episodes=num_episodes-episodes)]

    wins = outcomes.count([1,-1])

    losses = outcomes.count([-1,1])

    return (np.sum(wins) / len(outcomes))*100
random_mean_reward = win_percentage(my_agent,"random",num_episodes=10)

negamax_mean_reward = win_percentage(my_agent,"negamax",num_episodes=10)
print("My Agent V/S Random Agent ", random_mean_reward,"%")

print("My Agent V/S Negamax Agent ",negamax_mean_reward,"%")
env.play([my_agent,None],width=500,height=500)
submission_file="submission.py"
if os.path.exists(submission_file):

    os.remove(submission_file)
def write_agent_dependencies(file,dependencies):

    with open(file,"a" if os.path.exists(file) else "w") as f:

        for dependency in dependencies:

            f.write(f"import {dependency}\n")

        print(f"depedencies written to {file}")

    

def write_function_to_file(file,function):

    with open(file, "a" if os.path.exists(file) else "w") as f:

        f.write("")

        f.write(inspect.getsource(function))

        print(f"function written to {file}")
dependencies=["numpy as np","random"]
write_agent_dependencies("submission.py",dependencies)

write_function_to_file("submission.py",drop_piece)

write_function_to_file("submission.py",check_winning_move)

write_function_to_file("submission.py",my_agent)
with open("submission.py","r") as f:

    print(f.read())
import sys

out = sys.stdout

submission = utils.read_file("/kaggle/working/submission.py")

sys.stdout = out

saved_agent = agent.get_last_callable(submission)

env = make("connectx", debug=True)

env.run([saved_agent, saved_agent])

print("Success!" if env.state[0].status == env.state[1].status == "DONE" else "Failed...")