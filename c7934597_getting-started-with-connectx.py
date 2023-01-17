# ConnectX environment was defined in v0.1.6

!pip install 'kaggle-environments>=0.1.6'
from kaggle_environments import evaluate, make, utils



env = make("connectx", debug=True)

env.render()
def my_agent(obs, config):

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

    

    valid_moves = [col for col in range(config.columns) if obs.board[col] == 0]

    for col in valid_moves:

        if check_winning_move(obs, config, col, obs.mark):

            return col

    for col in valid_moves:

        if check_winning_move(obs, config, col, obs.mark%2+1):

            return col

    return random.choice(valid_moves)



# This agent random chooses a non-empty column.

#def my_agent(observation, configuration):

    #from random import choice

    #return choice([c for c in range(configuration.columns) if observation.board[c] == 0])
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

    return sum(r[0] for r in rewards) / float(len(rewards))



# Run multiple episodes to estimate its performance.

print("My Agent vs Random Agent:", mean_reward(evaluate("connectx", [my_agent, "random"], num_episodes=10)))

print("My Agent vs Negamax Agent:", mean_reward(evaluate("connectx", [my_agent, "negamax"], num_episodes=10)))
# "None" represents which agent you'll manually play as (first or second player).

env.play([None, "negamax"], width=500, height=450)
import inspect

import os



def write_agent_to_file(function, file):

    with open(file, "a" if os.path.exists(file) else "w") as f:

        f.write(inspect.getsource(function))

        print(function, "written to", file)



write_agent_to_file(my_agent, "submission.py")
# Note: Stdout replacement is a temporary workaround.

import sys

out = sys.stdout

submission = utils.read_file("/kaggle/working/submission.py")

agent = utils.get_last_callable(submission)

sys.stdout = out



env = make("connectx", debug=True)

env.run([agent, agent])

print("Success!" if env.state[0].status == env.state[1].status == "DONE" else "Failed...")