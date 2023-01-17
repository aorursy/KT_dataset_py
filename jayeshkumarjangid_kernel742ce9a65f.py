from kaggle_environments import make, evaluate



# Create the game environment

# Set debug=True to see the errors if your agent refuses to run

env = make("connectx", debug=True)



# List of available default agents

print(list(env.agents))

['random', 'negamax']

# Two random agents play one game round

env.run(["random", "random"])



# Show the game

env.render(mode="ipython")

import random

import numpy as np

# Selects random valid column

def agent_random(obs, config):

    valid_moves = [col for col in range(config.columns) if obs.board[col] == 0]

    return random.choice(valid_moves)



# Selects middle column

def agent_middle(obs, config):

    return config.columns//2



# Selects leftmost valid column

def agent_leftmost(obs, config):

    valid_moves = [col for col in range(config.columns) if obs.board[col] == 0]

    return valid_moves[0]

# Agents play one game round

env.run([agent_leftmost, agent_random])








