from kaggle_environments import make, evaluate



# create game environment

# set debug =True to see error

env = make('connectx', debug=True)



# list of available default agents

print(list(env.agents))

# two random agents play one game round

env.run(['random','random'])



# show the game

env.render(mode="ipython")
import random

import numpy as np
# first agent

# selects random valid column

def agent_random(obs, config):

    valid_moves = [ col for col in range(config.columns) if obs.board[col]==0 ]

    return random.choice(valid_moves)



# second agent

# select middle column

def agent_middle(obs, config):

    return config.columns//2



# third agent

# select leftmost valid column

def agent_leftmost(obs, config):

    valid_moves = [ col for col in range(config.columns) if obs.board[col]==0 ]

    return valid_moves[0]

    
# agents play one game round

env.run([agent_leftmost, agent_random])



env.render(mode="ipython")
# To learn more about the evaluate() function, check out the documentation here: (insert link here)

def get_win_percentages(agent1, agent2, n_rounds=100):

    # Use default Connect Four setup

    config = {'rows': 6, 'columns': 7, 'inarow': 4}

    # Agent 1 goes first (roughly) half the time          

    outcomes = evaluate("connectx", [agent1, agent2], config, [], n_rounds//2)

    # Agent 2 goes first (roughly) half the time      

    outcomes += [[b,a] for [a,b] in evaluate("connectx", [agent2, agent1], config, [], n_rounds-n_rounds//2)]

    print("Agent 1 Win Percentage:", np.round(outcomes.count([1,0])/len(outcomes), 2))

    print("Agent 2 Win Percentage:", np.round(outcomes.count([0,1])/len(outcomes), 2))

    print("Number of Invalid Plays by Agent 1:", outcomes.count([None, 0.5]))

    print("Number of Invalid Plays by Agent 2:", outcomes.count([0.5, None]))

    print("Number of Draws (in {} game rounds):".format(n_rounds), outcomes.count([0.5, 0.5]))
get_win_percentages(agent1=agent_middle, agent2=agent_random)
get_win_percentages(agent1=agent_leftmost, agent2=agent_random)