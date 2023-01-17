# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



!pip install kaggle_environments==0.1.6



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

#for dirname, _, filenames in os.walk('/kaggle/input'):

#    for filename in filenames:

#        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
from kaggle_environments import make, evaluate



# Create the game environment

# Set debug=True to see the errors if your agent refuses to run

env = make("connectx", debug=True)



# List of available default agents

print(list(env.agents))
# Two random agents play one game round

env.run(["random", "random"])



# Show the game

env.render(mode="ipython")
# Selects random valid column

def agent_random(obs, config):

    #an agent function should be fully encapsulated (no external dependencies)

    #then we need to import all librabries inside out agent

    import random

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

env.run([agent_random,agent_random])



# Show the game

env.render(mode="ipython")
# Link: https://www.kaggle.com/alexisbcook/play-the-game

def get_win_percentages(agent1, agent2, n_rounds=100):

    # Use default Connect Four setup

    config = {'rows': 6, 'columns': 7, 'inarow': 4}

    # Agent 1 goes first (roughly) half the time          

    outcomes = evaluate("connectx", [agent1, agent2], config, [], n_rounds//2)

    # Agent 2 goes first (roughly) half the time      

    outcomes += [[b,a] for [a,b] in evaluate("connectx", [agent2, agent1], config, [], n_rounds-n_rounds//2)]

    

    #these codes are not work because of the change in kaggle environment

    #you can fix this by downgrading the kaggle_environments

    #!pip install kaggle_environments==0.1.6 use this before any other code.

    

    print("Agent 1 Win Percentage:", np.round(outcomes.count([1,0])/len(outcomes), 2))

    print("Agent 2 Win Percentage:", np.round(outcomes.count([0,1])/len(outcomes), 2))

    print("Number of Invalid Plays by Agent 1:", outcomes.count([None, 0.5]))

    print("Number of Invalid Plays by Agent 2:", outcomes.count([0.5, None]))

    print("Number of Draws (in {} game rounds):".format(n_rounds), outcomes.count([0.5, 0.5]))

    

    #print("Agent 1 Win Percentage:", np.round(outcomes.count([1,-1])/len(outcomes), 2))

    #print("Agent 2 Win Percentage:", np.round(outcomes.count([-1,1])/len(outcomes), 2))

    #print("Number of Invalid Plays by Agent 1:", outcomes.count([None, 0]))

    #print("Number of Invalid Plays by Agent 2:", outcomes.count([0, None]))

    #print("Number of Draws (in {} game rounds):".format(n_rounds), outcomes.count([0, 0]))

get_win_percentages(agent1=agent_random, agent2=agent_random)