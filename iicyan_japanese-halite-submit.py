!pip install 'kaggle-environments>=0.2.1'
%%writefile agent.py
from kaggle_environments.envs.halite.helpers import *
from random import choice

from random import choice
def agent(obs):
    action = {}
    ship_id = list(obs.players[obs.player][2].keys())[0]
    ship_action = choice(["NORTH", "SOUTH", "EAST", "WEST", None])
    if ship_action is not None:
        action[ship_id] = ship_action
    return action
from kaggle_environments import make
env = make("halite", debug=True)
env.run(["agent.py", "random", "random", "random"])
env.render(mode="ipython", width=800, height=600)
from kaggle_environments import evaluate
def mean_reward(rewards):
    wins = 0
    ties = 0
    loses = 0
    for r in rewards:
        r0 = 0 if r[0] is None else r[0]
        r1 = 0 if r[1] is None else r[1]
        if r0 > r1:
            wins += 1
        elif r1 > r0:
            loses += 1
        else:
            ties += 1
    return f'wins={wins/len(rewards)}, ties={ties/len(rewards)}, loses={loses/len(rewards)}'

# Run multiple episodes to estimate its performance.
# Setup agentExec as LOCAL to run in memory (runs faster) without process isolation.
print("Swarm Agent vs Random Agent:", mean_reward(evaluate(
    "halite",
    ["agent.py", "random", "random", "random"],
    num_episodes=10, configuration={"agentExec": "LOCAL"}
)))