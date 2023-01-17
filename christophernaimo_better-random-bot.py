"""
Added feature of keeping at least one ship and at least one shipyard active when possible
"""


# Set Up Environment
from kaggle_environments import evaluate, make
env = make("halite", configuration={ "episodeSteps": 400 }, debug=True)
print (env.configuration)
%%writefile submission.py

from kaggle_environments.envs.halite.helpers import *
import random


def agent(obs, config):
    board = Board(obs, config)
    me = board.current_player

    # at least one shipyard
    if not me.shipyards:
        me.ships[0].next_action = ShipAction.CONVERT
        return me.next_actions

    # at least one ship
    if not me.ships and me.shipyards:
        me.shipyards[0].next_action = ShipyardAction.SPAWN
        return me.next_actions

    # Set ship actions
    for ship in me.ships:
        ship.next_action = random.choice(
            [ShipAction.NORTH, ShipAction.EAST, ShipAction.SOUTH, ShipAction.WEST, None])

    # Set shipyard actions
    for shipyard in me.shipyards:
        # 10% chance of spawning
        shipyard.next_action = random.choice([ShipyardAction.SPAWN] + [None] * 9)

    return me.next_actions

env.run(["/kaggle/working/submission.py", "random","random","random"])
env.render(mode="ipython", width=800, height=600)