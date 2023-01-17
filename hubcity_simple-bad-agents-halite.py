# !curl -X PURGE https://pypi.org/simple/kaggle-environments
!pip install 'kaggle-environments>=0.2.1'
def display_score(env):
    for state in env.steps[-1]:
        player = state.observation.player
        score = state.reward
        print(f'Player {player}: {score}')

# I wanted an agent that would live the entire game
# The "random" agent doesn't always do that
def yard_only_agent(obs, config):
    actions = {}
    me = obs.players[obs.player]
    num_yards = len(me[1])
    if num_yards == 0:
        first_ship = next(iter(me[2].keys()))
        actions = { first_ship: "CONVERT" }
    return actions

import random 

from kaggle_environments.envs.halite.halite import get_to_pos

def create_enemy_ship_possible_map(obs, config):
    map = [0] * config.size * config.size
    player = obs.player
    for idx, opp in enumerate(obs.players):
        if idx == player:
            continue
        for ship in opp[2].values():
            map[ship[0]] = 1
            for dir in ["NORTH", "SOUTH", "EAST", "WEST"]:
                map[get_to_pos(config.size, ship[0], dir)] = 1
    return map

def create_enemy_yard_map(obs, config):
    map = [0] * config.size * config.size
    player = obs.player
    for idx, opp in enumerate(obs.players):
        if idx == player:
            continue
        for yard_pos in opp[1].values():
            map[yard_pos] = 1
    return map

def runaway_agent(obs, config):
    me = obs.players[obs.player]
    first_ship = next(iter(me[2].keys()))
    pos = me[2][first_ship][0]
    cargo = me[2][first_ship][1]
    esp = create_enemy_ship_possible_map(obs, config)
    ey = create_enemy_yard_map(obs, config)
    bad_square = [a+b for a,b in zip(esp, ey)]
    # ignore negative halite
    good_square = [x if x >= 0 else 0 for x in obs.halite]
    square_score = [-b if b > 0 else g for b,g in zip(bad_square, good_square)]
    moves = ["NORTH", "SOUTH", "EAST", "WEST"]
    random.shuffle(moves)
    best_score = square_score[pos]
    best_move = ""
    actions = {}
    for move in moves:
        new_pos = get_to_pos(config.size, pos, move)
        pos_score = square_score[new_pos]
        if pos_score > best_score or (pos_score == 0 and best_score == 0):
            best_score = pos_score
            best_move = move
            actions = { first_ship: best_move }
    return actions

def run_yard_agent(obs, config):
    actions = {}
    me = obs.players[obs.player]
    if len(me[2]) > 0:  # if I have a ship
        first_ship = next(iter(me[2].keys()))
        if me[2][first_ship][1] > config.convertCost:
            actions = { first_ship: "CONVERT" }
        else:
            actions = runaway_agent(obs, config)
    return actions

def always_one_agent(obs, config):
    me = obs.players[obs.player]
    num_yards = len(me[1])
    num_ships = len(me[2])
    if num_yards == 0:
        first_ship = next(iter(me[2].keys()))
        actions = { first_ship: "CONVERT" }
    elif num_ships == 0:
        first_yard = next(iter(me[1].keys()))
        actions = { first_yard: "SPAWN" }
    else:
        actions = runaway_agent(obs, config)
    return actions

def run_yard_one_agent(obs, config):
    me = obs.players[obs.player]
    num_ships = len(me[2])
    if num_ships == 0:
        first_yard = next(iter(me[1].keys()))
        actions = { first_yard: "SPAWN" }
    else:
        actions = run_yard_agent(obs, config)
    return actions

def take_invalid_action_agent(obs, config):
    print("Action requested")
    return {"Non-existant ship": "CONVERT"}

from kaggle_environments import make

env = make("halite", configuration={"agentExec": "LOCAL"}, debug=True)
env.run(["random", yard_only_agent, runaway_agent, take_invalid_action_agent])
env.render(mode="ipython", width=800, height=600)
display_score(env)
env = make("halite", configuration={"agentExec": "LOCAL"}, debug=True)
env.run(["random", run_yard_agent, always_one_agent, run_yard_one_agent])
env.render(mode="ipython", width=800, height=600)
display_score(env)
from enum import Enum
from collections import OrderedDict

def create_friendly_ship_possible_map(obs, config, ignore_id):
    map = [0] * config.size * config.size
    player = obs.player
    me = obs.players[player]
    for id, ship in me[2].items():
        if id == ignore_id:
            continue
        map[ship[0]] = 1
        for dir in ["NORTH", "SOUTH", "EAST", "WEST"]:
            map[get_to_pos(config.size, ship[0], dir)] = 1
    return map

def create_friendly_yard_map(obs, config):
    map = [0] * config.size * config.size
    player = obs.player
    me = obs.players[player]
    for yard_pos in me[1].values():
        map[yard_pos] = 1
    return map

def yard_actions(obs, config):
    actions = {}
    me = obs.players[obs.player]
    bank = me[0]
    num_yards = len(me[1])
    num_ships = len(me[2])
    max_ships = max([len(p[2]) for p in obs.players])
    if (num_ships < (max_ships/2) or num_ships == 0) and bank > config.spawnCost:
        pick_yard = random.choice([id for id in me[1].keys()])
        actions[pick_yard] = "SPAWN"
    return actions

def ship_actions(obs, config):
    actions = {}
    is_last_move = (obs.step == config.episodeSteps-2)
    moves = ["NORTH", "SOUTH", "EAST", "WEST"]
    esp = create_enemy_ship_possible_map(obs, config)
    ey = create_enemy_yard_map(obs, config)
    fy = create_friendly_yard_map(obs, config)
    good_square = [x if x >= 0 else 0 for x in obs.halite]
    me = obs.players[obs.player]
    num_yards = len(me[1])
    for id in me[2]:
        pos = me[2][id][0]
        cargo = me[2][id][1]
        fsp = create_friendly_ship_possible_map(obs, config, id)
        bad_square = [a+b+c for a,b,c in zip(esp, ey, fsp)]
        square_score = [-b if b > 0 else g for b,g in zip(bad_square, good_square)]
        if cargo > 0 and square_score[pos] > 0:
            # stay on a positive square and collect if we already have cargo
            pass
        elif cargo > 0 and fy[pos] > 0:
            # stay on shipyard to unload cargo
            pass
        elif cargo > (config.convertCost + config.spawnCost):
            # we have enough cargo to build a yard and replace this ship
            actions[id] = "CONVERT"
        elif cargo > 0 and num_yards == 0:
            # we've completely mined (see above) our first square, build our first shipyard
            actions[id] = "CONVERT"
        elif cargo > config.convertCost and is_last_move:
            # this is the last move and this ship isn't at a shipyard (see above)
            actions[id] = "CONVERT"
        else:
            # move to better adjacent halite while avoiding danger
            best_score = square_score[pos]
            random.shuffle(moves)
            for move in moves:
                new_pos = get_to_pos(config.size, pos, move)
                pos_score = square_score[new_pos]
                if pos_score > best_score or (pos_score == 0 and best_score == 0):
                    best_score = pos_score
                    actions[id] = move
    return actions

class Actions(Enum):
    NORTH = 1
    SOUTH = 2
    EAST = 3
    WEST = 4
    CONVERT = 5
    SPAWN = 6

def simple_agent(obs, config):
    actions = {}
    me = obs.players[obs.player]
    if len(me[1]) > 0:
        actions.update(yard_actions(obs, config))
    if len(me[2]) > 0:
        actions.update(ship_actions(obs, config))
    # due to a kaggle bug all actions need to be ordered so that CONVERT/SPAWN are last
    actions_list = list(actions.items())
    sorted_list = sorted(actions_list, key=lambda x: Actions[x[1]].value)
    # I realize regular dicts are order preserving in python 3.7 and
    # also in 3.6 as an implementation detail.  I didn't want to make
    # any assumptions about the python version kaggle uses
    actions = OrderedDict(sorted_list)
    return actions

env = make("halite", configuration={"agentExec": "LOCAL"}, debug=True)
env.run(["random", run_yard_agent, run_yard_one_agent, simple_agent])
env.render(mode="ipython", width=800, height=600)
display_score(env)
