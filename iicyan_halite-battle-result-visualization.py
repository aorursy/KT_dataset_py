%%writefile submission.py

# for Debug/Train previous line (%%writefile submission.py) should be commented out, uncomment to write submission.py



#FUNCTIONS###################################################

def get_map_and_average_halite(obs):

    """

        get average amount of halite per halite source

        and map as two dimensional array of objects and set amounts of halite in each cell

    """

    game_map = []

    halite_sources_amount = 0

    halite_total_amount = 0

    for x in range(conf.size):

        game_map.append([])

        for y in range(conf.size):

            game_map[x].append({

                # value will be ID of owner

                "shipyard": None,

                # value will be ID of owner

                "ship": None,

                # value will be amount of halite

                "ship_cargo": None,

                # amount of halite

                "halite": obs.halite[conf.size * y + x]

            })

            if game_map[x][y]["halite"] > 0:

                halite_total_amount += game_map[x][y]["halite"]

                halite_sources_amount += 1

    average_halite = halite_total_amount / halite_sources_amount

    return game_map, average_halite



def get_swarm_units_coords_and_update_map(s_env):

    """ get lists of coords of Swarm's units and update locations of ships and shipyards on the map """

    # arrays of (x, y) coords

    swarm_shipyards_coords = []

    swarm_ships_coords = []

    # place on the map locations of units of every player

    for player in range(len(s_env["obs"].players)):

        # place on the map locations of every shipyard of the player

        shipyards = list(s_env["obs"].players[player][1].values())

        for shipyard in shipyards:

            x = shipyard % conf.size

            y = shipyard // conf.size

            # place shipyard on the map

            s_env["map"][x][y]["shipyard"] = player

            if player == s_env["obs"].player:

                swarm_shipyards_coords.append((x, y))

        # place on the map locations of every ship of the player

        ships = list(s_env["obs"].players[player][2].values())

        for ship in ships:

            x = ship[0] % conf.size

            y = ship[0] // conf.size

            # place ship on the map

            s_env["map"][x][y]["ship"] = player

            s_env["map"][x][y]["ship_cargo"] = ship[1]

            if player == s_env["obs"].player:

                swarm_ships_coords.append((x, y))

    return swarm_shipyards_coords, swarm_ships_coords



def get_c(c):

    """ get coordinate, considering donut type of the map """

    return c % conf.size



def clear(x, y, player, game_map):

    """ check if cell is safe to move in """

    # if there is no shipyard, or there is player's shipyard

    # and there is no ship

    if ((game_map[x][y]["shipyard"] == player or game_map[x][y]["shipyard"] == None) and

            game_map[x][y]["ship"] == None):

        return True

    return False



def move_ship(x_initial, y_initial, actions, s_env, ship_index):

    """ move the ship according to first acceptable tactic """

    ok, actions = go_for_halite(x_initial, y_initial, s_env["ships_keys"][ship_index], actions, s_env, ship_index)

    if ok:

        return actions

    ok, actions = unload_halite(x_initial, y_initial, s_env["ships_keys"][ship_index], actions, s_env, ship_index)

    if ok:

        return actions

    return standard_patrol(x_initial, y_initial, s_env["ships_keys"][ship_index], actions, s_env, ship_index)



def go_for_halite(x_initial, y_initial, ship_id, actions, s_env, ship_index):

    """ ship will go to safe cell with enough halite, if it is found """

    # biggest amount of halite among scanned cells

    most_halite = s_env["low_amount_of_halite"]

    for d in range(len(directions_list)):

        x = directions_list[d]["x"](x_initial)

        y = directions_list[d]["y"](y_initial)

        # if cell is safe to move in

        if (clear(x, y, s_env["obs"].player, s_env["map"]) and

                not hostile_ship_near(x, y, s_env["obs"].player, s_env["map"], s_env["ships_values"][ship_index][1])):

            # if current cell has more than biggest amount of halite

            if s_env["map"][x][y]["halite"] > most_halite:

                most_halite = s_env["map"][x][y]["halite"]

                direction = directions_list[d]["direction"]

                direction_x = x

                direction_y = y

    # if cell is safe to move in and has substantial amount of halite

    if most_halite > s_env["low_amount_of_halite"]:

        actions[ship_id] = direction

        s_env["map"][x_initial][y_initial]["ship"] = None

        s_env["map"][direction_x][direction_y]["ship"] = s_env["obs"].player

        return True, actions

    return False, actions



def unload_halite(x_initial, y_initial, ship_id, actions, s_env, ship_index):

    """ unload ship's halite if there is any and Swarm's shipyard is near """

    if s_env["ships_values"][ship_index][1] > 0:

        for d in range(len(directions_list)):

            x = directions_list[d]["x"](x_initial)

            y = directions_list[d]["y"](y_initial)

            # if shipyard is there and unoccupied

            if (clear(x, y, s_env["obs"].player, s_env["map"]) and

                    s_env["map"][x][y]["shipyard"] == s_env["obs"].player):

                actions[ship_id] = directions_list[d]["direction"]

                s_env["map"][x_initial][y_initial]["ship"] = None

                s_env["map"][x][y]["ship"] = s_env["obs"].player

                return True, actions

    return False, actions



def standard_patrol(x_initial, y_initial, ship_id, actions, s_env, ship_index):

    """ 

        ship will move in expanding circles clockwise or counterclockwise

        until reaching maximum radius, then radius will be minimal again

    """

    directions = ships_data[ship_id]["directions"]

    # set index of direction

    i = ships_data[ship_id]["directions_index"]

    direction_found = False

    for j in range(len(directions)):

        x = directions[i]["x"](x_initial)

        y = directions[i]["y"](y_initial)

        # if cell is ok to move in

        if (clear(x, y, s_env["obs"].player, s_env["map"]) and

                (s_env["map"][x][y]["shipyard"] == s_env["obs"].player or

                not hostile_ship_near(x, y, s_env["obs"].player, s_env["map"], s_env["ships_values"][ship_index][1]))):

            ships_data[ship_id]["moves_done"] += 1

            # apply changes to game_map, to avoid collisions of player's ships next turn

            s_env["map"][x_initial][y_initial]["ship"] = None

            s_env["map"][x][y]["ship"] = s_env["obs"].player

            # if it was last move in this direction

            if ships_data[ship_id]["moves_done"] >= ships_data[ship_id]["ship_max_moves"]:

                ships_data[ship_id]["moves_done"] = 0

                ships_data[ship_id]["directions_index"] += 1

                # if it is last direction in a list

                if ships_data[ship_id]["directions_index"] >= len(directions):

                    ships_data[ship_id]["directions_index"] = 0

                    ships_data[ship_id]["ship_max_moves"] += 1

                    # if ship_max_moves reached maximum radius expansion

                    if ships_data[ship_id]["ship_max_moves"] > max_moves_amount:

                        ships_data[ship_id]["ship_max_moves"] = 2

            actions[ship_id] = directions[i]["direction"]

            direction_found = True

            break

        else:

            # loop through directions

            i += 1

            if i >= len(directions):

                i = 0

    # if ship is not on shipyard and hostile ship is near

    if (not direction_found and s_env["map"][x_initial][y_initial]["shipyard"] == None and

            hostile_ship_near(x_initial, y_initial, s_env["obs"].player, s_env["map"],

                              s_env["ships_values"][ship_index][1])):

        # if there is enough halite to convert

        if s_env["ships_values"][ship_index][1] >= conf.convertCost:

            actions[ship_id] = "CONVERT"

            s_env["map"][x_initial][y_initial]["ship"] = None

        else:

            for i in range(len(directions)):

                x = directions[i]["x"](x_initial)

                y = directions[i]["y"](y_initial)

                # if it is opponent's shipyard

                if s_env["map"][x][y]["shipyard"] != None:

                    # apply changes to game_map, to avoid collisions of player's ships next turn

                    s_env["map"][x_initial][y_initial]["ship"] = None

                    s_env["map"][x][y]["ship"] = s_env["obs"].player

                    actions[ship_id] = directions[i]["direction"]

                    break

    return actions



def get_directions(i0, i1, i2, i3):

    """ get list of directions in a certain sequence """

    return [directions_list[i0], directions_list[i1], directions_list[i2], directions_list[i3]]



def hostile_ship_near(x, y, player, m, cargo):

    """ check if hostile ship is in one move away from game_map[x][y] and has less or equal halite """

    # m = game map

    n = get_c(y - 1)

    e = get_c(x + 1)

    s = get_c(y + 1)

    w = get_c(x - 1)

    if (

            (m[x][n]["ship"] != player and m[x][n]["ship"] != None and m[x][n]["ship_cargo"] <= cargo) or

            (m[x][s]["ship"] != player and m[x][s]["ship"] != None and m[x][s]["ship_cargo"] <= cargo) or

            (m[e][y]["ship"] != player and m[e][y]["ship"] != None and m[e][y]["ship_cargo"] <= cargo) or

            (m[w][y]["ship"] != player and m[w][y]["ship"] != None and m[w][y]["ship_cargo"] <= cargo)

        ):

        return True

    return False



def to_spawn_or_not_to_spawn(s_env):

    """ to spawn, or not to spawn, that is the question """

    # get ships_max_amount to decide whether to spawn new ships or not

    ships_max_amount = 0

    # decrease spawn_limit if half or less of game steps remained

    if s_env["obs"].step < middle_step:

        # sum of all ships of every player

        total_ships_amount = 0

        for player in range(len(s_env["obs"].players)):

            total_ships_amount += len(s_env["obs"].players[player][2])

        # to avoid division by zero

        if total_ships_amount > 0:

            ships_max_amount = (s_env["average_halite"] // total_ships_amount) * 10

        # if ships_max_amount is less than minimal allowed amount of ships in the Swarm

    if ships_max_amount < ships_min_amount:

        ships_max_amount = ships_min_amount

    return ships_max_amount



def define_some_globals(configuration):

    """ define some of the global variables """

    global conf

    global middle_step

    global convert_threshold

    global max_moves_amount

    global globals_not_defined

    conf = configuration

    middle_step = conf.episodeSteps // 2

    convert_threshold = conf.convertCost + conf.spawnCost * 3

    max_moves_amount = conf.size

    globals_not_defined = False



def adapt_environment(observation, configuration):

    """ adapt environment for the Swarm """

    s_env = {}

    s_env["obs"] = observation

    if globals_not_defined:

        define_some_globals(configuration)

    s_env["map"], s_env["average_halite"] = get_map_and_average_halite(s_env["obs"])

    s_env["low_amount_of_halite"] = s_env["average_halite"] / 2

    s_env["swarm_halite"] = s_env["obs"].players[s_env["obs"].player][0]

    s_env["swarm_shipyards_coords"], s_env["swarm_ships_coords"] = get_swarm_units_coords_and_update_map(s_env)

    s_env["ships_keys"] = list(s_env["obs"].players[s_env["obs"].player][2].keys())

    s_env["ships_values"] = list(s_env["obs"].players[s_env["obs"].player][2].values())

    s_env["shipyards_keys"] = list(s_env["obs"].players[s_env["obs"].player][1].keys())

    s_env["ships_max_amount"] = to_spawn_or_not_to_spawn(s_env)

    return s_env

    

def actions_of_ships(s_env):

    """ actions of every ship of the Swarm """

    global movement_tactics_index

    actions = {}

    shipyards_amount = len(s_env["shipyards_keys"])

    for i in range(len(s_env["swarm_ships_coords"])):

        x = s_env["swarm_ships_coords"][i][0]

        y = s_env["swarm_ships_coords"][i][1]

        # if this is a new ship

        if s_env["ships_keys"][i] not in ships_data:

            ships_data[s_env["ships_keys"][i]] = {

                "moves_done": 0,

                "ship_max_moves": 2,

                "directions": movement_tactics[movement_tactics_index]["directions"],

                "directions_index": 0

            }

            movement_tactics_index += 1

            if movement_tactics_index >= movement_tactics_amount:

                movement_tactics_index = 0

        # if it is last step

        elif s_env["obs"].step == (conf.episodeSteps - 2) and s_env["ships_values"][i][1] >= conf.convertCost:

            actions[s_env["ships_keys"][i]] = "CONVERT"

            s_env["map"][x][y]["ship"] = None

        # if there is no shipyards, necessity to have shipyard, no hostile ships near,

        # first half of the game and enough halite to spawn few ships

        elif (shipyards_amount == 0 and len(s_env["ships_keys"]) < s_env["ships_max_amount"] and

                not hostile_ship_near(x, y, s_env["obs"].player, s_env["map"], s_env["ships_values"][i][1]) and

                s_env["obs"].step < middle_step and

                (s_env["swarm_halite"] + s_env["ships_values"][i][1]) >= convert_threshold):

            s_env["swarm_halite"] = s_env["swarm_halite"] + s_env["ships_values"][i][1] - conf.convertCost

            actions[s_env["ships_keys"][i]] = "CONVERT"

            s_env["map"][x][y]["ship"] = None

            shipyards_amount += 1

        else:

            # if this cell has low amount of halite or hostile ship is near

            if (s_env["map"][x][y]["halite"] < s_env["low_amount_of_halite"] or

                    hostile_ship_near(x, y, s_env["obs"].player, s_env["map"], s_env["ships_values"][i][1])):

                actions = move_ship(x, y, actions, s_env, i)

    return actions

     

def actions_of_shipyards(actions, s_env):

    """ actions of every shipyard of the Swarm """

    ships_amount = len(s_env["ships_keys"])

    # spawn ships from every shipyard, if possible

    for i in range(len(s_env["swarm_shipyards_coords"])):

        if s_env["swarm_halite"] >= conf.spawnCost and ships_amount < s_env["ships_max_amount"]:

            x = s_env["swarm_shipyards_coords"][i][0]

            y = s_env["swarm_shipyards_coords"][i][1]

            # if there is currently no ship on shipyard

            if clear(x, y, s_env["obs"].player, s_env["map"]):

                s_env["swarm_halite"] -= conf.spawnCost

                actions[s_env["shipyards_keys"][i]] = "SPAWN"

                s_env["map"][x][y]["ship"] = s_env["obs"].player

                ships_amount += 1

        else:

            break

    return actions





#GLOBAL_VARIABLES#############################################

conf = None

middle_step = None

# max amount of moves in one direction before turning

max_moves_amount = None

# threshold of harvested by a ship halite to convert

convert_threshold = None

# object with ship ids and their data

ships_data = {}

# initial movement_tactics index

movement_tactics_index = 0

# minimum amount of ships that should be in the Swarm at any time

ships_min_amount = 10

# not all global variables are defined

globals_not_defined = True



# list of directions

directions_list = [

    {

        "direction": "NORTH",

        "x": lambda z: z,

        "y": lambda z: get_c(z - 1)

    },

    {

        "direction": "EAST",

        "x": lambda z: get_c(z + 1),

        "y": lambda z: z

    },

    {

        "direction": "SOUTH",

        "x": lambda z: z,

        "y": lambda z: get_c(z + 1)

    },

    {

        "direction": "WEST",

        "x": lambda z: get_c(z - 1),

        "y": lambda z: z

    }

]



# list of movement tactics

movement_tactics = [

    # N -> E -> S -> W

    {"directions": get_directions(0, 1, 2, 3)},

    # S -> E -> N -> W

    {"directions": get_directions(2, 1, 0, 3)},

    # N -> W -> S -> E

    {"directions": get_directions(0, 3, 2, 1)},

    # S -> W -> N -> E

    {"directions": get_directions(2, 3, 0, 1)},

    # E -> N -> W -> S

    {"directions": get_directions(1, 0, 3, 2)},

    # W -> S -> E -> N

    {"directions": get_directions(3, 2, 1, 0)},

    # E -> S -> W -> N

    {"directions": get_directions(1, 2, 3, 0)},

    # W -> N -> E -> S

    {"directions": get_directions(3, 0, 1, 2)},

]

movement_tactics_amount = len(movement_tactics)



#THE_SWARM####################################################

def swarm_agent(observation, configuration):

    """ RELEASE THE SWARM!!! """

    s_env = adapt_environment(observation, configuration)

    actions = actions_of_ships(s_env)

    actions = actions_of_shipyards(actions, s_env)

    return actions
from kaggle_environments.envs.halite.helpers import *

from kaggle_environments import evaluate, make

from kaggle_environments.envs.halite.helpers import *

import numpy as np

import pandas as pd

import submission

env = make("halite", debug=True)



trainer = env.train([None, "submission.py", "submission.py", "submission.py"])

observation: Observation = trainer.reset()

pre_count = 1

steps = []

board_halite = []

p0_halite = []

p1_halite = []

p2_halite = []

p3_halite = []

p0_cargo = []

p1_cargo = []

p2_cargo = []

p3_cargo = []

p0_ships = []

p1_ships = []

p2_ships = []

p3_ships = []

p0_shipyards = []

p1_shipyards = []

p2_shipyards = []

p3_shipyards = []

while not env.done:

    my_action = submission.swarm_agent(observation, env.configuration)

    observation, reward, done, info = trainer.step(my_action)

    board = Board(observation, env.configuration)

    steps.append(observation.step)

    last_step = observation.step

    board_halite.append(sum(observation.halite))

    p0_halite.append(board.players[0].halite)

    p1_halite.append(board.players[1].halite)

    p2_halite.append(board.players[2].halite)

    p3_halite.append(board.players[3].halite)

    p0_cargo.append(sum([ship.halite for ship in board.players[0].ships]))

    p1_cargo.append(sum([ship.halite for ship in board.players[1].ships]))

    p2_cargo.append(sum([ship.halite for ship in board.players[2].ships]))

    p3_cargo.append(sum([ship.halite for ship in board.players[3].ships]))

    p0_ships.append(len(board.players[0].ship_ids))

    p1_ships.append(len(board.players[1].ship_ids))

    p2_ships.append(len(board.players[2].ship_ids))

    p3_ships.append(len(board.players[3].ship_ids))

    p0_shipyards.append(len(board.players[0].shipyard_ids))

    p1_shipyards.append(len(board.players[1].shipyard_ids))

    p2_shipyards.append(len(board.players[2].shipyard_ids))

    p3_shipyards.append(len(board.players[3].shipyard_ids))

env.render(mode="ipython", width=800, height=600)

df = pd.DataFrame(

data={'step': steps, 'board_halite': board_halite,

    'p0_halite': p0_halite,

    'p1_halite': p1_halite,

    'p2_halite': p2_halite,

    'p3_halite': p3_halite,

    'p0_cargo': p0_cargo,

    'p1_cargo': p1_cargo,

    'p2_cargo': p2_cargo,

    'p3_cargo': p3_cargo,

    'p0_ships': p0_ships,

    'p1_ships': p1_ships,

    'p2_ships': p2_ships,

    'p3_ships': p3_ships,

    'p0_shipyards': p0_shipyards,

    'p1_shipyards': p1_shipyards,

    'p2_shipyards': p2_shipyards,

    'p3_shipyards': p3_shipyards,

},

columns=['step', 'board_halite',

     'p0_halite',

     'p1_halite',

     'p2_halite',

     'p3_halite',

     'p0_cargo',

     'p1_cargo',

     'p2_cargo',

     'p3_cargo',

     'p0_ships',

     'p1_ships',

     'p2_ships',

     'p3_ships',

     'p0_shipyards',

     'p1_shipyards',

     'p2_shipyards',

     'p3_shipyards',

 ]

)

df['p0_total_halite']  = df['p0_halite'] + df['p0_cargo']

df['p1_total_halite']  = df['p1_halite'] + df['p1_cargo']

df['p2_total_halite']  = df['p2_halite'] + df['p2_cargo']

df['p3_total_halite']  = df['p3_halite'] + df['p3_cargo']



df
df.describe()
import seaborn as sns

import numpy as np                             

import pandas as pd                              

import matplotlib.pyplot as plt

import seaborn as sns; sns.set() 

sns.set()

df0 = pd.DataFrame(

data={'player':'p0','step': steps, 'board_halite': board_halite,

    'halite': p0_halite,

    'cargo': p0_cargo,

    'ships': p0_ships,

    'shipyards': p0_shipyards,

},

columns=['player','step', 'board_halite',

     'halite',

     'cargo',

     'ships',

     'shipyards',

 ]

)

df1 = pd.DataFrame(

data={'player':'p1','step': steps, 'board_halite': board_halite,

    'halite': p1_halite,

    'cargo': p1_cargo,

    'ships': p1_ships,

    'shipyards': p1_shipyards,

},

columns=['player','step', 'board_halite',

     'halite',

     'cargo',

     'ships',

     'shipyards',

 ]

)

df2 = pd.DataFrame(

data={'player':'p2','step': steps, 'board_halite': board_halite,

    'halite': p2_halite,

    'cargo': p2_cargo,

    'ships': p2_ships,

    'shipyards': p2_shipyards,

},

columns=['player','step', 'board_halite',

     'halite',

     'cargo',

     'ships',

     'shipyards',

 ]

)

df3 = pd.DataFrame(

data={'player':'p3','step': steps, 'board_halite': board_halite,

    'halite': p3_halite,

    'cargo': p3_cargo,

    'ships': p3_ships,

    'shipyards': p3_shipyards,

},

columns=['player','step', 'board_halite',

     'halite',

     'cargo',

     'ships',

     'shipyards',

 ]

)



df_merged = pd.concat([df0,df1,df2,df3])

df_merged['total_halite'] = df_merged['halite'] + df_merged['cargo']

df_merged['cargo_average']  = df_merged['cargo'] / df_merged['ships']

df_merged['cargo_percentage'] = df_merged['cargo'] / df_merged['total_halite']
df_merged
plt.figure(figsize=(12,8))

plt.title("player halite at game end", fontsize=15)

sns.barplot(data=df_merged[df_merged['step']==last_step],x='player',y='halite',ci=None)

plt.ylabel('halite', fontsize=12)

plt.xlabel('player', fontsize=12)

plt.show()
plt.figure(figsize=(12,8))

plt.title("average halite in game", fontsize=15)

sns.barplot(data=df_merged,x='player',y='halite',ci=None)

plt.ylabel('mean halite', fontsize=12)

plt.xlabel('player', fontsize=12)

plt.show()
plt.figure(figsize=(12,8))

plt.title("player cargo at game end", fontsize=15)

sns.barplot(data=df_merged[df_merged['step']==last_step],x='player',y='cargo',ci=None)

plt.ylabel('cargo', fontsize=12)

plt.xlabel('player', fontsize=12)

plt.show()
plt.figure(figsize=(12,8))

plt.title("player halite time line", fontsize=15)

sns.lineplot(data=df_merged,x='step',y='halite' ,hue='player')

plt.ylabel('halite', fontsize=12)

plt.show()
plt.figure(figsize=(12,8))

plt.title("player and board halite time line", fontsize=15)

sns.lineplot(data=df_merged,x='step',y='halite' ,hue='player')

sns.lineplot(data=df,x='step',y='board_halite' ,color='black')

plt.ylabel('halite', fontsize=12)

plt.show()
plt.figure(figsize=(12,8))

plt.title("player cargo time line", fontsize=15)

sns.lineplot(data=df_merged,x='step',y='cargo', hue='player')

plt.ylabel('cargo', fontsize=12)

plt.show()

plt.figure(figsize=(12,8))

plt.title("player cargo and board halite time line", fontsize=15)

sns.lineplot(data=df_merged,x='step',y='cargo', hue='player')

sns.lineplot(data=df,x='step',y='board_halite' ,color='black')

plt.ylabel('cargo', fontsize=12)

plt.show()

plt.figure(figsize=(12,8))

plt.title("total halite (halite + cargo) time line", fontsize=15)

sns.lineplot(data=df_merged,x='step',y='total_halite', hue='player')

plt.ylabel('halite (halite + cargo)', fontsize=12)

plt.show()

plt.figure(figsize=(12,8))

plt.title("cargo percentage (cargo / (halite + cargo)) time line", fontsize=15)

sns.lineplot(data=df_merged,x='step',y='cargo_percentage', hue='player')

plt.ylabel('cargo percentage (cargo / (halite + cargo))', fontsize=12)

plt.show()

plt.figure(figsize=(12,8))

plt.title("total shipyard count time line", fontsize=15)

sns.lineplot(data=df_merged,x='step',y='shipyards', hue='player')

plt.ylabel('shipyard count', fontsize=12)

plt.show()
plt.figure(figsize=(12,8))

plt.title("average shipyard count in game", fontsize=15)

sns.barplot(data=df_merged,x='player',y='shipyards',ci=None)

plt.ylabel('mean shipyard count', fontsize=12)

plt.xlabel('player', fontsize=12)

plt.show()
plt.figure(figsize=(12,8))

plt.title("total ship count time line", fontsize=15)

sns.lineplot(data=df_merged,x='step',y='ships', hue='player')

plt.ylabel('ship count', fontsize=12)

plt.show()

plt.figure(figsize=(12,8))

plt.title("average ship count in game", fontsize=15)

sns.barplot(data=df_merged,x='player',y='ships',ci=None)

plt.ylabel('mean ship count', fontsize=12)

plt.xlabel('player', fontsize=12)

plt.show()
plt.figure(figsize=(12,8))

plt.title("cargo average(cargo / ship count)  time line", fontsize=15)

sns.lineplot(data=df_merged,x='step',y='cargo_average', hue='player')

plt.ylabel('cargo average', fontsize=12)

plt.show()
