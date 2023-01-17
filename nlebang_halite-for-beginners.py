# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
from kaggle_environments import make, evaluate

env = make("halite", debug=True)

env.run(["random", "random", "random", "random"])

env.render(mode="ipython", width=800, height=600)
%%writefile submission0.py



# Imports helper functions

from kaggle_environments.envs.halite.helpers import *



# Returns best direction to move from one position (fromPos) to another (toPos)

# Example: If I'm at pos 0 and want to get to pos 55, which direction should I choose?

def getDirTo(fromPos, toPos, size):

    fromX, fromY = divmod(fromPos[0],size), divmod(fromPos[1],size)

    toX, toY = divmod(toPos[0],size), divmod(toPos[1],size)

    if fromY < toY: return ShipAction.NORTH

    if fromY > toY: return ShipAction.SOUTH

    if fromX < toX: return ShipAction.EAST

    if fromX > toX: return ShipAction.WEST



# Directions a ship can move

directions = [ShipAction.NORTH, ShipAction.EAST, ShipAction.SOUTH, ShipAction.WEST]



# Will keep track of whether a ship is collecting halite or carrying cargo to a shipyard

ship_states = {}



# Returns the commands we send to our ships and shipyards

def agent(obs, config):

    size = config.size

    board = Board(obs, config)

    me = board.current_player



    # If there are no ships, use first shipyard to spawn a ship.

    if len(me.ships) == 0 and len(me.shipyards) > 0:

        me.shipyards[0].next_action = ShipyardAction.SPAWN



    # If there are no shipyards, convert first ship into shipyard.

    if len(me.shipyards) == 0 and len(me.ships) > 0:

        me.ships[0].next_action = ShipAction.CONVERT

    

    for ship in me.ships:

        if ship.next_action == None:

            

            ### Part 1: Set the ship's state 

            if ship.halite < 200: # If cargo is too low, collect halite

                ship_states[ship.id] = "COLLECT"

            if ship.halite > 500: # If cargo gets very big, deposit halite

                ship_states[ship.id] = "DEPOSIT"

                

            ### Part 2: Use the ship's state to select an action

            if ship_states[ship.id] == "COLLECT":

                # If halite at current location running low, 

                # move to the adjacent square containing the most halite

                if ship.cell.halite < 100:

                    neighbors = [ship.cell.north.halite, ship.cell.east.halite, 

                                 ship.cell.south.halite, ship.cell.west.halite]

                    best = max(range(len(neighbors)), key=neighbors.__getitem__)

                    ship.next_action = directions[best]

            if ship_states[ship.id] == "DEPOSIT":

                # Move towards shipyard to deposit cargo

                direction = getDirTo(ship.position, me.shipyards[0].position, size)

                if direction: ship.next_action = direction

                

    return me.next_actions
from kaggle_environments import make

env = make("halite", debug=True)

env.run(["submission0.py", "random", "random", "random"])

env.render(mode="ipython", width=800, height=600)
%%writefile submission1.py



from kaggle_environments.envs.halite.helpers import *

from random import choice



def agent(obs,config):

    

    board = Board(obs,config)

    me = board.current_player

    

    # Set actions for each ship

    for ship in me.ships:

        ship.next_action = choice([ShipAction.NORTH,ShipAction.EAST,ShipAction.SOUTH,ShipAction.WEST,None])

    

    # Set actions for each shipyard

    for shipyard in me.shipyards:

        shipyard.next_action = None

    

    return me.next_actions
#Test this template bot

env.run(["submission1.py", "random","random","random"])

env.render(mode="ipython", width=800, height=600)

%%writefile submission.py

####################

# Helper functions #

####################



# Helper function we'll use for getting adjacent position with the most halite

def argmax(arr, key=None):

    return arr.index(max(arr, key=key)) if key else arr.index(max(arr))



# Converts position from 1D to 2D representation

def get_col_row(size, pos):

    return (pos % size, pos // size)



# Returns the position in some direction relative to the current position (pos) 

def get_to_pos(size, pos, direction):

    col, row = get_col_row(size, pos)

    if direction == "NORTH":

        return pos - size if pos >= size else size ** 2 - size + col

    elif direction == "SOUTH":

        return col if pos + size >= size ** 2 else pos + size

    elif direction == "EAST":

        return pos + 1 if col < size - 1 else row * size

    elif direction == "WEST":

        return pos - 1 if col > 0 else (row + 1) * size - 1



# Get positions in all directions relative to the current position (pos)

# Especially useful for figuring out how much halite is around you

def getAdjacent(pos, size):

    return [

        get_to_pos(size, pos, "NORTH"),

        get_to_pos(size, pos, "SOUTH"),

        get_to_pos(size, pos, "EAST"),

        get_to_pos(size, pos, "WEST"),

    ]



# Returns best direction to move from one position (fromPos) to another (toPos)

# Example: If I'm at pos 0 and want to get to pos 55, which direction should I choose?

def getDirTo(fromPos, toPos, size):

    fromY, fromX = divmod(fromPos, size)

    toY,   toX   = divmod(toPos,   size)

    if fromY < toY: return "SOUTH"

    if fromY > toY: return "NORTH"

    if fromX < toX: return "EAST"

    if fromX > toX: return "WEST"



# Possible directions a ship can move in

DIRS = ["NORTH", "SOUTH", "EAST", "WEST"]

# We'll use this to keep track of whether a ship is collecting halite or 

# carrying its cargo to a shipyard

ship_states = {}



#############

# The agent #

#############



def agent(obs, config):

    # Get the player's halite, shipyard locations, and ships (along with cargo) 

    player_halite, shipyards, ships = obs.players[obs.player]

    size = config["size"]

    # Initialize a dictionary containing commands that will be sent to the game

    action = {}



    # If there are no ships, use first shipyard to spawn a ship.

    if len(ships) == 0 and len(shipyards) > 0:

        uid = list(shipyards.keys())[0]

        action[uid] = "SPAWN"

        

    # If there are no shipyards, convert first ship into shipyard.

    if len(shipyards) == 0 and len(ships) > 0:

        uid = list(ships.keys())[0]

        action[uid] = "CONVERT"

        

    for uid, ship in ships.items():

        if uid not in action: # Ignore ships that will be converted to shipyards

            pos, cargo = ship # Get the ship's position and halite in cargo

            

            ### Part 1: Set the ship's state 

            if cargo < 200: # If cargo is too low, collect halite

                ship_states[uid] = "COLLECT"

            if cargo > 500: # If cargo gets very big, deposit halite

                ship_states[uid] = "DEPOSIT"

                

            ### Part 2: Use the ship's state to select an action

            if ship_states[uid] == "COLLECT":

                # If halite at current location running low, 

                # move to the adjacent square containing the most halite

                if obs.halite[pos] < 100:

                    best = argmax(getAdjacent(pos, size), key=obs.halite.__getitem__)

                    action[uid] = DIRS[best]

            

            if ship_states[uid] == "DEPOSIT":

                # Move towards shipyard to deposit cargo

                direction = getDirTo(pos, list(shipyards.values())[0], size)

                if direction: action[uid] = direction

                

    return action
#Test this starter bot

env.run(["submission.py", "random","random","random"])

env.render(mode="ipython", width=800, height=600)