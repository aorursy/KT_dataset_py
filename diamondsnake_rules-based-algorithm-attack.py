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
###### makes sure next spot has enough halite

###### go to nearest good spot

###### create multiple ships

# avoid friendly crashes

# create second shipyard

# choose shipyard location

# create n shipyards

# avoid enemy crashes

# change destination if occupied

# kamikaze ship
%%writefile submission.py



# Imports helper functions

from kaggle_environments.envs.halite.helpers import *



# Returns best direction to move from one position (fromPos) to another (toPos)

# Example: If I'm at pos 0 and want to get to pos 55, which direction should I choose?

def getDirTo(fromPos, toPos, size):

    fromX, fromY = divmod(fromPos[0],size), divmod(fromPos[1],size)

    toX, toY = divmod(toPos[0],size), divmod(toPos[1],size)

    if fromX < toX: return ShipAction.EAST

    if fromX > toX: return ShipAction.WEST

    if fromY < toY: return ShipAction.NORTH

    if fromY > toY: return ShipAction.SOUTH

    



# Directions a ship can move

#directions = [ShipAction.NORTH, ShipAction.SOUTH, ShipAction.EAST, ShipAction.WEST]

directions = [[ShipAction.NORTH,ShipAction.WEST], [ShipAction.SOUTH,ShipAction.EAST], [ShipAction.WEST,ShipAction.SOUTH], [ShipAction.EAST,ShipAction.NORTH]]



# Will keep track of whether a ship is collecting halite or carrying cargo to a shipyard

ship_states = {}



# define movement positions 

dirs = [['north','west'], ['south','east'], ['west','south'], ['east','north']]





# Returns the commands we send to our ships and shipyards

def agent(obs, config):

    size = config.size

    board = Board(obs, config)

    me = board.current_player

    

    ship_max_halite = 500

    location_min_halite = 100

    max_miners = 2

    max_shipyards = 1

    

    player_ids = [player.id for player in board.players.values()]

    shipyard_ids = [shipyard.id for shipyard in board.shipyards.values()]

    shipyard_positions = [shipyard.position for shipyard in board.shipyards.values()]

    my_shipyard_positions = [shipyard.position for shipyard in me.shipyards]

    enemy_shipyard_positions = [i for i in shipyard_positions if i not in my_shipyard_positions]

    



    # If there are no ships, use first shipyard to spawn a ship.

    if len(me.ships) < len(enemy_shipyard_positions)+max_miners and len(me.shipyards) > 0:

        me.shipyards[0].next_action = ShipyardAction.SPAWN



    # If there are no shipyards, convert first ship into shipyard.

    if len(me.shipyards) == 0 and len(me.ships) > 0:

        me.ships[0].next_action = ShipAction.CONVERT

        



    for ship_no in range(len(me.ships)):

        ship = me.ships[ship_no]

        

        # attack ships

        if ship_no>=max_miners and ship_no<len(enemy_shipyard_positions)+max_miners:

            ship.next_action = getDirTo(ship.position, enemy_shipyard_positions[ship_no-max_miners], size)

            

        # mining ships

        if ship.next_action == None:

            

            ### Part 1: Set the ship's state 

            if ship.halite < 200: # If cargo is too low, collect halite

                ship_states[ship.id] = "COLLECT"

            if ship.halite > ship_max_halite and len(me.shipyards) > 0: # If cargo gets very big, deposit halite

                ship_states[ship.id] = "DEPOSIT"

                

            ### Part 2: Use the ship's state to select an action

            if ship_states[ship.id] == "COLLECT":

                # If halite at current location running low, 

                # find closest square with enough halite

                if ship.cell.halite < location_min_halite:

                    if ship.cell.halite > location_min_halite:

                        do = 'mine'

                    else:

                        neighbors = [getattr(ship.cell, dirs[ship_no][0]).halite, getattr(ship.cell, dirs[ship_no][1]).halite]

                        best = max(range(len(neighbors)), key=neighbors.__getitem__)

                        if neighbors[best] >= location_min_halite:

                            ship.next_action = directions[ship_no][best]

                        # look one step further away

                        else:

                            next_neighbors = [getattr(getattr(ship.cell, dirs[ship_no][0]), dirs[ship_no][0]).halite, 

                                              getattr(getattr(ship.cell, dirs[ship_no][0]), dirs[ship_no][1]).halite, 

                                              getattr(getattr(ship.cell, dirs[ship_no][1]), dirs[ship_no][0]).halite,

                                              getattr(getattr(ship.cell, dirs[ship_no][1]), dirs[ship_no][1]).halite]

                            next_neighbors_pos = [getattr(getattr(ship.cell, dirs[ship_no][0]), dirs[ship_no][0]).position, 

                                                  getattr(getattr(ship.cell, dirs[ship_no][0]), dirs[ship_no][1]).position, 

                                                  getattr(getattr(ship.cell, dirs[ship_no][1]), dirs[ship_no][0]).position,

                                                  getattr(getattr(ship.cell, dirs[ship_no][1]), dirs[ship_no][1]).position]

                            next_best = max(range(len(next_neighbors)), key=next_neighbors.__getitem__)

                            if next_neighbors[next_best] >= location_min_halite:

                                ship.next_action = getDirTo(ship.position, next_neighbors_pos[next_best], size)

                            else:

                                ship.next_action = directions[ship_no][0]

            if ship_states[ship.id] == "DEPOSIT":

                # Move towards shipyard to deposit cargo

                direction = getDirTo(ship.position, me.shipyards[0].position, size)

                if direction: ship.next_action = direction

    

                            

    return me.next_actions
from kaggle_environments import make

env = make("halite", debug=True)

env.run(["submission.py", "random", "random", "random"])

env.render(mode="ipython", width=800, height=600)