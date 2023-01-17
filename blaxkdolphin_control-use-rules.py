!pip install kaggle-environments --upgrade
%%writefile submission.py



# Imports helper functions

import numpy as np

import random

from kaggle_environments import make

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

    

# Returns number of steps between one position (fromPos) to another (toPos)

def stepTo(fromPos, toPos, size):

    x, y = fromPos - toPos

    if x<0: x=-x

    if y<0: y=-y

    return min(x, size-x) + min(y, size-y)



def exp_halite(cell):

    return np.mean([cell.halite, cell.north.halite, cell.east.halite, cell.south.halite, cell.west.halite])



# Directions a ship can move

directions = [ShipAction.NORTH, ShipAction.EAST, ShipAction.SOUTH, ShipAction.WEST]



# Will keep track of whether a ship is collecting halite or carrying cargo to a shipyard

ship_states = {}

ship_moves = {}





# Returns the commands we send to our ships and shipyards

def agent(obs, config):

    size = config.size

    board = Board(obs, config)

    me = board.current_player

    

    ## avoid opponent's shipyards 

    avoid_positions = []

    for shipyard in board.shipyards.values():

        if not shipyard.player.is_current_player:

            avoid_positions.append(shipyard.position)      

    ## avoid ships       

    for ship in board.ships.values():

        avoid_positions.append(ship.position)

    

    ### spawn a ship

    ## If there are no ships within N steps to this shipyard, then spawn a ship

    for shipyard in me.shipyards:

        spawn_flag = True

        for ship in me.ships:

            if stepTo(shipyard.position, ship.position, size) <= 5: spawn_flag = False

        if spawn_flag: 

            shipyard.next_action = ShipyardAction.SPAWN



    # If there are no shipyards within N steps to this ship, convert ship into shipyard

    for ship in me.ships: 

        convert_flag = True

        for shipyard in me.shipyards:

            if stepTo(shipyard.position, ship.position, size) <= 8: convert_flag = False

        if convert_flag: ship.next_action = ShipAction.CONVERT

    

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

                    #neighbors = [ship.cell.north.halite, ship.cell.east.halite, ship.cell.south.halite, ship.cell.west.halite]

                    neighbors = [exp_halite(board[ship.cell.north.position]), 

                                 exp_halite(board[ship.cell.east.position]), 

                                 exp_halite(board[ship.cell.south.position]), 

                                 exp_halite(board[ship.cell.west.position])]

                    ## avoid going back

                    if ship.id in ship_moves.keys():

                        if ship_moves[ship.id] == 0:

                            neighbors[2] = -999

                        elif ship_moves[ship.id] == 1:

                            neighbors[3] = -999

                        elif ship_moves[ship.id] == 2:

                            neighbors[0] = -999

                        else:

                            neighbors[1] = -999

                    ## avoid opponent's shipyard      

                    neighbor_positions = [ship.cell.north.position, ship.cell.east.position, ship.cell.south.position, ship.cell.west.position]

                    for i, point in enumerate(neighbor_positions):

                        if point in avoid_positions:

                            neighbors[i] = -999

                        

                    best = max(range(len(neighbors)), key=neighbors.__getitem__)

                    ship.next_action = directions[best]

                    ship_moves[ship.id] = best

                    

            if ship_states[ship.id] == "DEPOSIT":

                # Move towards the colest shipyard to deposit cargo

                cloest_steps = size

                closet_shipyard = me.shipyards[0]

                

                for shipyard in me.shipyards:

                    if stepTo(shipyard.position, ship.position, size)<cloest_steps:

                        closet_shipyard = shipyard

                        

                direction = getDirTo(ship.position, closet_shipyard.position, size)

                ship.next_action = direction

                    

    return me.next_actions
#from kaggle_environments import make

#env = make("halite", configuration={"size": 21}, debug=True)

#env.run(["submission.py", "random", "random", "random"])

#env.render(mode="ipython", width=800, height=600)
#from kaggle_environments import make

#from kaggle_environments.envs.halite.helpers import *



# Create a test environment for use later

#board_size = 5

#environment = make("halite", configuration={"size": board_size, "startingHalite": 1000})

#agent_count = 2

#environment.reset(agent_count)

#state = environment.state[0]



#board = Board(state.observation, environment.configuration)

#print(board)