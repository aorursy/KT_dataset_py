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
from kaggle_environments import make

env = make("halite", debug=True)

env.run(["submission.py", "random", "random", "random"])

env.render(mode="ipython", width=800, height=600)