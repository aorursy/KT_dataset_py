%%writefile submission.py



# Imports helper functions

from kaggle_environments.envs.halite.helpers import *



# Directions a ship can move

directions = [ShipAction.NORTH, ShipAction.EAST, ShipAction.SOUTH, ShipAction.WEST]



init_Controller = False

#controller = None
%%writefile -a submission.py



class Controller():

        

    def __init__(self, obs, config):

        #Initialise Instance Independent Constants

        self.size = config.size

        # Will keep track of whether a ship is collecting halite or carrying cargo to a shipyard

        self.ship_states = {}

        self.ships_in_last_turn = []

        

    def next_actions(self, obs, config):

        #Initialise Instance Specific Variables

        self.board = Board(obs, config)

        self.me = self.board.current_player

        self.steps = self.board.step

        self.ships = self.me.ships

        self.shipyards = self.me.shipyards  



        # If there are no ships, use first shipyard to spawn a ship.

        if len(self.ships) == 0 and len(self.shipyards) > 0:

            self.shipyards[0].next_action = ShipyardAction.SPAWN



        # If there are no shipyards, convert first ship into shipyard.

        if len(self.shipyards) == 0 and len(self.ships) > 0:

            self.ships[0].next_action = ShipAction.CONVERT



        for ship in self.ships:

            if ship.id not in self.ships_in_last_turn:

                print("New ship {} Added to {} in step : {}".format(ship.id,self.ships_in_last_turn,self.steps))

                self.ships_in_last_turn.append(ship.id)

                

            if ship.next_action == None:



                ### Part 1: Set the ship's state 

                if ship.halite < 200: # If cargo is too low, collect halite

                    self.ship_states[ship.id] = "COLLECT"

                if ship.halite > 500: # If cargo gets very big, deposit halite

                    self.ship_states[ship.id] = "DEPOSIT"



                ### Part 2: Use the ship's state to select an action

                if self.ship_states[ship.id] == "COLLECT":

                    # If halite at current location running low, 

                    # move to the adjacent square containing the most halite

                    if ship.cell.halite < 100:

                        neighbors = [ship.cell.north.halite, ship.cell.east.halite, 

                                     ship.cell.south.halite, ship.cell.west.halite]

                        best = max(range(len(neighbors)), key=neighbors.__getitem__)

                        ship.next_action = directions[best]

                if self.ship_states[ship.id] == "DEPOSIT":

                    # Move towards shipyard to deposit cargo

                    direction = self.getDirTo(ship.position, self.shipyards[0].position, self.size)

                    if direction: ship.next_action = direction



        return self.me.next_actions



    # Returns best direction to move from one position (fromPos) to another (toPos)

    # Example: If I'm at pos 0 and want to get to pos 55, which direction should I choose?

    def getDirTo(self, fromPos, toPos, size):

        fromX, fromY = divmod(fromPos[0],size), divmod(fromPos[1],size)

        toX, toY = divmod(toPos[0],size), divmod(toPos[1],size)

        if fromY < toY: return ShipAction.NORTH

        if fromY > toY: return ShipAction.SOUTH

        if fromX < toX: return ShipAction.EAST

        if fromX > toX: return ShipAction.WEST
%%writefile -a submission.py



# Returns the commands we send to our ships and shipyards

def agent(obs, config):

    global init_Controller

    global controller

    if not init_Controller:

        controller = Controller(obs, config) 

        init_Controller = True

    return controller.next_actions(obs, config)   
from kaggle_environments import make

env = make("halite", debug=True)

env.run(["submission.py", "random", "random", "random"])

env.render(mode="ipython", width=800, height=600)