import pandas as pd



def determine_directions(point1, point2, size=21):

    """ Given two points determine the closest directions to take to get to the point. """

    x1, y1 = point1.x, point1.y

    x2, y2 = point2.x, point2.y



    # For both x and y they are two type of paths to take

    diff_x_1 = abs(x2 - x1)

    diff_x_2 = abs(size - x2 + x1)

    diff_y_1 = abs(y2 - y1)

    diff_y_2 = abs(size - y2 + y1)

    # Given that x1=x2 or y1=y2 then return None, so we know

    # no movement was needed in that axis.

    best_x, best_y = 'None', 'None'



    if diff_x_1 > diff_x_2:

        if x2 - x1 > 0:

            best_x = "W"

        elif x2 - x1 < 0:

            best_x = "E"

    else:

        if x2 - x1 > 0:

            best_x = "E"

        elif x2 - x1 < 0:

            best_x = "W"



    if diff_y_1 > diff_y_2:

        if y2 - y1 > 0:

            best_y = "S"

        elif y2 - y1 < 0:

            best_y = "N"

    else:

        if y2 - y1 > 0:

            best_y = "N"

        elif y2 - y1 < 0:

            best_y = "S"



    return best_x, best_y





def grid(cell):

    """ Returns a dictionary of cells which are in 10 moves distance of the given cell """

    # The directions that are one move away

    north, south, west, east = cell.north, cell.south, cell.west, cell.east

    # The directions that are two moves away

    n2, s2, w2, e2 = north.north, south.south, west.west, east.east

    n3, s3, w3, e3 = n2.north, s2.south, w2.west, e2.east

    n4, s4, w4, e4 = n3.north, s3.south, w3.west, e3.east

    n5, s5, w5, e5 = n4.north, s4.south, w4.west, e4.east

    n6, s6, w6, e6 = n5.north, s5.south, w5.west, e5.east

    n7, s7, w7, e7 = n6.north, s6.south, w6.west, e6.east

    n8, s8, w8, e8 = n7.north, s7.south, w7.west, e7.east

    n9, s9, w9, e9 = n8.north, s8.south, w8.west, e8.east

    n10, s10, w10, e10 = n9.north, s9.south, w9.west, e9.east



    return {

        # 1 move away

        'N': north, 'S': south, 'W': west, 'E': east,

        # 2 moves away

        'NW': north.west, 'NE': north.east, 'SW': south.west, 'SE': south.east, 'WW': w2, 'EE': e2, 'NN': n2, 'SS': s2,

        # 3 moves away

        'SSS': s3, 'EEE': e3, 'WWW': w3, 'NNN': n3,

        'ENN': n2.east, 'WNN': n2.west, 'ESS': s2.east, 'WSS': s2.west,

        'SEE': e2.south, 'NEE': e2.north, 'SWW': w2.south, 'NWW': w2.north,

        # 4 moves away

        'NNNN': n4, 'SSSS': s4, 'WWWW': w4, 'EEEE': e4,

        'EESS': e2.south.south, 'EENN': e2.north.north, 'WWNN': w2.north.north, 'WWSS': w2.south.south,

        'WWWS': w3.south, 'EEES': e3.south, 'EEEN': e3.north, 'WWWN': w3.north,

        'SSSW': s3.west, 'SSSE': s3.east, 'NNNE': n3.east, 'NNNW': n3.west,

        # 5 moves away

        'SSSSS': s5, 'NNNNN': n5, 'WWWWW': w5, 'EEEEE': e5,

        'WWWWN': w4.north, 'WWWWS': w4.south, 'EEEEN': e4.north, 'EEEES': e4.south,

        'SSSSE': s4.east, 'SSSSW': s4.west, 'NNNNW': n4.west, 'NNNNE': n4.east,

        'EESSS': s3.east.east, 'WWSSS': s3.west.west, 'EENNN': n3.east.east, 'WWNNN': n3.west.west,

        'EEESS': e3.south.south, 'EEENN': e3.north.north, 'WWWSS': w3.south.south, 'WWWNN': w3.north.north

    }
class Locator:

    """ This module returns dataframes that could be used to analyze the board much faster """



    def __init__(self, board, ship):

        self.board = board

        self.ship = ship

        self.ship_position = ship.position

        # Get the grid

        self.grid = grid(ship.cell)



    def get_ship_info(self):

        """ Returns the info about ships in all of the board. """

        ships_info = {}



        for ship_id, ship in self.board.ships.items():

            base_info = {"my_ship": 0, "moves": 0, "position": (ship.position.x, ship.position.y),

                         'cargo': ship.cell.halite,

                         'dirX': (determine_directions(self.ship_position, ship.position))[0],

                         'dirY': (determine_directions(self.ship_position, ship.position))[1],

                         'movesX': min(abs(self.ship_position.x - ship.position.x),

                                       abs(21 - self.ship_position.x + ship.position.x)),

                         'movesY': min(abs(self.ship_position.y - ship.position.y),

                                       abs(21 - self.ship_position.y + ship.position.y))}



            base_info['moves'] = base_info['movesX'] + base_info['movesY']



            if ship_id in self.board.current_player.ship_ids and ship.id != self.ship.id:

                base_info['my_ship'] = 1

                ships_info[ship_id] = base_info

            elif not (ship_id in self.board.current_player.ship_ids):

                ships_info[ship_id] = base_info



        return pd.DataFrame(ships_info)



    def get_shipyard_info(self):

        """ Returns the info about shipyards in all of the board. """

        shipyards_info = {}



        for shipyard_id, shipyard in self.board.shipyards.items():

            base_info = {"my_shipyard": 0, "position": (shipyard.position.x, shipyard.position.y),

                         'dirX': (determine_directions(self.ship_position, shipyard.position))[0],

                         'dirY': (determine_directions(self.ship_position, shipyard.position))[1],

                         'player_halite': shipyard.player.halite,

                         'movesX': min(abs(self.ship_position.x - shipyard.position.x),

                                       abs(21 - self.ship_position.x + shipyard.position.x)),

                         'movesY': min(abs(self.ship_position.y - shipyard.position.y),

                                       abs(21 - self.ship_position.y + shipyard.position.y))}



            base_info['moves'] = base_info['movesX'] + base_info['movesY']



            if shipyard_id in self.board.current_player.shipyard_ids:

                base_info['my_shipyard'] = 1

                shipyards_info[shipyard_id] = base_info

            else:

                shipyards_info[shipyard_id] = base_info



        return pd.DataFrame(shipyards_info)



    def generate_grid_df(self):

        """ Generates a Dataframe describing the information of objects and cells in the grid of the ship. """

        all_dirs = {}



        for direction, cell in self.grid.items():



            base_info = {

                "ship_id": None, "shipyard_id": None,

                "my_ship": 0, "my_shipyard": 0,

                "halite": 0, "moves": 0,

                "movesX": 0, "movesY": 0,

                "dirY": 'None', "dirX": 'None',

                'weightX': 0, 'weightY': 0

            }



            if "N" in direction:

                base_info['dirY'] = 'N'

                base_info['movesY'] = direction.count("N")

            elif "S" in direction:

                base_info['dirY'] = 'S'

                base_info['movesY'] = direction.count("S")



            if "W" in direction:

                base_info['dirX'] = 'W'

                base_info['movesX'] = direction.count("W")

            elif "E" in direction:

                base_info['dirX'] = 'E'

                base_info['movesX'] = direction.count("E")



            if base_info['dirY'] != 'None':

                base_info['weightY'] = 1 / (len(direction) ** 2 * base_info['movesY']) 



            if base_info['dirX'] != 'None':

                base_info['weightX'] = 1 / (len(direction) ** 2 * base_info['movesX'])



            if cell.ship is not None:

                base_info["ship_id"] = cell.ship.id

                if cell.ship.id in self.ship.player.ship_ids:

                    base_info["my_ship"] = 1



            if cell.shipyard is not None:

                base_info["shipyard_id"] = cell.shipyard.id

                if cell.shipyard.id in self.ship.player.shipyard_ids:

                    base_info['my_shipyard'] = 1



            base_info['halite'] = cell.halite

            # The number of letters in the direction would indicate the number of moves needed to get there

            base_info['moves'] = len(direction)



            all_dirs[direction] = base_info



        return pd.DataFrame(all_dirs)

from kaggle_environments import make

from kaggle_environments.envs.halite.helpers import *



# Create a test environment for use later

board_size = 21

environment = make("halite", configuration={"size": board_size, "startingHalite": 30000}, debug=True)

agent_count = 4

environment.reset(agent_count)

state = environment.state[0]
# Change the behviour of my own movement from board.current_player

board = Board(state.observation, environment.configuration)



board.current_player.ships[0].next_action = ShipAction.CONVERT

board.ships['0-2'].next_action = ShipAction.CONVERT

board.ships['0-3'].next_action = ShipAction.CONVERT

board.ships['0-4'].next_action = ShipAction.CONVERT

board = board.next()



board.current_player.shipyards[0].next_action = ShipyardAction.SPAWN

board.shipyards['1-2'].next_action = ShipyardAction.SPAWN

board.shipyards['1-3'].next_action = ShipyardAction.SPAWN

board.shipyards['1-4'].next_action = ShipyardAction.SPAWN

board = board.next()



# Current-Player

board.current_player.ships[0].next_action  = ShipAction.EAST

board.current_player.shipyards[0].next_action = ShipyardAction.SPAWN

# Opponents

board.ships['2-2'].next_action = ShipAction.NORTH

board.shipyards['1-2'].next_action = ShipyardAction.SPAWN

board.ships['2-3'].next_action = ShipAction.WEST

board.shipyards['1-3'].next_action = ShipyardAction.SPAWN

board.ships['2-4'].next_action = ShipAction.EAST

board.shipyards['1-4'].next_action = ShipyardAction.SPAWN

board = board.next()



# Current-Player

board.current_player.ships[0].next_action  = ShipAction.NORTH

board.current_player.ships[1].next_action  = ShipAction.SOUTH

board.current_player.shipyards[0].next_action = ShipyardAction.SPAWN

# Opponents

board.ships['2-2'].next_action = ShipAction.WEST

board.shipyards['1-2'].next_action = ShipyardAction.SPAWN

board.ships['2-3'].next_action = ShipAction.NORTH

board.shipyards['1-3'].next_action = ShipyardAction.SPAWN

board.ships['2-4'].next_action = ShipAction.SOUTH

board.shipyards['1-4'].next_action = ShipyardAction.SPAWN

board = board.next()



# Current-Player

board.current_player.ships[0].next_action  = ShipAction.EAST

board.current_player.ships[1].next_action  = ShipAction.SOUTH

board.current_player.ships[2].next_action  = ShipAction.EAST

# Opponents

board.ships['2-2'].next_action = ShipAction.WEST

board.ships['2-3'].next_action = ShipAction.EAST

board.ships['2-4'].next_action = ShipAction.SOUTH

board = board.next()



# Current-Player

board.current_player.ships[0].next_action  = ShipAction.EAST

board.current_player.ships[1].next_action  = ShipAction.SOUTH

board.current_player.ships[2].next_action  = ShipAction.EAST

# Opponents

board.ships['2-2'].next_action = ShipAction.WEST

board.ships['2-3'].next_action = ShipAction.EAST

board.ships['2-4'].next_action = ShipAction.SOUTH

board = board.next()



# Current-Player

board.current_player.ships[0].next_action  = ShipAction.EAST

board.current_player.ships[1].next_action  = ShipAction.SOUTH

board.current_player.ships[2].next_action  = ShipAction.EAST

# Opponents

board.ships['2-2'].next_action = ShipAction.WEST

board.ships['2-3'].next_action = ShipAction.EAST

board.ships['2-4'].next_action = ShipAction.SOUTH

board = board.next()



# Current-Player

board.current_player.ships[0].next_action  = ShipAction.EAST

board.current_player.ships[1].next_action  = ShipAction.SOUTH

board.current_player.ships[2].next_action  = ShipAction.EAST

# Opponents

board.ships['2-2'].next_action = ShipAction.NORTH

board.ships['2-3'].next_action = ShipAction.EAST

board.ships['2-4'].next_action = ShipAction.SOUTH

board = board.next()



board.current_player.ships[0].next_action = ShipAction.CONVERT

board = board.next()



ship = board.current_player.ships[0]
grid = Locator(board, ship).generate_grid_df().T

# Checking to see if the directions are implemented properly

grid.iloc[30:70, :]
%%writefile /kaggle/working/current.py



from kaggle_environments.envs.halite.helpers import Board, ShipAction, ShipyardAction

import pandas as pd

import numpy as np





class DecisionShip:

    """ 

        Decides ship's next move

        params:

            board: the board that we will base our decisions on

            ship: the ship we are deciding for

            step: the steps into the stimulation

        returns:

            determine: returns the next-action that should be taken

    """

    def __init__(self, board: Board, ship_id, step):

        self.board = board

        self.ship = board.ships[ship_id]

        self.step = step



        # Some usefull properties

        self.player = self.board.current_player

        self.ship_cargo = self.ship.halite

        self.current_cell = self.ship.cell

        self.current_position = self.ship.position

        

        # All moves ship can take

        self.moves = {"N": ShipAction.NORTH, 'S': ShipAction.SOUTH, 'W': ShipAction.WEST,

                      'E': ShipAction.EAST, 'convert': ShipAction.CONVERT, 'mine': None}



        # The list of moves that should not be taken

        self.eliminated_moves = ['None']



        # Weights of different moves

        self.weights = {"N": 0, "E": 0, "W": 0, "S": 0, "mine": 0, "convert": 0, 'None': 0}



        # The object's relative situation to other ship/shipyards

        self.locator = Locator(board, self.ship)

        self.Ships = self.locator.get_ship_info()

        self.Shipyards = self.locator.get_shipyard_info()

        self.grid = self.locator.generate_grid_df()



        # Closest shipyard id

        self.closest_shipyard_id = self.closest_shipyard()

        # Gets the distance to the closest shipyard

        if self.closest_shipyard_id != 0:

            self.closest_shipyard_distance = self.Shipyards[self.closest_shipyard_id]['moves']

        else:

            self.closest_shipyard_distance = 1



        # Default move which is set to mining (None)

        self.next_move = None



        # This variable holds the direction that is being evaluated

        self.current_direction = ""



    def determine(self):

        """ Returns next action decided for the ship based on the observations that have been made. """

        self.weight_moves()  # Calculate the weights for main four directions

        self.round()  # Round the weights

        self.apply_elimination()  # Apply the eliminations



        # Sort the values

        sorted_weights = {k: v for k, v in sorted(self.weights.items(), key=lambda item: item[1], reverse=True)}



        # Choose the action with highest value given that it is not eliminated

        for action in sorted_weights.keys():

            return self.moves[action], action



        # If none were chosen, then just return the default move which is mining

        return self.next_move, 'mine'



    def add_accordingly(self, value, title="", loging=False):

        """ Adds a value to directions according to their corresponding weights """

        dirX, dirY = self.grid[self.current_direction]['dirX'], self.grid[self.current_direction]['dirY']

        weightX, weightY = self.grid[self.current_direction]['weightX'], self.grid[self.current_direction]['weightY']



        if weightX != 0: self.weights[dirX] += value * weightX

        if weightY != 0: self.weights[dirY] += value * weightY

    # Add the relation with the amount of halite on the cell

    def weight_convert(self, base_threshold=600):

        """ Weights the option for ship conversion. """

        # Calculating the threshhold

        threshold = base_threshold + 1000 * (len(self.player.shipyards) // 3)

        

        # 1. If they are no shipyards left

        no_shipyards = len(self.player.shipyards) == 0

        # 2. There will be a threshold for the amount of cargo any ship could have

        threshhold_reach = self.ship.halite > threshold

        # 3. On shipyard already

        on_shipyard = self.ship.cell.shipyard is not None



        if self.player.halite > 500:

            if no_shipyards and not on_shipyard:

                self.weights['convert'] = 1e4

            elif self.Shipyards[self.closest_shipyard_id]['moves'] < 10 and not on_shipyard:

                self.weights['convert'] = (self.ship_cargo - threshold) * 60

            else:

                self.eliminated_moves.append('convert')

        else:

            self.eliminated_moves.append('convert')



    def weight_moves(self):

        """

            This is the main function and runs other helper functions within the module to to weight the different moves that could be taken.



            The weightings will take numerous considerations into account (halite, closeness to shipyard, etc.) yet there are going to be 

            some values manual multiplications in order to encourage or discourage certain moves. In order to keep things simple, the values 

            will be multiplied by a number between -10 to 10.



            Now some cells are closer to the current cell and in order to put an emphasis on that, we will take into  account the fact that 

            distance would have a indirect correlation in our weighting process.



            When going through the different directions on the ship's grid, they are going to be some scenarios that we should  decide for them accordingly:

                1. If there is one of my ships: should avoid collision

                2. If there is an enemy shipyard: depending on variables, could either runaway or attack it

                3. If there is one of my shipyards: possibly deposit or ignore, also check for protection

                4. If there is an enemy shipyard: depending on the situation might attack it

        """

        # Weight the CONVERT option

        self.weight_convert()



        # See if any of the shipyards need defending

        self.shipyard_status()



        # Iterate through different directions

        for direction in list(self.grid.columns):

            # Set the global direction to the one at hand

            self.current_direction = direction



            # Get the ids just for the ease of use

            Ship_id = self.grid[direction]['ship_id']

            Shipyard_id = self.grid[direction]['shipyard_id']





            # 1. Evaluate the moves based on other objects present in the map

            # 1.1 If there was a ship

            if not pd.isna(Ship_id):

                # If it was my ship

                if self.grid[direction].my_ship == 1:

                    if self.Ships[Ship_id]['moves'] == 1:

                        self.eliminated_moves.append(direction)

                    else:

                        self.distribute_ships(Ship_id)

                else:

                    if self.Ships[Ship_id]['moves'] == 1 and self.Ships[Ship_id]['cargo'] < self.ship_cargo:

                        self.eliminated_moves.append(direction)

                        self.eliminated_moves.append('mine')

                        # Go to the closest shipyard preferably

                        self.go_to_closest_shipyard(self.ship_cargo ** 3)

                    else:

                        self.deal_enemy_ship(Ship_id)



            # 1.2 If there was a shipyard

            if not pd.isna(Shipyard_id):

                if self.grid[direction].my_shipyard == 1:

                    self.deposit()

                else:

                    self.attack_enemy_shipyard(Shipyard_id)



            # 2. Trigger movement in the main four direction solely based on the amount of halite each cell has

            main_dir_encourage = 8 * self.grid[direction].halite ** 1.2

            self.add_accordingly(main_dir_encourage, title='  main4: ', loging=False)



            # 3. Either encourage mining or discourage it by adding the difference between cells to the mine

            mining_trigger = 5 * (self.current_cell.halite - self.grid[direction].halite + 10) / self.grid[direction].moves



            self.weights['mine'] += mining_trigger



        # The correlation of the mining with cell's halite

        self.weights['mine'] += self.current_cell.halite ** 2.5



    def distribute_ships(self, ship_id):

        """ This function lowers the ships tendency to densely populate an area """

        distribution_encouragement = -10 * abs(self.ship_cargo - self.Ships[ship_id].cargo) / self.Ships[ship_id].moves



        self.add_accordingly(distribution_encouragement, title='Distribution', loging=False)



    def deal_enemy_ship(self, ship_id):

        """ This function will evaluate to either attack or get_away from an enemy ship based on the 

        simple observation: If my ship had more cargo then I should not attack. """

        # If the ship's cargo was more than enemy's cargo and it was not equal to zero then get away otherwise attack

        if self.ship_cargo > self.Ships[ship_id].cargo + 0.4 * self.grid[self.current_direction]['halite'] and self.ship_cargo != 0:

            self.get_away(cargo_diff=abs(self.Ships[ship_id].cargo - self.ship_cargo))

        else:

            self.attack_enemy_ship(self.Ships[ship_id].cargo - self.ship_cargo)



    def attack_enemy_ship(self, diff):

        """ This function encourages attacking the enemy ship """

        attack_encouragement = 10 * (diff + 1) ** 2 / (self.closest_shipyard_distance + 0.1)

        self.add_accordingly(attack_encouragement, title='Attacking-Enemy-Ship', loging=False)



    def get_away(self, cargo_diff=0):

        """ This function is called when my ship needs to get away from a ship which might be following it """

        # 1. Directly discouraging the movement

        if len(self.current_direction) == 2:

            # When the enemy ship is two moves away, there should be a strong discouragement

            direction_discouragement = -10 * (self.ship.halite + 10) ** 5

        else:

            direction_discouragement = -10 * cargo_diff

        self.add_accordingly(direction_discouragement, title='Get-Away', loging=False)



        # 2. Encouraging going to the closest shipyard

        closest_shipyard_encouragement = cargo_diff / self.grid[self.current_direction]['moves']

        self.go_to_closest_shipyard(closest_shipyard_encouragement)



    def deposit(self):

        """ Weights the tendency to deposit and adds to the directions which lead to the given shipyard """

        if self.near_end():

            deposit_tendency = self.ship_cargo ** 4

        else:

            deposit_tendency = 10 * self.ship_cargo ** (1.2 + 2 * self.step // 390)

        self.add_accordingly(deposit_tendency, title='Deposit', loging=False)



    def attack_enemy_shipyard(self, shipyard_id):

        """ Weights the tendency to attack the enemy shipyard. """

        if len(self.player.ships) >= 2 and self.player.halite > 700 and self.ship_cargo < 30 and self.closest_shipyard_distance < 5:

            destory_shipyard = 1e4 / len(self.current_direction)

            self.add_accordingly(destory_shipyard, title='Destroy_en_shipyard', loging=False)

        elif len(self.current_direction) == 1 and self.ship_cargo > 100:

            self.eliminated_moves.append(self.current_direction)



    def go_to_closest_shipyard(self, value):

        """ Encourage movement towards the nearest shipyard """

        if self.closest_shipyard_id != 0:

            dirX, dirY = self.Shipyards[self.closest_shipyard_id]['dirX'], self.Shipyards[self.closest_shipyard_id]['dirY']

            

            self.weights[dirX] += value

            self.weights[dirY] += value



    def shipyard_status(self):

        """ Measures tendency for the shipyards within the map """

        if not self.Shipyards.empty:

            for shipyard in self.player.shipyards:

                self.analyze_shipyard_surroundings(shipyard.id)



    def analyze_shipyard_surroundings(self, shipyard_id):

        """ Checks to see if a given shipyard needs protection or not? """

        shipyard = self.board.shipyards[shipyard_id]

        dirX, dirY = self.Shipyards[shipyard_id]['dirX'], self.Shipyards[shipyard_id]['dirY']

        value = 0

        shipyard_grid = Locator(self.board, shipyard).generate_grid_df()



        for direction in list(shipyard_grid.columns):

            ship_id = shipyard_grid[direction].ship_id

            # If there is a ship on that cell

            if not pd.isna(ship_id):

                if shipyard_grid[direction].my_ship == 1:

                    value += -1e4 / shipyard_grid[direction]['moves'] ** 2

                else:

                    value += 1e4 / shipyard_grid[direction]['moves'] ** 2



        if value > 0:

            # More concenteration on the closest shipyard

            if shipyard_id == self.closest_shipyard_id: value *= 2

            

            self.weights[dirX] += value

            self.weights[dirY] += value



            # log('  Shipyard status: ' + str(shipyard.position) + ', adding: ' + str(round(value, 3)) + ' to ' + dirX + ', ' + dirY)

        

    def closest_shipyard(self):

        """ Returns the closest shipyard's id. """

        shipyard_id = 0  # The default value would be zero meaning that they either no shipyard or I did not have any

        # First we should check to see if there are any Shipyards at all

        if not self.Shipyards.empty:

            # Then we should check to see if I have any shipyards

            if not self.Shipyards.T[self.Shipyards.T['my_shipyard'] == 1].empty:

                shipyard_id = self.Shipyards.T[self.Shipyards.T['my_shipyard'] == 1]['moves'].astype('int').idxmin()



        return shipyard_id



    def near_end(self):

        """ Determines if the game is about to end so the ships with halite can convert to shipyard and maximum the halite we will end up with. """

        count = 0



        # If the halite was less than 500 and it had no ships

        for opp in self.board.opponents:

            if opp.halite < 500 and len(opp.ships) == 0 and self.player.halite > opp.halite: count += 1

            if opp.halite > 2000 and len(opp.ships) > 1: count -= 1



        # If count was more than 2 return True

        return count >= 2



    def apply_elimination(self):

        """ Eliminates the moves to be eliminated. """

        for move in self.eliminated_moves:

            if move in self.weights.keys():

                del self.weights[move]



    def round(self):

        """ This functions rounds the  weights so they can be easily printed """

        self.weights['mine'] = round(self.weights['mine'], 1)

        self.weights['N'] = round(self.weights['N'], 1)

        self.weights['S'] = round(self.weights['S'], 1)

        self.weights['E'] = round(self.weights['E'], 1)

        self.weights['W'] = round(self.weights['W'], 1)

        self.weights['convert'] = round(self.weights['convert'], 1)





class ShipyardDecisions:

    def __init__(self, board: Board, player, step):

        """

            Decides the Shipyard's next action based on the given parameters

            board: The board that we will be observing

            step: step of the stimulation

        """

        self.board = board

        self.player_halite = player.halite

        self.step = step

        self.Shipyards = player.shipyards

        self.shipyard_tendencies = {}



    def determine(self):

        """ Determines which shipyards should SPAWN, returns a dictionary of id: 'SPAWN' """

        self.weight_shipyard_tendencies()

        sorted_weights = {k: v for k, v in

                          sorted(self.shipyard_tendencies.items(), key=lambda item: item[1], reverse=True)}

        

        shipyard_ids = []

        for shipyard_id, tendency in sorted_weights.items():

            if tendency > 0 and self.player_halite >= 500:

                shipyard_ids.append(shipyard_id)



        return shipyard_ids



    def weight_shipyard_tendencies(self):

        """ Iterates through the shipyards and weights their tendencies. """

        for shipyard in self.Shipyards:

            # Weighting will take place only when there are no ships on the cell

            if shipyard.cell.ship is None:

                grid = Locator(self.board, shipyard).generate_grid_df()

                weight = self.weight(grid)



                self.shipyard_tendencies[shipyard.id] = weight



    def weight(self, grid):

        """

            Weights shipyard's tendency to spawn solely based on the objects around it

            The weighting system is rather simple:

                - If there was an enemy ship add to the weight

                - If there was one of my own ships, then subtract from the weight

            Take the distance of the ship into account

        """

        if len(self.board.current_player.ships) == 0:

            return 10

        if self.step < 70 and self.player_halite > 500 + 500 * (self.step // 50):

            return 10



        value = 0

        # Iterating through the grid

        for direction in grid.columns:

            if not pd.isna(grid[direction].ship_id):

                if grid[direction].my_ship == 1:

                    value -= 100 / grid[direction]['moves']

                else:

                    value += 100 / grid[direction]['moves']

                    # If there was an enemy ship one move away from my shipyard then spawn

                    if grid[direction]['moves'] == 1 and self.player_halite > 500: 

                        value += 1e3 



            if not pd.isna(grid[direction].shipyard_id):

                if grid[direction].my_shipyard == 0:

                    value += 200 / grid[direction]['moves']



        return round(value, 2)





class Locator:

    """ This module returns dataframes that could be used to analyze the board much faster """



    def __init__(self, board, ship):

        self.board = board

        self.ship = ship

        self.ship_position = ship.position

        # Get the grid

        self.grid = grid(ship.cell)



    def get_ship_info(self):

        """ Returns the info about ships in all of the board. """

        ships_info = {}



        for ship_id, ship in self.board.ships.items():

            base_info = {"my_ship": 0, "moves": 0, "position": (ship.position.x, ship.position.y),

                         'cargo': ship.cell.halite,

                         'dirX': (determine_directions(self.ship_position, ship.position))[0],

                         'dirY': (determine_directions(self.ship_position, ship.position))[1],

                         'movesX': min(abs(self.ship_position.x - ship.position.x),

                                       abs(21 - self.ship_position.x + ship.position.x)),

                         'movesY': min(abs(self.ship_position.y - ship.position.y),

                                       abs(21 - self.ship_position.y + ship.position.y))}



            base_info['moves'] = base_info['movesX'] + base_info['movesY']



            if ship_id in self.board.current_player.ship_ids and ship.id != self.ship.id:

                base_info['my_ship'] = 1

                ships_info[ship_id] = base_info

            elif not (ship_id in self.board.current_player.ship_ids):

                ships_info[ship_id] = base_info



        return pd.DataFrame(ships_info)



    def get_shipyard_info(self):

        """ Returns the info about shipyards in all of the board. """

        shipyards_info = {}



        for shipyard_id, shipyard in self.board.shipyards.items():

            base_info = {"my_shipyard": 0, "position": (shipyard.position.x, shipyard.position.y),

                         'dirX': (determine_directions(self.ship_position, shipyard.position))[0],

                         'dirY': (determine_directions(self.ship_position, shipyard.position))[1],

                         'player_halite': shipyard.player.halite,

                         'movesX': min(abs(self.ship_position.x - shipyard.position.x),

                                       abs(21 - self.ship_position.x + shipyard.position.x)),

                         'movesY': min(abs(self.ship_position.y - shipyard.position.y),

                                       abs(21 - self.ship_position.y + shipyard.position.y))}



            base_info['moves'] = base_info['movesX'] + base_info['movesY']



            if shipyard_id in self.board.current_player.shipyard_ids:

                base_info['my_shipyard'] = 1

                shipyards_info[shipyard_id] = base_info

            else:

                shipyards_info[shipyard_id] = base_info



        return pd.DataFrame(shipyards_info)



    def generate_grid_df(self):

        """ Generates a Dataframe describing the information of objects and cells in the grid of the ship. """

        all_dirs = {}



        for direction, cell in self.grid.items():



            base_info = {

                "ship_id": None, "shipyard_id": None,

                "my_ship": 0, "my_shipyard": 0,

                "halite": 0, "moves": 0,

                "movesX": 0, "movesY": 0,

                "dirY": 'None', "dirX": 'None',

                'weightX': 0, 'weightY': 0

            }



            if "N" in direction:

                base_info['dirY'] = 'N'

                base_info['movesY'] = direction.count("N")

            elif "S" in direction:

                base_info['dirY'] = 'S'

                base_info['movesY'] = direction.count("S")



            if "W" in direction:

                base_info['dirX'] = 'W'

                base_info['movesX'] = direction.count("W")

            elif "E" in direction:

                base_info['dirX'] = 'E'

                base_info['movesX'] = direction.count("E")



            if base_info['dirY'] != 'None':

                base_info['weightY'] = 1 / (len(direction) ** 2 * base_info['movesY']) 



            if base_info['dirX'] != 'None':

                base_info['weightX'] = 1 / (len(direction) ** 2 * base_info['movesX']) 



            if cell.ship is not None:

                base_info["ship_id"] = cell.ship.id

                if cell.ship.id in self.ship.player.ship_ids:

                    base_info["my_ship"] = 1



            if cell.shipyard is not None:

                base_info["shipyard_id"] = cell.shipyard.id

                if cell.shipyard.id in self.ship.player.shipyard_ids:

                    base_info['my_shipyard'] = 1



            base_info['halite'] = cell.halite

            # The number of letters in the direction would indicate the number of moves needed to get there

            base_info['moves'] = len(direction)



            all_dirs[direction] = base_info



        return pd.DataFrame(all_dirs)





####################

# Helper Functions #

####################

def determine_directions(point1, point2, size=21):

    """ Given two points determine the closest directions to take to get to the point. """

    x1, y1 = point1.x, point1.y

    x2, y2 = point2.x, point2.y



    # For both x and y they are two type of paths to take

    diff_x_1 = abs(x2 - x1)

    diff_x_2 = abs(size - x2 + x1)

    diff_y_1 = abs(y2 - y1)

    diff_y_2 = abs(size - y2 + y1)

    # Given that x1=x2 or y1=y2 then return None, so we know

    # no movement was needed in that axis.

    best_x, best_y = 'None', 'None'



    if diff_x_1 > diff_x_2:

        if x2 - x1 > 0:

            best_x = "W"

        elif x2 - x1 < 0:

            best_x = "E"

    else:

        if x2 - x1 > 0:

            best_x = "E"

        elif x2 - x1 < 0:

            best_x = "W"



    if diff_y_1 > diff_y_2:

        if y2 - y1 > 0:

            best_y = "S"

        elif y2 - y1 < 0:

            best_y = "N"

    else:

        if y2 - y1 > 0:

            best_y = "N"

        elif y2 - y1 < 0:

            best_y = "S"



    return best_x, best_y





def grid(cell):

    """ Returns a dictionary of cells which are in 10 moves distance of the given cell """

    # The directions that are one move away

    north, south, west, east = cell.north, cell.south, cell.west, cell.east

    # The directions that are two moves away

    n2, s2, w2, e2 = north.north, south.south, west.west, east.east

    n3, s3, w3, e3 = n2.north, s2.south, w2.west, e2.east

    n4, s4, w4, e4 = n3.north, s3.south, w3.west, e3.east

    n5, s5, w5, e5 = n4.north, s4.south, w4.west, e4.east



    return {

        # 1 move away

        'N': north, 'S': south, 'W': west, 'E': east,

        # 2 moves away

        'NW': north.west, 'NE': north.east, 'SW': south.west, 'SE': south.east, 'WW': w2, 'EE': e2, 'NN': n2, 'SS': s2,

        # 3 moves away

        'SSS': s3, 'EEE': e3, 'WWW': w3, 'NNN': n3,

        'ENN': n2.east, 'WNN': n2.west, 'ESS': s2.east, 'WSS': s2.west,

        'SEE': e2.south, 'NEE': e2.north, 'SWW': w2.south, 'NWW': w2.north,

        # 4 moves away

        'NNNN': n4, 'SSSS': s4, 'WWWW': w4, 'EEEE': e4,

        'EESS': e2.south.south, 'EENN': e2.north.north, 'WWNN': w2.north.north, 'WWSS': w2.south.south,

        'WWWS': w3.south, 'EEES': e3.south, 'EEEN': e3.north, 'WWWN': w3.north,

        'SSSW': s3.west, 'SSSE': s3.east, 'NNNE': n3.east, 'NNNW': n3.west,

        # # 5 moves away

        'SSSSS': s5, 'NNNNN': n5, 'WWWWW': w5, 'EEEEE': e5,

        'WWWWN': w4.north, 'WWWWS': w4.south, 'EEEEN': e4.north, 'EEEES': e4.south,

        'SSSSE': s4.east, 'SSSSW': s4.west, 'NNNNW': n4.west, 'NNNNE': n4.east,

        'EESSS': s3.east.east, 'WWSSS': s3.west.west, 'EENNN': n3.east.east, 'WWNNN': n3.west.west,

        'EEESS': e3.south.south, 'EEENN': e3.north.north, 'WWWSS': w3.south.south, 'WWWNN': w3.north.north

    }



movement_dictionary = {"S": "SOUTH", 'N': 'NORTH', 'W': 'WEST', 'E': 'EAST', 'convert': 'CONVERT'}



import operator

def agent(obs, config):

    # Another for updates

    board = Board(obs, config)



    # Step of the board

    step = board.observation['step']



    ships = [ship.id for ship in sorted(board.current_player.ships, key=operator.attrgetter("halite"), reverse=True)]

    actions = {}





    for ship_id in ships:

        if ship_id in board.current_player.ship_ids:

                

            next_action, action_type = DecisionShip(board, ship_id, step).determine()

                

            if action_type != 'mine':

                actions[ship_id] = movement_dictionary[action_type]

                board.ships[ship_id].next_action = next_action

                    

            board = board.next()



    shipyard_ids = ShipyardDecisions(board, board.current_player, step).determine()



    for shipyard_id in board.current_player.shipyard_ids:

        if shipyard_id in shipyard_ids:

            actions[shipyard_id] = 'SPAWN'

            board.shipyards[shipyard_id].next_action = ShipyardAction.SPAWN

            

            board = board.next()

        

    return actions

from kaggle_environments import make

from kaggle_environments.envs.halite.helpers import *



env = make("halite", debug=True)

env.run(['/kaggle/input/mockagents/agent_c.py', "/kaggle/working/current.py", '/kaggle/input/mockagents/agent_b.py', '/kaggle/input/mockagents/agent_a.py'])

env.render(mode="ipython", width=800, height=600)