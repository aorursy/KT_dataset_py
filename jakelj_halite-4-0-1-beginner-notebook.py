!pip install 'kaggle-environments>=0.2.1'



import kaggle_environments
from kaggle_environments import evaluate, make

import numpy as np

env = make("halite", debug=True)
%%writefile submission.py



import numpy as np

class gameInfo:

    '''Infomation about the state of the game'''

    def __init__(self, observation):

        #player id

        self.player = observation['player']

        # gives a list containing two player lists - which contain:

        # - player halite (int); dict('shipyardid': shipyardloc); dict('shipn': shipn_loc) 

        self.players = observation['players']

        # turn number

        self.step = observation['step']

        





class haliteBoard:

    ''' Functions for the board

    observation - dict with 3 keys

        player: 0 # player id

        step: turn in the game

        halite map - array of halite

    '''

    def __init__(self, observation):

         # creates a 1d array that matches the halite board for reference

        self.loc_board1d = np.array(list(range(225)))

        # creates a 2d array that matches the halite board for reference

        self.loc_board2d = np.array(list(range(225))).reshape(15,15)

        # the halite board - this should contain positions of all assets 1d

        self.halite_board1d = np.array(observation.halite)

         # the halite board - this should contain positions of all assets 2d

        self.halite_board2d = np.array(observation.halite).reshape(15,15)

        # map details

        self.width = 15

        self.height = 15

        

    def get_xy(self,posistion):

        ''' Takes a position e.g 101 and returns the x,y coordinates as a tuple''' 

        x,y = np.where(self.loc_board2d == posistion)

        coords = list(zip(x,y))

        return coords[0]

    

    # make function that takes coordinates and gets the index of the map

    def get_index(self, xy):

        '''takes tuple of (x, y) and returns position on board'''

        index = (xy[0] * 15) + xy[1] 

        return index

        

    

    

    

    def get_nearest_halite(self,posistion):

        '''finds the coordiantes for the nearest halite, 

        not accounting for wraparound returns (x,y)'''

        # where the ship is

        current_coords = self.get_xy(posistion)

        # where the location on the board has halite

        # this isn't working correctly

        hx, hy = np.where(self.halite_board2d > 100)

        #list of halite locations

        halite_coords = list(zip(hx,hy))

        distances = {}

        

        for i in halite_coords:

            # find euclidean distance, doesn't take into account wrap around

            dist = np.sqrt((i[0] - current_coords[0])**2 + (i[1] - current_coords[1])**2)

            distances[i] = dist

        # from the dict get the closest set of coords     

        closest_xy =  min(distances, key=distances.get)

        return closest_xy

    

   

    def get_surrounding_halite(self, posistion):

        '''returns a dict with halite in each direction accounting for wrap'''

        b = self.halite_board1d

        surrounding = {None:b[posistion],'NORTH':np.take(b,[posistion-15], mode = 'wrap'), 

                       'SOUTH':np.take(b,[posistion+15],mode = 'wrap'),

                       'EAST':np.take(b,[posistion+1],mode = 'wrap'), 

                       'WEST':np.take(b,[posistion-1],mode = 'wrap')}

        return surrounding



    def get_surrounding_loc(self,posistion):

        ''' returns the board location number of the surrounding locations 

        - N,S,E,W accounting for wrap '''

        tb = self.loc_board1d

        surrounding_locations = [np.take(tb,[posistion-15], mode = 'wrap'),

                                 np.take(tb,[posistion+15],mode = 'wrap'),

                                np.take(tb,[posistion+1],mode = 'wrap'),

                                np.take(tb,[posistion-1],mode = 'wrap')]



        return surrounding_locations

    

    def get_occupied_locs(self, posistion, shipyards, ships, opp_shipyards, opp_ships):

        ''' Returns all the occupied locations (number) on the map 

        excluding the current position'''

        

        # need to handle for when there are no ships or shipyards

        opp_ship_locs = [i[0] for i in list(opp_ships.values())]  # e.g [0, 34, 59]

        player_ship_locs = [i[0] for i in list(ships.values())]

        #logic ish

        if len(opp_shipyards.values()) == 0:

            return opp_ship_locs + player_ship_locs

           

        else:

            opp_shipyard_locs = [i for i in list(opp_shipyards.values())]

            return opp_ship_locs + player_ship_locs + opp_shipyard_locs

        

     # working well   

    def is_shipyard_occupied(self, ships, shipyards):

        ''' Returns true if there is a ship in our shipyard'''

        player_ship_locs  = [i[0] for i in list(ships.values())]

        if len(shipyards.values()) > 1:

            shipyards = [i[0] for i in list(shipyards.values())]

        elif len(shipyards.values()) < 1:

            return False

        else:

            #we have one shipyard

            shipyards =[i for i in list(shipyards.values())]

            for i in player_ship_locs:

                if i in shipyards:

                    return True

                else:

                    return False

                

    def get_safe_options_surrounding(self, posistion, ship_locations,surrounding):

        ''' Will take the ships position and the dict (ship_action) of moves 

        for greedy navigation and check if they are safe'''

        # get the number locations around the ship

        surrounding_locs = self.get_surrounding_loc(posistion)

        # I want to remove the occupied locations from the max halite dict

        surrounding_locs = [i[0] for i in surrounding_locs]

        nav_dict = {}

        nav_dict['NORTH'], nav_dict['SOUTH'], nav_dict['EAST'], nav_dict['WEST'] = surrounding_locs

        # now simply remove the keys where their values are in ship_locations

        for k,v in list(nav_dict.items()):

            if v in ship_locations:

                surrounding.pop(k)

        return surrounding   

        



def get_moves_to_target(ship_loc,ship_locations, board, target):

    '''Takes ship location and target and returns a list of 1 or more viable moves'''

    posistion_xy = board.get_xy(ship_loc)

    # takes int

    target_xy = board.get_xy(target)



    move_dict = {}

    if posistion_xy[0] > target_xy[0]:

        move_dict['NORTH'] = ship_loc + 15

    if posistion_xy[0] < target_xy[0]:

        move_dict['SOUTH'] = ship_loc - 15

    if posistion_xy[1] < target_xy[1]:

         move_dict['EAST'] = ship_loc + 1

    if posistion_xy[1] > target_xy[1]:

         move_dict['WEST'] = ship_loc - 1

    for k,v in list(move_dict.items()):

            if v in ship_locations:

                move_dict.pop(k)

    return move_dict







def greedy_collect(ship_loc, ship_locations, board):

    ''' Takes a ship location and the halite board returns a valid move'''

    # needs to get the greedy move with a bias, and check if moves are safe

    # How much it cost's to move?

    move_cost = 0.1

    surrounding = board.get_surrounding_halite(ship_loc)

    #Bias staying still to stop wasting halite

    surrounding[None] = surrounding[None] + (surrounding[None]* move_cost)

    # for now if there is no halite at all move south

    if sum([i for i in surrounding.values()]) == 0.0:

        nearest_halite = board.get_nearest_halite(ship_loc)

        target = board.get_index(nearest_halite)

        moves = get_moves_to_target(ship_loc, ship_locations, board, target)

        # now need to somehow check if the NORTH, SOUTH, WEST etc is 

        return np.random.choice(list(moves.keys()))



    else:

        surrounding = board.get_safe_options_surrounding(ship_loc, ship_locations,surrounding)

        return max(surrounding, key=surrounding.get)

    

# Each ship id will be assigned a state, one of COLLECT or DEPOSIT, 

# this was something that was in the tutorial in halite 3

global states

states = {}



COLLECT = "COLLECT"

DEPOSIT = "DEPOSIT"

    



def my_agent(obs):

    state = gameInfo(obs)

    board = haliteBoard(obs)

    

   

    

    halite, shipyards, ships = state.players[state.player]

    opp_halite, opp_shipyards, opp_ships = state.players[1]

    

    action = {}

    

    for uid, shipyard in shipyards.items():

    # Maintain one ship 

        if len(ships) == 0:

            action[uid] = "SPAWN"

    

    for uid, ship in ships.items():

        # Maintain one shipyard 

        if len(shipyards) == 0:

            action[uid] = "CONVERT"

            continue        

    

    for uid, ship_info in ships.items():

        #Assuming it has just spawned

        if uid not in states:

            states[uid] = COLLECT

        # If we are collecting    

        if states[uid] == COLLECT:

            if ship_info[1] > 500:

                states[uid] = DEPOSIT

            else:

                #greedy collect

                surrounding = board.get_surrounding_halite(ship_info[0])  # index 0 is the location, 1 is the amount of halite of the ship

                # get ship locations

                ship_locations = board.get_occupied_locs(ship_info[0], shipyards, ships, opp_shipyards, opp_ships)

                ship_action = greedy_collect(ship_info[0], ship_locations, board)

                if ship_action is not None:

                    action[uid] = ship_action

        

        # return to shipyard

        if states[uid] == DEPOSIT:

            if ship_info[1] < 20:

                states[uid] = COLLECT

            else:

                ship_locations = board.get_occupied_locs(ship_info[0], shipyards, ships, opp_shipyards, opp_ships)

                moves = get_moves_to_target(ship_info[0], ship_locations, board, shipyard)

                if moves == {}:

                    ship_action = None

                else:

                    ship_action = np.random.choice(list(moves.keys()))

                if ship_action is not None:

                    action[uid] = ship_action    



    return action
class gameInfo:

    '''Infomation about the state of the game'''

    def __init__(self, observation):

        #player id

        self.player = observation['player']

        # gives a list containing two player lists - which contain:

        # - player halite (int); dict('shipyardid': shipyardloc); dict('shipn': shipn_loc) 

        self.players = observation['players']

        # turn number

        self.step = observation['step']

        





class haliteBoard:

    ''' Functions for the board

    observation - dict with 3 keys

        player: 0 # player id

        step: turn in the game

        halite map - array of halite

    '''

    def __init__(self, observation):

         # creates a 1d array that matches the halite board for reference

        self.loc_board1d = np.array(list(range(225)))

        # creates a 2d array that matches the halite board for reference

        self.loc_board2d = np.array(list(range(225))).reshape(15,15)

        # the halite board - this should contain positions of all assets 1d

        self.halite_board1d = np.array(observation.halite)

         # the halite board - this should contain positions of all assets 2d

        self.halite_board2d = np.array(observation.halite).reshape(15,15)

        # map details

        self.width = 15

        self.height = 15

        

    def get_xy(self,posistion):

        ''' Takes a position e.g 101 and returns the x,y coordinates as a tuple''' 

        x,y = np.where(self.loc_board2d == posistion)

        coords = list(zip(x,y))

        return coords[0]

    

    # make function that takes coordinates and gets the index of the map

    def get_index(self, xy):

        '''takes tuple of (x, y) and returns position on board'''

        index = (xy[0] * 15) + xy[1] 

        return index

        

    

    

    

    def get_nearest_halite(self,posistion):

        '''finds the coordiantes for the nearest halite, 

        not accounting for wraparound returns (x,y)'''

        # where the ship is

        current_coords = self.get_xy(posistion)

        # where the location on the board has halite

        # this isn't working correctly

        hx, hy = np.where(self.halite_board2d > 100)

        #list of halite locations

        halite_coords = list(zip(hx,hy))

        distances = {}

        

        for i in halite_coords:

            # find euclidean distance, doesn't take into account wrap around

            dist = np.sqrt((i[0] - current_coords[0])**2 + (i[1] - current_coords[1])**2)

            distances[i] = dist

        # from the dict get the closest set of coords     

        closest_xy =  min(distances, key=distances.get)

        return closest_xy

    

   

    def get_surrounding_halite(self, posistion):

        '''returns a dict with halite in each direction accounting for wrap'''

        b = self.halite_board1d

        surrounding = {None:b[posistion],'NORTH':np.take(b,[posistion-15], mode = 'wrap'), 

                       'SOUTH':np.take(b,[posistion+15],mode = 'wrap'),

                       'EAST':np.take(b,[posistion+1],mode = 'wrap'), 

                       'WEST':np.take(b,[posistion-1],mode = 'wrap')}

        return surrounding



    def get_surrounding_loc(self,posistion):

        ''' returns the board location number of the surrounding locations 

        - N,S,E,W accounting for wrap '''

        tb = self.loc_board1d

        surrounding_locations = [np.take(tb,[posistion-15], mode = 'wrap'),

                                 np.take(tb,[posistion+15],mode = 'wrap'),

                                np.take(tb,[posistion+1],mode = 'wrap'),

                                np.take(tb,[posistion-1],mode = 'wrap')]



        return surrounding_locations

    

    def get_occupied_locs(self, posistion, shipyards, ships, opp_shipyards, opp_ships):

        ''' Returns all the occupied locations (number) on the map 

        excluding the current position'''

        

        # need to handle for when there are no ships or shipyards

        opp_ship_locs = [i[0] for i in list(opp_ships.values())]  # e.g [0, 34, 59]

        player_ship_locs = [i[0] for i in list(ships.values())]

        #logic ish

        if len(opp_shipyards.values()) == 0:

            return opp_ship_locs + player_ship_locs

           

        else:

            opp_shipyard_locs = [i for i in list(opp_shipyards.values())]

            return opp_ship_locs + player_ship_locs + opp_shipyard_locs

        

     # working well   

    def is_shipyard_occupied(self, ships, shipyards):

        ''' Returns true if there is a ship in our shipyard'''

        player_ship_locs  = [i[0] for i in list(ships.values())]

        if len(shipyards.values()) > 1:

            shipyards = [i[0] for i in list(shipyards.values())]

        elif len(shipyards.values()) < 1:

            return False

        else:

            #we have one shipyard

            shipyards =[i for i in list(shipyards.values())]

            for i in player_ship_locs:

                if i in shipyards:

                    return True

                else:

                    return False

                

    def get_safe_options_surrounding(self, posistion, ship_locations,surrounding):

        ''' Will take the ships position and the dict (ship_action) of moves 

        for greedy navigation and check if they are safe'''

        # get the number locations around the ship

        surrounding_locs = self.get_surrounding_loc(posistion)

        # I want to remove the occupied locations from the max halite dict

        surrounding_locs = [i[0] for i in surrounding_locs]

        nav_dict = {}

        nav_dict['NORTH'], nav_dict['SOUTH'], nav_dict['EAST'], nav_dict['WEST'] = surrounding_locs

        # now simply remove the keys where their values are in ship_locations

        for k,v in list(nav_dict.items()):

            if v in ship_locations:

                surrounding.pop(k)

        return surrounding   

        
# Play as first position against random agent.

trainer = env.train([None, "random"])



observation = trainer.reset()



def get_moves_to_target(ship_loc,ship_locations, board, target):

    '''Takes ship location and target and returns a list of 1 or more viable moves'''

    posistion_xy = board.get_xy(ship_loc)

    # takes int

    target_xy = board.get_xy(target)



    move_dict = {}

    if posistion_xy[0] > target_xy[0]:

        move_dict['NORTH'] = ship_loc + 15

    if posistion_xy[0] < target_xy[0]:

        move_dict['SOUTH'] = ship_loc - 15

    if posistion_xy[1] < target_xy[1]:

         move_dict['EAST'] = ship_loc + 1

    if posistion_xy[1] > target_xy[1]:

         move_dict['WEST'] = ship_loc - 1

    for k,v in list(move_dict.items()):

            if v in ship_locations:

                move_dict.pop(k)

    return move_dict







def greedy_collect(ship_loc, ship_locations, board):

    ''' Takes a ship location and the halite board returns a valid move'''

    # needs to get the greedy move with a bias, and check if moves are safe

    # How much it cost's to move?

    move_cost = 0.1

    surrounding = board.get_surrounding_halite(ship_loc)

    #Bias staying still to stop wasting halite

    surrounding[None] = surrounding[None] + (surrounding[None]* move_cost)

    # for now if there is no halite at all move south

    if sum([i for i in surrounding.values()]) == 0.0:

        nearest_halite = board.get_nearest_halite(ship_loc)

        target = board.get_index(nearest_halite)

        moves = get_moves_to_target(ship_loc, ship_locations, board, target)

        # now need to somehow check if the NORTH, SOUTH, WEST etc is 

        return np.random.choice(list(moves.keys()))



    else:

        surrounding = board.get_safe_options_surrounding(ship_loc, ship_locations,surrounding)

        return max(surrounding, key=surrounding.get)

    

# Each ship id will be assigned a state, one of COLLECT or DEPOSIT, 

# this was something that was in the tutorial in halite 3

global states

states = {}



COLLECT = "COLLECT"

DEPOSIT = "DEPOSIT"

    



def my_agent(obs):

    state = gameInfo(obs)

    board = haliteBoard(obs)

    

   

    

    halite, shipyards, ships = state.players[state.player]

    opp_halite, opp_shipyards, opp_ships = state.players[1]

    

    action = {}

    

    for uid, shipyard in shipyards.items():

    # Maintain one ship 

        if len(ships) == 0:

            action[uid] = "SPAWN"

    

    for uid, ship in ships.items():

        # Maintain one shipyard 

        if len(shipyards) == 0:

            action[uid] = "CONVERT"

            continue        

    

    for uid, ship_info in ships.items():

        #Assuming it has just spawned

        if uid not in states:

            states[uid] = COLLECT

        # If we are collecting    

        if states[uid] == COLLECT:

            if ship_info[1] > 500:

                states[uid] = DEPOSIT

            else:

                #greedy collect

                surrounding = board.get_surrounding_halite(ship_info[0])  # index 0 is the location, 1 is the amount of halite of the ship

                # get ship locations

                ship_locations = board.get_occupied_locs(ship_info[0], shipyards, ships, opp_shipyards, opp_ships)

                ship_action = greedy_collect(ship_info[0], ship_locations, board)

                if ship_action is not None:

                    action[uid] = ship_action

        

        # return to shipyard

        if states[uid] == DEPOSIT:

            if ship_info[1] < 20:

                states[uid] = COLLECT

            else:

                ship_locations = board.get_occupied_locs(ship_info[0], shipyards, ships, opp_shipyards, opp_ships)

                moves = get_moves_to_target(ship_info[0], ship_locations, board, shipyard)

                if moves == {}:

                    ship_action = None

                else:

                    ship_action = np.random.choice(list(moves.keys()))

                if ship_action is not None:

                    action[uid] = ship_action    



    return action







while not env.done:

    my_action = my_agent(observation)

    print("My Action", my_action)

    observation, reward, done, info = trainer.step(my_action)

    
env.render(mode = 'ipython')
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

print("My Agent vs Random Agent:", mean_reward(evaluate(

    "halite",

    ["/kaggle/working/submission.py", "random"],

    num_episodes=10, configuration={"agentExec": "LOCAL"}

)))