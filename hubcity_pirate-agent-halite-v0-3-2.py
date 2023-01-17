!pip install 'kaggle-environments==0.3.2'
from enum import Enum

import random



import numpy as np





class Direction(Enum):

    NORTH = 1

    SOUTH = 2

    EAST = 3

    WEST = 4

    STATIONARY = 5



opposite_direction = { Direction.NORTH: Direction.SOUTH,

                       Direction.SOUTH: Direction.NORTH,

                       Direction.EAST: Direction.WEST,

                       Direction.WEST: Direction.EAST,

                       Direction.STATIONARY: None }



class Geometry():

    def __init__(self, rows, cols = None):

        self.rows = rows

        self.cols = cols

        if cols is None:

            self.cols = rows



    def to_location(self, pos):

        return (pos // self.rows, pos % self.cols)



    def all_locations(self):

        x, y = np.indices([self.rows, self.cols])

        return zip(x.flatten(), y.flatten())



    def distance_between(self, p1, p2):

        row_d = abs(p1[0] - p2[0])

        row_d = min(row_d, self.rows - row_d)

        col_d = abs(p1[1] - p2[1])

        col_d = min(col_d, self.cols - col_d)

        return row_d + col_d



    def move_direction(self, dir, location):

        if dir == Direction.NORTH:

            return ((location[0] - 1) % self.rows, location[1])

        elif dir == Direction.SOUTH:

            return ((location[0] + 1) % self.rows, location[1])

        elif dir == Direction.EAST:

            return (location[0], (location[1] + 1) % self.cols)

        elif dir == Direction.WEST:

            return (location[0], (location[1] - 1) % self.cols)

        else: # STATIONARY

            return location



    def order_directions(self, loc, goal):

        dir_dist = {}

        dirs = list(Direction)

        random.shuffle(dirs)

        for d in dirs:

            new_loc = self.move_direction(d, loc)

            dir_dist[d] = self.distance_between(new_loc, goal)

        best_dir = min(dir_dist, key=dir_dist.get)

        opp_dir = opposite_direction[best_dir]

        return sorted(dir_dist, key=lambda x: (dir_dist[x], x==opp_dir))





class Map:

    def __init__(self, geometry, value_list, unoccupied = 0):

        self.geometry = geometry

        self.unoccupied = unoccupied

        map = np.zeros((geometry.rows, geometry.cols))

        for pos, value in enumerate(value_list):

            map[geometry.to_location(pos)] = value

        self.map = map



    def locations(self):

        return zip(*(self.map > self.unoccupied).nonzero())



    def __getitem__(self, key):

        return self.map[key]



    def __setitem__(self, key, value):

        self.map[key] = value



    def adjacent(self, location):

        adj = {}

        for dir in Direction:

            if dir == Direction.STATIONARY:

                continue

            loc = self.geometry.move_direction(dir, location)

            adj[dir] = self.map[loc]

        return adj



    def closest(self, location):

        min_dist = float("inf")

        closest_loc = None

        for goal in self.locations():

            dist = self.geometry.distance_between(location, goal)

            if dist < min_dist:

                min_dist = dist

                closest_loc = goal

        return closest_loc, min_dist



    def max_location(self):

        pos = self.map.argmax()

        return self.geometry.to_location(pos)



    def discount_filter(self, center, dropoff):

        discounted = np.zeros((self.geometry.rows, self.geometry.cols))

        for loc in self.geometry.all_locations():

            dist = self.geometry.distance_between(loc, center)

            discounted[loc] = self.map[loc] * (dropoff ** dist)

        return discounted

from kaggle_environments.utils import Struct





def create_obs_maps(obs):

    size = int(len(obs.halite) ** 0.5)

    geo = Geometry(size)

    friends = [obs.player]

    enemies = [i for i in range(len(obs.players)) if i != obs.player]

    everyone = friends + enemies



    def yards_for(player_ids):

        for p in player_ids:

            for item in obs.players[p][1].items():

                yield item



    def ships_for(player_ids):

        for p in player_ids:

            for item in obs.players[p][2].items():

                yield item



    def halite_map():

        return Map(geo, [x if x > 0 else 0 for x in obs.halite])



    def yard_map(group):

        lst = [0] * size * size

        for _, pos in yards_for(group):

            lst[pos] = 1

        return Map(geo, lst)



    def ship_map(group):

        unoccupied = -1

        lst = [unoccupied] * size * size

        for _, (pos, cargo) in ships_for(group):

            lst[pos] = cargo

        return Map(geo, lst, unoccupied=unoccupied)



    def unit_owners():

        owner = {}

        for i in everyone:

            for name, _ in yards_for([i]):

                owner[name] = i

            for name, _ in ships_for([i]):

                owner[name] = i

        return owner



    def locations():

        locs = {}

        for i in everyone:

            for name, pos in yards_for([i]):

                locs[name] = geo.to_location(pos)

            for name, (pos, cargo) in ships_for([i]):

                locs[name] = geo.to_location(pos)

        return locs



    def player_scores():

        score = [0] * len(everyone)

        for i in everyone:

            score[i] = obs.players[i][0]

        return score



    result = {

        # geometry

        "geometry": geo,

        # wraparound maps

        "halite": halite_map(),

        "friendly_yards": yard_map(friends),

        "friendly_ships": ship_map(friends),

        "enemy_yards": yard_map(enemies),

        "enemy_ships": ship_map(enemies),

        # owner lookup by uid

        "owner_of": unit_owners(),

        # uid lookup by location

        "ship_at": {geo.to_location(p):u for u,(p,c) in ships_for(everyone)},

        "yard_at": {geo.to_location(p):u for u,p in yards_for(everyone)},

        # location lookup by uid

        "location_of": locations(),

        # score lookup

        "score_of": player_scores(),

        # general info

        "step": obs.step,

        "bank": obs.players[obs.player][0],

        "num_yards": len(obs.players[obs.player][1]),

        "num_ships": len(obs.players[obs.player][2]),

        # id info

        "my_id": obs.player,

        "my_ship_ids": [x for x, _ in ships_for(friends)],

        "my_yard_ids": [x for x, _ in yards_for(friends)]

    }

    # use the same Struct as kaggle obs

    return Struct(**result)

def simple_convert(config, obsm):

    actions = {}

    if obsm.num_yards == 0 or obsm.num_ships / obsm.num_yards >= 5:

        loc = obsm.friendly_ships.max_location()

        fyl = obsm.friendly_yards.locations()

        money_left = obsm.friendly_ships[loc] + obsm.bank - config.convertCost

        can_afford = (obsm.num_ships == 1 and money_left > config.spawnCost)

        can_afford = can_afford or (obsm.num_ships > 1 and money_left > 0)

        if loc not in fyl and can_afford:

            ship_id = obsm.ship_at[loc]

            actions[ship_id] = "CONVERT"

            obsm.bank += (obsm.friendly_ships[loc] - config.convertCost)

    return actions



def ship_share(obsm, add_one = True):

    total_num_ships = len(obsm.ship_at)

    total_halite = obsm.halite.map.sum()

    return total_halite / (total_num_ships + add_one)



def simple_spawn(config, obsm):

    actions = {}

    yard_exists = obsm.num_yards > 0

    worth_it = ship_share(obsm) > config.spawnCost

    if yard_exists and worth_it and obsm.bank > config.spawnCost:

        yard_id = obsm.my_yard_ids[-1]

        actions[yard_id] = "SPAWN"

        obsm.bank -= config.spawnCost

    return actions



def simple_move(obsm, goals, locations_taken):

    actions = {}

    for ship_id, goal_loc in goals.items():

        ship_loc = obsm.location_of[ship_id]

        preferred_dirs = obsm.geometry.order_directions(ship_loc, goal_loc)

        for dir in preferred_dirs:

            move_to = obsm.geometry.move_direction(dir, ship_loc)

            if move_to not in locations_taken:

                if dir != Direction.STATIONARY:

                    actions[ship_id] = dir.name

                locations_taken.add(move_to)

                break

    return actions





import numpy as np

from scipy.optimize import linear_sum_assignment





def assign_goals(obsm, ship_ids, goal_map, discount_factor, cargo_min):

    goal_for = {}

    needs_goal = []

    # if it meets cargo minimum send to closest shipyard

    for ship_id in ship_ids:

        ship_loc = obsm.location_of[ship_id]

        ship_cargo = obsm.friendly_ships[ship_loc]

        if ship_cargo > cargo_min and obsm.num_yards > 0:

            closest_yard = obsm.friendly_yards.closest(ship_loc)[0]

            goal_for[ship_id] = closest_yard

        else:

            needs_goal.append(ship_id)

    # assign goals to remaining ships with linear_sum_assignment 

    # based on distance-discounted goal map

    assignments_matrix = np.zeros((len(needs_goal), goal_map.map.flatten().size))

    for i, ship_id in enumerate(needs_goal):

        loc = obsm.location_of[ship_id]

        df = goal_map.discount_filter(loc, discount_factor)

        assignments_matrix[i] = df.flatten()

    ship_idx, goal_pos = linear_sum_assignment(assignments_matrix, True)

    for idx, pos in zip(ship_idx, goal_pos):

        ship_id = needs_goal[idx]

        goal_loc = obsm.geometry.to_location(pos)

        if goal_map[goal_loc] == goal_map.unoccupied:

            # if you don't have a goal, chase the largest enemy ship

            goal_for[ship_id] = obsm.enemy_ships.max_location()

        else:

            goal_for[ship_id] = goal_loc

    return goal_for

def pirate_agent(obs, config):

    actions = {}

    obsm = create_obs_maps(obs)

    needs_goal = obsm.my_ship_ids.copy()

    locations_taken = set()

    

    # should spawn

    spawns = simple_spawn(config, obsm)

    for yard_id in spawns.keys():

        locations_taken.add(obsm.location_of[yard_id])

    actions.update(spawns)

    

    # should convert

    converts = simple_convert(config, obsm)

    for ship_id in converts.keys():

        needs_goal.remove(ship_id)

    actions.update(converts)

    

    # assign goals

    goal_for = {}

    miners = needs_goal[0:1]

    goal_for.update(assign_goals(obsm, miners, obsm.halite, 0.8, 217))

    pirates = needs_goal[len(miners):-1]

    goal_for.update(assign_goals(obsm, pirates, obsm.enemy_ships, 0.8, 0))

    destroyers = needs_goal[len(miners)+len(pirates):]

    goal_for.update(assign_goals(obsm, destroyers, obsm.enemy_yards, 0.8, 0))

    

    # move toward goals

    moves = simple_move(obsm, goal_for, locations_taken)

    actions.update(moves)

    return actions

from kaggle_environments import make



env = make("halite", debug=True)

env.run([pirate_agent, "random", "random", "random"])

env.render(mode="ipython", width=800, height=600)