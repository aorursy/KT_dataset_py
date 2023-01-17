import random





class Board:

    """Collection of useful, static functions on the board. Do not instantiate this class!"""

    size = 15

    all_fields = [i for i in range(size**2)]

    directions = ['WEST', 'NORTH', 'EAST', 'SOUTH']



    @staticmethod

    def field_to_coord(field):

        """Return 2d coordinates corresponding to 1d field ID"""

        return field % Board.size, field // Board.size



    @staticmethod

    def coord_to_field(coord):

        """Return 1d field id for given 2d coordinate"""

        return coord[1] * Board.size + coord[0]



    @staticmethod

    def west(field):

        """Return field ID west to given field"""

        x, y = Board.field_to_coord(field)

        return Board.coord_to_field(((x-1) % Board.size, y))



    @staticmethod

    def east(field):

        """Return field ID east to given field"""

        x, y = Board.field_to_coord(field)

        return Board.coord_to_field(((x+1) % Board.size, y))



    @staticmethod

    def north(field):

        """Return field ID north to given field"""

        x, y = Board.field_to_coord(field)

        return Board.coord_to_field((x, (y-1) % Board.size))



    @staticmethod

    def south(field):

        """Return field ID south to given field"""

        x, y = Board.field_to_coord(field)

        return Board.coord_to_field((x, (y+1) % Board.size))



    @staticmethod

    def field_in_direction(field, direction):

        """Return field in given direction"""

        if direction == 'WEST': return Board.west(field)

        if direction == 'NORTH': return Board.north(field)

        if direction == 'EAST': return Board.east(field)

        if direction == 'SOUTH': return Board.south(field)

        return None



    @staticmethod

    def surrounding_fields(field):

        """Return list of surrounding fields (W,N,E,S of field)"""

        return [Board.west(field), Board.north(field), Board.east(field), Board.south(field)]



    @staticmethod

    def direction_to_dest(src_field, dst_field, avoid_fields=[], random_if_blocked=True):

        """

        Return recommended move direction to get from_pos to_pos.

        Avoid moving onto given avoid_fields (if any) to avoid collisions.

        If direction to destination is blocked, either stay still or move into a random other (non-avoided) field

        # TODO: use shortcuts around the borders of the map

        """

        src_x, src_y = Board.field_to_coord(src_field)

        dst_x, dst_y = Board.field_to_coord(dst_field)

        free_directions = [direction for direction in Board.directions

                           if Board.field_in_direction(src_field, direction) not in avoid_fields]



        if src_x < dst_x:

            if Board.east(src_field) in free_directions:

                return "EAST"

        if src_x > dst_x:

            if Board.west(src_field) in free_directions:

                return "WEST"

        if src_y < dst_y:

            if Board.south(src_field) in free_directions:

                return "SOUTH"

        if src_y > dst_y:

            if Board.north(src_field) in free_directions:

                return "NORTH"

        if random_if_blocked:

            return random.choice(free_directions + [None])

        return None

    

    @staticmethod

    def distance(field1, field2):

        """

        Return the shortest path distance from field1 to field2 (= vice versa)

        This is moving within the field, no shortcuts around the borders!

        """

        x1, y1 = Board.field_to_coord(field1)

        x2, y2 = Board.field_to_coord(field2)

        return abs(x1-x2) + abs(y1-y2)
class Ship:

    """Auxiliary class holding info about a ship"""

    def __init__(self, ship_id, player_id, obs, dest=None):

        self.id = ship_id

        self.player_id = player_id

        self.obs = obs

        self.dest = dest



        # get further info from observation

        self.pos = obs.players[player_id][2][ship_id][0]

        self.coord = Board.field_to_coord(self.pos)

        self.halite = obs.players[player_id][2][ship_id][1]

        # halite at the field where the ship currently is

        self.halite_here = obs.halite[self.pos]



    def __str__(self):

        return self.id



    def __repr__(self):

        return self.id



    def neighbor_with_max_halite(self):

        """Return neighboring field with most halite"""

        surrounding_fields = Board.surrounding_fields(self.pos)

        best_index = Agent.argmax(surrounding_fields, key=self.obs.halite.__getitem__)

        best = surrounding_fields[best_index]

        best_halite = round(self.obs.halite[best], 2)

        return best



    def direction_to_max_halite(self, candidates, avoid_fields=[]):

        """Return field and movement direction to most halite from all candidate fields"""

        # TODO: prefer closer fields with similar amounts of halite

        best = Agent.argmax(candidates, key=self.obs.halite.__getitem__)

        direction = Board.direction_to_dest(self.pos, best, avoid_fields)

        best_halite = round(self.obs.halite[best], 2)

        return best, direction
class Shipyard:

    """Auxiliary class holding info about a shipyard"""

    def __init__(self, id, pos, player_id):

        self.id = id

        self.pos = pos

        self.coord = Board.field_to_coord(pos)

        self.player_id = player_id



    def __repr__(self):

        return self.id
class Agent:

    """Agent helper class with multiple auxiliary functions that don't fit anywhere else"""

    def __init__(self, collect_thres, load_threshold, max_yards, max_yard_distance, max_ships):

        """

        Create new agent object

        :param collect_thres: Min halite (remaining) at a field to stop and collect it. Good value: 50

        :param load_threshold: Halite load at the ship up to which to search for and collect halite. After reaching threshold --> return home. Good value: 1000

        :param max_yards: Max number of shipyards to create

        :param max_yard_distance: If distance of ship to existing yard is higher than this, create a new shipyard

        :param max_ships: Max number of ships allowed --> try to maintain that number

        """

        self.obs = None

        self.collect_thres = collect_thres

        self.load_threshold = load_threshold

        self.max_yards = max_yards

        self.max_yard_distance = max_yard_distance

        self.max_ships = max_ships



        self.ships = []

        self.yards = []

        # keep track of num ship and shipyards to detect if I lost a ship

        self.num_ships = 0

        self.num_yards = 0



    # from https://www.kaggle.com/awanderingsoul/halite-basic-greedy-agent

    @staticmethod

    def argmax(arr, key=None):

        return arr.index(max(arr, key=key)) if key else arr.index(max(arr))



    @staticmethod

    def reverse_direction(direction):

        """Return reverse direction"""

        if direction == 'WEST': return 'EAST'

        if direction == 'EAST': return 'WEST'

        if direction == 'NORTH': return 'SOUTH'

        if direction == 'SOUTH': return 'NORTH'

        return None



    def occupied_fields(self):

        """Return occupied fields to where the ship cannot move in form of two lists: Enemy shipyards, enemy ships"""

        enemy_yards = []

        enemy_ships = []

        num_players = len(self.obs.players)

        other_players = [i for i in range(num_players) if i != self.obs.player]



        for p in other_players:

            # add shipyards

            shipyards = self.obs.players[p][1]

            enemy_yards.extend(shipyards.values())

            # add ships

            ships = self.obs.players[p][2]

            enemy_ships.extend([ship_attr[0] for ship_attr in ships.values()])



        return enemy_yards, enemy_ships



    def danger_fields(self, enemy_ships):

        """Return set of fields around the given enemy ships, which are dangerous because the ships may move there."""

        # set avoids duplicates

        fields = set()

        for ship in enemy_ships:

            surr_fields = set(Board.surrounding_fields(ship))

            fields.update(surr_fields)

        return fields



    def create_yard_objects(self, yard_dict, player):

        """Create and return list of shipyard objects based on given yard dict from the observations"""

        yards = []

        for id, pos in yard_dict.items():

            yards.append(Shipyard(id, pos, player))

        return yards



    def create_ship_objects(self, ship_dict, player):

        """Create and return list of ship objects based on ship_dict"""

        # get old destinations and maintain

        ship_dests = {ship.id: ship.dest for ship in self.ships}

        return [Ship(id, player, self.obs, ship_dests.get(id)) for id in ship_dict.keys()]



    def retrieve_info(self, obs):

        """

        Retrieve main info from observation and set as self attributes.

        Should be the first function to call in each step to retrieve and set observations.

        """

        self.obs = obs

        # retrieve info from observation

        self.halite, yard_dict, ship_dict = obs.players[obs.player]



        # create list of shipyard objects

        self.yards = self.create_yard_objects(yard_dict, self.obs.player)

        self.ships = self.create_ship_objects(ship_dict, self.obs.player)



        # check if I lost a ship; then update

        self.num_ships = len(self.ships)

        self.num_yards = len(self.yards)



    def check_create_ships(self):

        """Create new ship"""

        # ensure not to exceed maximum and to have enough remaining halite

        if len(self.ships) < self.max_ships and self.halite > 500:

            # exclude shipyards where there is currently already a ship (would lead to collision)

            my_ship_pos = [ship.pos for ship in self.ships]

            free_yards = [yard for yard in self.yards if yard.pos not in my_ship_pos]

            if len(free_yards) > 0:

                # select random shipyard to spawn new ship

                yard = random.choice(free_yards)

                self.num_ships += 1

                self.halite -= 500

                return {yard.id: 'SPAWN'}

        return None



    def create_yard(self, ship):

        """Create shipyard at ship if none exists yet"""

        if len(self.yards) < self.max_yards:

            # do I still have enough remaining halite to respawn after creating the shipyard?

            if self.halite + ship.halite - 2000 > 500:

                # decrement ship counter already to avoid warning about having lost a ship

                self.num_ships -= 1

                self.num_yards += 1

                self.halite -= 2000

                return {ship.id: 'CONVERT'}

        else:

            # already at max number of yards

            return None



    def ship_can_dropoff_halite(self, ship):

        """Return true iff ship is at one of my shipyards and is carrying halite that it can drop off"""

        if ship.halite > 0:

            for yard in self.yards:

                if ship.pos == yard.pos:

                    return True

        return False



    def closest_yard(self, ship):

        """Return closes shipyard and distance to ship"""

        closest_yard = None

        closest_dist = None

        for yard in self.yards:

            dist = Board.distance(ship.pos, yard.pos)

            if closest_dist is None or dist < closest_dist:

                closest_yard = yard

                closest_dist = dist

        assert closest_yard is not None and closest_dist is not None, "There is no shipyard!"

        return closest_yard, closest_dist



    def act(self, obs):

        """The main acting function called by my_agent. Returns the selected action."""

        self.retrieve_info(obs)

        action = {}



        # ensure there are enough ships and at least one shipyard

        ship_action = self.check_create_ships()

        if ship_action is not None:

            action.update(ship_action)

        if len(self.yards) == 0:

            yard_action = self.create_yard(self.ships[0])

            if yard_action is not None:

                action.update(yard_action)



        # avoid all fields with enemy shipyards, ships, and fields surrounding ships to where they could move

        enemy_yards, enemy_ships = self.occupied_fields()

        avoid_fields = self.danger_fields(enemy_ships)

        avoid_fields.update(set(enemy_yards), set(enemy_ships))

        # field blocked because another of my ships will move there

        own_move_fields = set()



        # select ships without (convert) actions for movement

        free_ships = [ship for ship in self.ships if ship.id not in action.keys()]

        for ship in free_ships:

            own_other_ships = [s.pos for s in self.ships if s != ship]

            blocked_fields = list(avoid_fields.union(own_move_fields, own_other_ships))



            # reset destination if reached

            if ship.dest == ship.pos:

                ship.dest = None

            # reset destination if it's occupied by another ship

            if ship.dest in enemy_ships:

                ship.dest = None



            # if there is enough halite to collect or the ship can drop off halite at a shipyard, stop

            # also collect on the way home

            if ship.halite_here >= self.collect_thres or self.ship_can_dropoff_halite(ship):

                continue



            # move towards destination

            if ship.dest is not None:

                direction = Board.direction_to_dest(ship.pos, ship.dest, avoid_fields=blocked_fields)

                # already at the destination or no place to move (without collisions) --> stay

                if direction is None:

                    continue

                own_move_fields.add(Board.field_in_direction(ship.pos, direction))

                action[ship.id] = direction



            # pick new destination

            else:

                # if enough load, go back to ship yard to drop off halite

                # or when the game is about to end

                if ship.halite >= self.load_threshold or self.obs.step >= 395:

                    # select closest shipyard; create new if existing one(s) are too far

                    closest_yard, yard_dist = self.closest_yard(ship)

                    if yard_dist > self.max_yard_distance:

                        yard_action = self.create_yard(ship)

                        if yard_action is not None:

                            action.update(yard_action)

                            continue

                    # navigate to closest shipyard

                    ship.dest = closest_yard.pos

                    direction = Board.direction_to_dest(ship.pos, ship.dest, avoid_fields=blocked_fields)

                    # already at the destination or no place to move (without collisions) --> stay

                    if direction is None:

                        continue

                    own_move_fields.add(Board.field_in_direction(ship.pos, direction))

                    action[ship.id] = direction

                # else move to field with high halite to collect more

                else:

                    # go to most halite overall (collect on the way)

                    # destination nodes of other ships --> avoid sending all ships to the same destination

                    own_dest_fields = [ship.dest for ship in self.ships]

                    # exclude occupied fields and dest fields of other ships from candidates to avoid waiting

                    candidates = [field for field in Board.all_fields if field not in list(blocked_fields)+own_dest_fields]

                    dest_field, direction = ship.direction_to_max_halite(candidates, avoid_fields=blocked_fields)

                    ship.dest = dest_field

                    # already at the destination or no place to move (without collisions) --> stay

                    if direction is None:

                        continue

                    own_move_fields.add(Board.field_in_direction(ship.pos, direction))

                    action[ship.id] = direction



        return action
# global agent variable for persistent state between steps

agent = Agent(collect_thres=50, load_threshold=1000, max_yards=5, max_yard_distance=5, max_ships=5)





def my_agent(obs):

    global agent

    return agent.act(obs)
from kaggle_environments import make, evaluate



env = make("halite", debug=True)



def validate(agent):

#     assert agent == 'random' or os.path.isfile(agent)



    print(f"Validating {agent} agent:")

    env.run([agent, agent])

    print("EXCELLENT SUBMISSION!" if env.toJSON()["statuses"] == ["DONE", "DONE"] else "MAYBE BAD SUBMISSION?")

    

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

    return f'wins={wins/len(rewards):.2f}, ties={ties/len(rewards):.2f}, loses={loses/len(rewards):.2f}'





def evaluate_agents(agent1, agent2, eval_eps):

    # Run multiple episodes to estimate its performance.

    # Setup agentExec as LOCAL to run in memory (runs faster) without process isolation.

    rewards = evaluate("halite", [agent1, agent2], num_episodes=eval_eps, configuration={"agentExec": "LOCAL"})

    print(f"{agent1} vs {agent2} Agent:", mean_reward(rewards))
agent1 = my_agent

agent2 = 'random'
validate(agent1)
evaluate_agents(agent1, agent2, eval_eps=30)
env.render(mode="ipython", width=800, height=600)