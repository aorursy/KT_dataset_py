from dataclasses import dataclass, field

from typing import NamedTuple, Tuple, List, Set, FrozenSet, Dict, Iterable, Optional, Any, Iterator

import itertools

from itertools import chain

import collections

from operator import itemgetter, attrgetter

from collections import Counter, defaultdict

import numpy as np

from scipy.optimize import linear_sum_assignment
SIZE_X: int = 15

SIZE_Y: int = 15

    

NUM_SHIPS: int = 7

MIN_NUM_GATHERER: int = 3

NUM_HUNTER: int = 1



HALITE_MINE_FACTOR: float = 0.25

TIME_FACTOR: float = 0.99
class Pos(NamedTuple):

    x: int

    y: int



    def __repr__(self):

        return f"[{self.x}:{self.y}]"





DiffType = Tuple[int, int]





class Geometry:

    def __init__(self, size_x: int, size_y: int):

        self.size_x = size_x

        self.size_y = size_y



        self.poses = {Pos(x, y) for x in range(size_x) for y in range(size_y)}



    def int_to_pos(self, int_pos: int) -> Pos:

        x = int_pos % self.size_x

        y = int_pos // self.size_x

        assert 0 <= y < self.size_y

        return Pos(x, y)



    def _to_xy(self, pos: Pos) -> DiffType:

        assert isinstance(pos, Pos), f"Invalid position {pos}"

        x = pos.x

        y = pos.y

        if not 0 <= x < self.size_x and 0 <= y < self.size_y:

            raise ValueError(

                f"Position {pos} is illegal for geometry of size {self.size_x} x {self.size_y}"

            )



        return (x, y)



    def _to_pos(self, x: int, y: int) -> Pos:

        x %= self.size_x

        y %= self.size_y

        return Pos(x, y)



    def _diff_to(self, pos1: Pos, pos2: Pos) -> DiffType:

        """

        Return diff vector for shortest path (torus)

        External function currently are not supposed to deal with position differences

        """

        x1, y1 = self._to_xy(pos1)

        x2, y2 = self._to_xy(pos2)



        dxy_shifted = [

            (x2 + shift_x - x1, y2 + shift_y - y1)

            for shift_x, shift_y in itertools.product(

                [-self.size_x, 0, self.size_x], [-self.size_y, 0, self.size_y]

            )

        ]



        dx, dy = min(dxy_shifted, key=lambda dxy: self._raw_dist(dxy[0], dxy[1]))



        return (dx, dy)



    @staticmethod

    def _raw_dist(dx: int, dy: int) -> int:

        return abs(dx) + abs(dy)



    def dist(self, pos1: Pos, pos2: Pos) -> int:

        dx, dy = self._diff_to(pos1, pos2)

        return self._raw_dist(dx, dy)



    def pos_towards(self, pos1: Pos, pos2: Pos) -> Set[Pos]:

        if pos1 == pos2:

            return {pos2}



        x1, y1 = self._to_xy(pos1)

        dx, dy = self._diff_to(pos1, pos2)



        result = []



        if dx > 0:

            result.append((x1 + 1, y1))



        if dx < 0:

            result.append((x1 - 1, y1))



        if dy > 0:

            result.append((x1, y1 + 1))



        if dy < 0:

            result.append((x1, y1 - 1))



        return set(itertools.starmap(self._to_pos, result))



    def get_prox(self, pos: Pos, *dists: int) -> Set[Pos]:

        x, y = self._to_xy(pos)



        result = []

        for dist in dists:

            if dist == 0:

                result.append((x, y))

                continue

            for d in range(dist):

                result.append((x + d, y + dist - d))

                result.append((x - d, y - dist + d))

                result.append((x - dist + d, y + d))

                result.append((x + dist - d, y - d))



        return set(itertools.starmap(self._to_pos, result))



    
geometry = Geometry(SIZE_X, SIZE_Y)

dist = geometry.dist

get_prox = geometry.get_prox

pos_towards = geometry.pos_towards

int_to_pos = geometry.int_to_pos

_diff_to = geometry._diff_to





@dataclass

class ClosestDist:

    idx: int

    pos: Pos

    dist: int





def find_closest(pos: Pos, dest_poses: Iterable[Pos]) -> Optional[ClosestDist]:

    dists = [

        (i, dest_pos, dist(pos, dest_pos)) for i, dest_pos in enumerate(dest_poses)

    ]



    if not dists:

        return None



    closest = min(dists, key=itemgetter(2),)



    return ClosestDist(*closest)





def is_unique(elems: Iterable):  # what if empty?

    cnts = Counter(elems)



    if not cnts:

        return True



    return cnts.most_common(1)[0][1] == 1
class ObservationShip(NamedTuple):

    pos: Pos

    halite: float





class ObservationShipYard(NamedTuple):

    pos: Pos





Id = str

EnemyId = Tuple[int, Id]





@dataclass  # pylint: disable=too-many-instance-attributes

class Observation:

    """

    Holds obs info with types Pos, ObservationShip, ObservationShipYard for clarity

    Precalculates some frequently needed information

    """



    step: int

    player_idx: int

    map_halite: Dict[Pos, float]



    player_halite: List[float]

    ships: List[Dict[Id, ObservationShip]]

    shipyards: List[Dict[Id, ObservationShipYard]]



    my_halite: float = field(init=False)

    my_ships: Dict[Id, ObservationShip] = field(init=False)

    my_shipyards: Dict[Id, ObservationShipYard] = field(init=False)

    ship_poses: Set[Pos] = field(init=False)

    shipyard_poses: Set[Pos] = field(init=False)



    enemy_ships: Dict[EnemyId, ObservationShip] = field(init=False)

    enemy_shipyards: Dict[EnemyId, ObservationShipYard] = field(init=False)

    enemy_ship_poses: Set[Pos] = field(init=False)

    enemy_shipyard_poses: Set[Pos] = field(init=False)



    num_ships: int = field(init=False)



    def __repr__(self):

        return (

            f"Observation(step={self.step}, {self.num_ships} ships, "

            f"{len(self.enemy_ships)} enemy ships)"

        )



    def __post_init__(self):

        self.my_halite = self.player_halite[self.player_idx]



        self.my_ships = self.ships[self.player_idx]



        self.enemy_ships = {

            (idx, id_): ship

            for idx, cur_ships in enumerate(self.ships)

            if idx != self.player_idx

            for id_, ship in cur_ships.items()

        }



        self.my_shipyards = self.shipyards[self.player_idx]



        self.enemy_shipyards = {

            (idx, id_): ship

            for idx, cur_shipyards in enumerate(self.shipyards)

            if idx != self.player_idx

            for id_, ship in cur_shipyards.items()

        }



        self.ship_poses = set(ship.pos for ship in self.my_ships.values())



        self.shipyard_poses = set(

            shipyard.pos for shipyard in self.my_shipyards.values()

        )



        self.enemy_ship_poses = set(ship.pos for ship in self.enemy_ships.values())



        self.enemy_shipyard_poses = set(

            shipyard.pos for shipyard in self.enemy_shipyards.values()

        )



        self.num_ships = len(self.my_ships)



    @classmethod

    def from_obs(cls, obs):

        player_halite = []

        shipyards = []

        ships = []



        for (cur_player_halite, cur_shipyards, cur_ships) in obs["players"]:

            player_halite.append(cur_player_halite)



            ships.append(

                {

                    id_: ObservationShip(pos=int_to_pos(int_pos), halite=halite)

                    for id_, (int_pos, halite) in cur_ships.items()

                }

            )



            shipyards.append(

                {

                    id_: ObservationShipYard(pos=int_to_pos(int_pos))

                    for id_, int_pos in cur_shipyards.items()

                }

            )



        return cls(

            step=obs["step"],

            player_idx=obs["player"],

            map_halite={int_to_pos(pos): val for pos, val in enumerate(obs["halite"])},

            player_halite=player_halite,

            ships=ships,

            shipyards=shipyards,

        )

class AssignSolverMixin:

    """

    You can use goal=None to signify a non-conflicting goal

    """



    @property

    def obj_goal_penalty(self) -> Tuple[Any, Any, float]:

        raise NotImplementedError()





def solve_assign(penalty_objs: List[AssignSolverMixin]):

    if not penalty_objs:

        return []

    

    obj_goal_penalties = [penalty_obj.obj_goal_penalty for penalty_obj in penalty_objs]



    # Rewrite non-conflict goals

    non_conflict_goals = (("nc", i) for i in itertools.count())



    obj_goal_penalties = [

        (obj, goal if goal is not None else next(non_conflict_goals), score)

        for obj, goal, score in obj_goal_penalties

    ]



    all_to_resolve_objs = defaultdict(list)



    for obj_goal_penalty, penalty_obj in zip(obj_goal_penalties, penalty_objs):

        all_to_resolve_objs[obj_goal_penalty[:2]].append(

            (obj_goal_penalty, penalty_obj)

        )



    best_to_resolve_objs = list(

        min(objs, key=lambda x: x[1].obj_goal_penalty[2])

        for objs in all_to_resolve_objs.values()

    )



    best_obj_goal_penalties, best_penalty_objs = zip(*best_to_resolve_objs)



    matrix, obj_goal_penalty_map = make_matrix(

        best_obj_goal_penalties, best_penalty_objs

    )



    x_idxs, y_idxs = linear_sum_assignment(matrix)



    try:

        result = [

            obj_goal_penalty_map[x_idx, y_idx] for x_idx, y_idx in zip(x_idxs, y_idxs)

        ]

    except KeyError as exc:

        raise ValueError(

            f"Assignment solution could not be resolved for {exc}. "

            "You may need to add a stay on the spot move to the bot."

        )



    assert is_unique(x.obj_goal_penalty[0] for x in result), result

    assert is_unique(x.obj_goal_penalty[1] for x in result), result



    return result





def make_matrix(obj_goal_penalties, penalty_objs: List[AssignSolverMixin]):

    assert is_unique(obj[:2] for obj in obj_goal_penalties)



    xs = list(set(x[0] for x in obj_goal_penalties))

    ys = list(set(x[1] for x in obj_goal_penalties))

    penalty_vals = list(x[2] for x in obj_goal_penalties)



    result = np.full(shape=(len(xs), len(ys)), fill_value=np.inf)



    obj_goal_penalty_map = {}



    for (x, y, penalty), obj in zip(obj_goal_penalties, penalty_objs):

        x_idx = xs.index(x)

        y_idx = ys.index(y)



        obj_goal_penalty_map[x_idx, y_idx] = obj



        result[x_idx, y_idx] = penalty



    return result, obj_goal_penalty_map

@dataclass

class Plan(AssignSolverMixin):

    """

    Bot plan with all information for resolution



    Every plan probably should also have a StayAction to guarantee resolution

    """



    id: str

    score: float



    @property

    def actions(self):

        raise NotImplementedError()





@dataclass

class MovePlan(Plan):

    start_pos: Pos

    end_pos: Pos

    forbidden_pos: Set[Pos] = field(default_factory=set)



    @property

    def obj_goal_penalty(self):

        return (self.id, self.end_pos, -self.score)



    @property

    def actions(self):

        if self.start_pos == self.end_pos:

            return [StayAction(id=self.id, pos=self.start_pos, score=1)]



        next_poses = pos_towards(self.start_pos, self.end_pos) - self.forbidden_pos



        return [

            MoveAction(id=self.id, from_pos=self.start_pos, pos=pos, score=1)

            for pos in next_poses

        ] + [StayAction(id=self.id, pos=self.start_pos, score=0)]





@dataclass

class ConvertPlan(Plan):

    pos: Pos



    @property

    def obj_goal_penalty(self):

        return (self.id, None, -self.score)



    @property

    def actions(self):

        return [

            ConvertAction(id=self.id, pos=self.pos, score=self.score),

            StayAction(id=self.id, pos=self.pos, score=0),

        ]





@dataclass

class SpawnPlan(Plan):

    pos: Pos



    @property

    def obj_goal_penalty(self):

        return (self.id, None, -self.score)



    @property

    def actions(self):

        return [

            SpawnAction(id=self.id, pos=self.pos, score=self.score),

            NoShipYardAction(id=self.id, pos=self.pos, score=0),

        ]





@dataclass

class ScatterPlan(Plan):

    start_pos: Pos



    @property

    def obj_goal_penalty(self):

        return (self.id, None, -self.score)



    @property

    def actions(self):

        return [

            MoveAction(id=self.id, from_pos=self.start_pos, pos=next_pos, score=1)

            for next_pos in get_prox(self.start_pos, 1)

        ] + [StayAction(id=self.id, pos=self.start_pos, score=0)]

@dataclass

class Action(AssignSolverMixin):

    id: Id

    score: float

    pos: Pos



    @property

    def obj_goal_penalty(self):

        return (self.id, self.pos, -self.score)



    @property

    def halite_command(self):

        raise NotImplementedError()





@dataclass

class ConvertAction(Action):

    @property

    def obj_goal_penalty(self):

        return (self.id, ("sy", self.pos), -self.score)



    @property

    def halite_command(self):

        return {self.id: "CONVERT"}





@dataclass

class SpawnAction(Action):

    @property

    def halite_command(self):

        return {self.id: "SPAWN"}





@dataclass

class StayAction(Action):

    @property

    def halite_command(self):

        return {}





class NoShipYardAction(Action):

    @property

    def obj_goal_penalty(self):

        return (self.id, ("sy", self.pos), -self.score)



    @property

    def halite_command(self):

        return {}





@dataclass

class MoveAction(Action):

    from_pos: Pos



    def __repr__(self):

        return f"MoveAction({self.id}: {self.from_pos}->{self.pos}; {self.score})"



    @property

    def halite_command(self):

        if self.from_pos == self.pos:

            return {}



        dx, dy = _diff_to(self.from_pos, self.pos)

        result = {

            (1, 0): "EAST",

            (-1, 0): "WEST",

            (0, 1): "SOUTH",

            (0, -1): "NORTH",

        }.get((dx, dy))



        if result is None:

            raise ValueError(

                f"Cannot move in one step from {self.from_pos} to {self.pos}"

            )



        return {self.id: result}

class Strategy:

    def __init__(self, *, id):

        self.id: str = id



    def make_plans(self, num) -> List[Plan]:

        """

        Num says how many strategies you need at most. To save time you can limit your strategies to this number.

        """

        raise NotImplementedError()



    def notify_action(self, action: Action) -> None:

        pass





class Ship(Strategy):  # pylint: disable=abstract-method

    @property

    def pos(self):

        return obs.my_ships[self.id].pos



    @property

    def halite(self):

        return obs.my_ships[self.id].halite





class ShipYard(Strategy):  # pylint: disable=abstract-method

    @property

    def pos(self):

        return obs.my_shipyards[self.id].pos
class FirstShip(Ship):

    def make_plans(self, num):

        return [ConvertPlan(id=self.id, pos=self.pos, score=1)]



    



class PlainShipYard(ShipYard):

    def make_plans(self, num):

        if (

            obs.num_ships < NUM_SHIPS

            and obs.my_halite > 500

            and self.pos not in obs.ship_poses

        ):

            return [SpawnPlan(id=self.id, pos=self.pos, score=1)]



        return []

    

class Gatherer(Ship):

    def make_plans(self, num) -> List[MovePlan]:

        plans = []



        if not mine_scores:

            return []



        for end_pos in geometry.poses:

            ship_pos_dist = dist(self.pos, end_pos)



            mine_score = mine_scores[end_pos]



            total_steps = ship_pos_dist + mine_score.steps



            extra_halite = (

                HALITE_MINE_FACTOR * mine_score.halite

                if self.pos == end_pos

                else 0

            )



            trip_score = (

                (self.halite + extra_halite) / (total_steps + 1)

                + HALITE_MINE_FACTOR * mine_score.score

            ) * (TIME_FACTOR if self.pos == end_pos else 1)



            plans.append(

                MovePlan(

                    id=self.id, start_pos=self.pos, end_pos=end_pos, score=trip_score

                )

            )



        plans.sort(key=attrgetter("score"), reverse=True)



        return plans[:num]

    

    

class Hunter(Ship):

    def make_plans(self, num):

        enemy_ship_poses = obs.enemy_ship_poses

        if not enemy_ship_poses:

            return []



        closest_enemy_pos = find_closest(self.pos, enemy_ship_poses)



        return [

            MovePlan(

                id=self.id, start_pos=self.pos, end_pos=closest_enemy_pos.pos, score=1

            )

        ]
class MineScore(NamedTuple):

    """

    Score of a mining position independent of ship

    Used only for precalculations

    """



    score: float  # final value

    halite: float

    steps: int





def update_new_state(new_obs):

    global obs

    

    obs = new_obs

    update_mine_scores()

    update_strategies()





def update_strategies():

    # Ships

    for id_ in obs.my_ships.keys():

        if id_ not in strategies:

            #if obs.step == 0:   # use this with non-buggy kaggle env again

            if obs.step == 1:

                strat_class = FirstShip

            else:

                ship_type_cnts = Counter(

                    strat.__class__.__name__ for strat in strategies.values()

                )



                if ship_type_cnts["Gatherer"] >= MIN_NUM_GATHERER and ship_type_cnts["Hunter"] < NUM_HUNTER:

                    strat_class = Hunter

                else:

                    strat_class = Gatherer



            strategies[id_] = strat_class(id=id_)



    # Shipyards

    for id_ in obs.my_shipyards.keys():

        if id_ not in strategies:

            strategies[id_] = PlainShipYard(id=id_)



    # Delete dead

    for id_ in (

        strategies.keys()

        - obs.my_ships.keys()

        - obs.my_shipyards.keys()

    ):

        del strategies[id_]





def update_mine_scores():

    global mine_scores

    

    mine_scores = {}



    if not obs.my_shipyards:

        return



    for pos in geometry.poses:

        closest = find_closest(pos, obs.shipyard_poses)

        pos_dist = closest.dist



        discount = 1 / (pos_dist + 1)



        halite = obs.map_halite[pos]



        mine_scores[pos] = MineScore(

            score=halite * discount, halite=halite, steps=pos_dist,

        )

strategies: Dict[Id, Strategy] = {}



obs: Observation = None



mine_scores: Dict[Pos, MineScore] = {}



        

def agent(raw_obs) -> Dict[str, str]:

    obs = Observation.from_obs(raw_obs)



    update_new_state(obs)



    assert obs.my_ships.keys() | obs.my_shipyards.keys() == strategies.keys(), (

        obs.my_ships,

        obs.my_shipyards,

        strategies,

    )



    bot_plans = list(

        chain.from_iterable(

            bot.make_plans(num=len(strategies))

            for bot in strategies.values()

        )

    )



    best_plans = solve_assign(bot_plans)  # TODO type



    possible_actions = list(chain.from_iterable(plan.actions for plan in best_plans))



    best_actions = solve_assign(possible_actions)



    for action in best_actions:

        strategies[action.id].notify_action(action)



    halite_actions = {}

    for action in best_actions:

        halite_actions.update(action.halite_command)



    return halite_actions
from kaggle_environments import evaluate, make

import random

import numpy



random.seed(5)



env = make("halite", debug=True)



trainer = env.train([None, "random"])



obs_ = trainer.reset()



while not env.done:

    # dirty fix for ids due to buggy kaggle envs

    player_info = obs_["players"][0]

    if player_info[1]:

        player_info[1] = {key+"sy": val for key, val in player_info[1].items()}



    action = agent(obs_)

    obs_, reward, done, info = trainer.step(action)

    

    

    

env.render(mode="ipython", width=800, height=600)