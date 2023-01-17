#!pip install --upgrade kaggle-environments > /dev/null
%%writefile MyBot.py

__name__ = ""





import itertools

import pprint

import random

import re

import sys

import traceback

from abc import ABC, abstractmethod

from collections import Counter, defaultdict

from dataclasses import dataclass

from operator import attrgetter, itemgetter

from statistics import mean, median

from typing import (

    Any,

    Callable,

    Dict,

    Generic,

    Iterable,

    List,

    Optional,

    Sequence,

    Set,

    Tuple,

    TypeVar,

)



import numpy as np

from kaggle_environments.envs.halite.helpers import (  # https://www.kaggle.com/sam/halite-sdk-overview

    Board,

    Cell,

    Point,

    Ship,

    Shipyard,

)

from scipy.optimize import linear_sum_assignment

from scipy.signal import convolve2d



Id = str





step_action: Dict[Tuple[int, int], str] = {

    (1, 0): "EAST",

    (-1, 0): "WEST",

    (0, 1): "NORTH",

    (0, -1): "SOUTH",

}





PRIO_SCORE_STEP = 10

MAX_RAW_SCORE = 10000

DIST_SCORE_BASE = 0.89  # exponential which roughly fits the optimal mining rate per step if you go to a distant location

TINY_SCORE_STEP = 1e-6





def calc_prio_score(*, prio: int, score: float = 0, dist: int = 0) -> float:

    """

    Use this to create a two level score so that mission priorities are clear



    score should not decrease across different steps to ensure assignment consistency



    Under some assumptions, only an exponential score on distance ensures consistent score order

    across different turns without making the agents swap plans a lot after linear max sum assignment

    """

    if not 0 <= score < MAX_RAW_SCORE:

        raise ValueError(f"Raw score {score} out of bounds")



    return PRIO_SCORE_STEP * prio + score / MAX_RAW_SCORE * DIST_SCORE_BASE ** dist





def prio_add_score(score: float, *, diff_prio: int = 0) -> float:

    """

    Raise or lower the priority score part (e.g. to lower scores for dangerous cells)

    """

    return score + PRIO_SCORE_STEP * diff_prio





def halite_array_from_board(board: Board) -> np.ndarray:  # type: ignore

    """

    Return halite configuration as numpy array

    """

    size_x = board.configuration.size

    size_y = size_x



    result = np.full(shape=(size_x, size_y), fill_value=np.nan)  # type: ignore



    for (x, y), cell in board.cells.items():

        result[x, y] = cell.halite



    return result





def _make_dx_dy_1d(

    x: int, y: int, size_x: int, size_y: int

) -> Tuple[np.ndarray, np.ndarray]:  # type: ignore

    a, b = divmod(size_x, 2)

    x_template = np.r_[  # type: ignore

        : a + b, a:0:-1

    ]  # [0 1 2 1] for size_x == 4 and [0 1 2 2 1] for size_x == 5

    x_template = np.roll(x_template, x)  # type: ignore

    # for x == 2, size_x == 8: [2 1 0 1 2 3 4 3]



    a, b = divmod(size_y, 2)

    y_template = np.r_[: a + b, a:0:-1]  # type: ignore

    y_template = np.roll(y_template, y)  # type: ignore



    return x_template, y_template





def _make_dist_array(x: int, y: int, size_x: int, size_y: int) -> np.ndarray:  # type: ignore

    """

    >>> f(x=2, y=3, size_x=8, size_y=8)

    array([[5, 4, 3, 2, 3, 4, 5, 6],

           [4, 3, 2, 1, 2, 3, 4, 5],

           [3, 2, 1, 0, 1, 2, 3, 4],

           [4, 3, 2, 1, 2, 3, 4, 5],

           [5, 4, 3, 2, 3, 4, 5, 6],

           [6, 5, 4, 3, 4, 5, 6, 7],

           [7, 6, 5, 4, 5, 6, 7, 8],

           [6, 5, 4, 3, 4, 5, 6, 7]])

    >>> f(x=1, y=1, size_x=3, size_y=3)

    array([[2, 1, 2],

           [1, 0, 1],

           [2, 1, 2]])

    """

    x_template, y_template = _make_dx_dy_1d(x, y, size_x, size_y)



    return np.add.outer(x_template, y_template)  # type: ignore





def make_cell_dist_array(cells: Iterable[Cell]) -> np.ndarray:  # type: ignore

    """

    Takes in cells and creates an array of distances to these cells

    This version is when you have very few points

    """

    a_cell = next(iter(cells))

    size_x = size_y = a_cell._board.configuration.size



    dist_arrays = [

        _make_dist_array(cell.position.x, cell.position.y, size_x, size_y)

        for cell in cells

    ]



    all_dist_arrays = np.array(dist_arrays)  # type: ignore



    min_dist = all_dist_arrays.min(axis=0)



    return min_dist





def cell_pos_diff(cell1: Cell, cell2: Cell) -> Tuple[int, int]:

    """

    Calculate shortest distance displacement from `cell1` to `cell2` considering the torus geometry

    """

    dx, dy = cell2.position - cell1.position



    if (dx, dy) == (0, 0):

        return (0, 0)



    size_x = size_y = cell1._board.configuration.size

    half_x = half_y = size_x // 2



    if dx < -half_x:

        dx += size_x

    elif dx > half_x:

        dx -= size_x



    if dy < -half_y:

        dy += size_y

    elif dy > half_y:

        dy -= size_y



    return (dx, dy)





def pos_towards(cell1: Cell, cell2: Cell) -> Set[Point]:

    """

    Calculate points where to go next in order to go from `cell1` towards `cell2`

    """

    if cell1.position == cell2.position:

        return {cell2.position}



    dx, dy = cell_pos_diff(cell1, cell2)



    steps = []



    if dx > 0:

        steps.append((1, 0))



    if dx < 0:

        steps.append((-1, 0))



    if dy > 0:

        steps.append((0, 1))



    if dy < 0:

        steps.append((0, -1))



    return set(cell1.neighbor(step).position for step in steps)





def format_pos(pos: Optional[Point]) -> str:

    """

    Nicer format to use in __repr__ and prints

    """

    if pos is None:

        return "None"



    x, y = pos



    return f"[{x}:{y}]"  # helpful to use the library `colorful` for color print here





def get_prox(cell: Cell, *dists) -> Set[Point]:

    """

    Get points are multiple taxicab distances away from `cell`

    """

    prox_poses = []



    for dist in dists:

        if dist == 0:

            prox_poses.append((0, 0))

            continue

        for d in range(dist):

            prox_poses.append((d, dist - d))

            prox_poses.append((-d, -dist + d))

            prox_poses.append((-dist + d, d))

            prox_poses.append((dist - d, -d))



    return {cell.neighbor(pos).position for pos in prox_poses}





def calc_dist(cell1: Cell, cell2: Cell) -> int:

    """

    Calculate the taxicab distance between two cells

    """

    dx, dy = cell_pos_diff(cell1, cell2)

    return abs(dx) + abs(dy)





def calc_max_dist(cell1: Cell, cell2: Cell) -> int:

    dx, dy = cell_pos_diff(cell1, cell2)

    return max(abs(dx), abs(dy))





def find_closest(cell: Cell, other_cells: Iterable[Cell]) -> Optional[Cell]:

    try:

        return min(other_cells, key=lambda other_cell: calc_dist(cell, other_cell))  # type: ignore

    except ValueError:

        return None





def is_unique(elems: Iterable) -> bool:

    """

    Mainly used in consistency checks

    """

    cnts = Counter(elems)



    if not cnts:

        return True



    return cnts.most_common(1)[0][1] == 1





def _make_dist_array(

    poses: List[Tuple[int, int]], size_x: int, size_y: int, max_dist: int = np.inf

) -> Tuple[Dict[Tuple[int, int], int], Dict[Tuple[int, int], int]]:

    border_pos = [{pos} for pos in poses]



    pos_dists = {pos: 0 for pos in poses}

    pos_idxs = {pos: idx for idx, pos in enumerate(poses)}



    dist = 0



    while any(border_pos) and dist < max_dist:

        dist += 1



        for i, idx_border_pos in enumerate(border_pos):

            new_idx_border_pos = set()



            for x, y in idx_border_pos:

                for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:

                    xy = (x + dx) % size_x, (y + dy) % size_y

                    if xy not in pos_dists:

                        new_idx_border_pos.add(xy)

                        pos_dists[xy] = dist

                        pos_idxs[xy] = i



            border_pos[i] = new_idx_border_pos



    return pos_dists, pos_idxs





def make_dist_array(cells: List[Cell]) -> Optional[np.ndarray]:

    if not cells:

        return None



    size_x = size_y = cells[0]._board.configuration.size



    poses = [(cell.position.x, cell.position.y) for cell in cells]



    pos_dists, pos_idxs = _make_dist_array(

        poses, size_x=size_x, size_y=size_y

    )  # TODO: also return idxs



    result = np.full(shape=(size_x, size_y), fill_value=np.nan)



    for pos, dist in pos_dists.items():

        result[pos] = dist



    return result





class Filter:

    def __init__(self, size: int, func):

        assert size % 2 == 1, size



        if func is not None:

            middle = size // 2

            dists_from_middle = np.array(  # type: ignore

                [

                    [abs(i - middle) + abs(j - middle) for i in range(size)]

                    for j in range(size)

                ]

            )



            self.filter_arr = func(dists_from_middle)

        else:

            self.filter_arr = np.full(shape=(size, size), fill_value=1)  # type: ignore



    def __call__(self, array: np.ndarray):  # type: ignore

        return convolve2d(array, self.filter_arr, boundary="wrap", mode="same")  # type: ignore
%%writefile -a MyBot.py

@dataclass

class Config:

    """

    A class the will hold your configuration parameters



    Configuration that will be passed in the Context to the mission factories

    """



    MAX_NUM_SHIPS: int = 40

    LESS_SHIPS_THAN_ENEMY: int = 5

    MIN_NUM_SHIPS_DESTROYER: int = 3

    NEW_SHIPYARD_DIST: int = 9

    SHIPS_PER_SHIPYARD: int = 6



    # priority of different missions (higher means higher priority)

    PRIO_SPAWN: int = 1

    PRIO_MINE: int = 2

    PRIO_NEW_SHIPYARD: int = 3

    PRIO_EARLY_RETURN: int = 4

    PRIO_EMERGENCY_SHIPYARD: int = 5

    PRIO_FIRST: int = 6

    PRIO_HARASSED_RETURN_HALITE: int = 7

    PRIO_DESTROYER: int = 8

    PRIO_FINAL_CONVERT: int = 9
%%writefile -a MyBot.py

@dataclass

class State:

    """

    It's up to you how to use this. 



    It will be initialized with Board and pre-calculates useful data for the current step

    and will be passed in the Context to the mission factories

    """



    halite_array: np.ndarray  # type: ignore

    enemy_ship_poses: Set[Point]

    enemy_shipyard_poses: Set[Point]

    enemy_halite: Dict[Point, int]

    # enemy_dist_array: np.ndarray

    min_mine_halite: float

    min_spawn_halite: float



    @classmethod

    def from_board(cls, board: Board):

        size_x = size_y = board.configuration.size



        halite_array = halite_array_from_board(board)



        enemy_ships = [

            enemy_ship

            for enemy_ship in board.ships.values()

            if enemy_ship.player_id != board.current_player_id

        ]



        enemy_ship_poses = set(enemy_ship.position for enemy_ship in enemy_ships)



        enemy_shipyard_poses = set(

            enemy_shipyard.position

            for enemy_shipyard in board.shipyards.values()

            if enemy_shipyard.player_id != board.current_player_id

        )



        enemy_halite = {

            ship.position: ship.halite

            for ship in board.ships.values()

            if ship.player_id != board.current_player_id

        }



        # _enemy_dist_array = make_dist_array(

        #     [enemy_ship.cell for enemy_ship in enemy_ships]

        # )



        # if _enemy_dist_array is not None:

        #     enemy_dist_array = _enemy_dist_array

        # else:

        #     enemy_dist_array = np.full(shape=(size_x, size_y), fill_value=np.nan)



        min_mine_halite = np.nanmean(np.where(halite_array > 0, halite_array, np.nan))  # type: ignore



        min_spawn_halite = (

            board.configuration.spawn_cost + board.configuration.convert_cost

        )



        return cls(

            halite_array=halite_array,

            enemy_ship_poses=enemy_ship_poses,

            enemy_shipyard_poses=enemy_shipyard_poses,

            enemy_halite=enemy_halite,

            # enemy_dist_array=enemy_dist_array,

            min_mine_halite=min_mine_halite,

            min_spawn_halite=min_spawn_halite,

        )
%%writefile -a MyBot.py

@dataclass

class Context:

    """

    Gathers all information that will be passed to mission factories

    """



    board: Board

    config: Config

    state: State

    memory: Dict[Any, Any]
%%writefile -a MyBot.py

ObjType = TypeVar("ObjType", Ship, Shipyard)





class IAction(ABC, Generic[ObjType]):

    """

    Base class for actions moving/spawning/converting

    You probably don't need to define more

    """



    def __init__(

        self,

        *,

        obj: ObjType,

        score: float,

        halite_cost: float,

        pos: Optional[Point],

        info: str,

    ):

        self.obj = obj

        self.score = score

        self.pos = pos

        self.halite_cost = halite_cost

        self.info = info



        self.resolve_id_goal_score = (self.obj.id, self.pos, self.score)



    @abstractmethod

    def make_halite_actions(self) -> Dict[Id, str]:

        pass



    def __repr__(self) -> str:

        return f"{self.__class__.__name__}({self.obj.id}->{format_pos(self.pos)}: {self.score:g}; {self.info})"





class MoveAction(IAction[Ship]):

    def __init__(self, *, dest: Cell, ship: Ship, score: float, info: str):

        super().__init__(

            obj=ship, score=score, pos=dest.position, halite_cost=0, info=info

        )

        self.dest = dest



    def make_halite_actions(self) -> Dict[Id, str]:

        step = cell_pos_diff(self.obj.cell, self.dest)



        if step == (0, 0):

            return {}



        halite_action = step_action.get(step)



        if halite_action is None:

            raise ValueError(

                f"Invalid step {step} for move action for id {self.obj.id}"

            )



        return {self.obj.id: halite_action}





class SpawnAction(IAction[Shipyard]):

    def make_halite_actions(self) -> Dict[Id, str]:

        return {self.obj.id: "SPAWN"}





class ConvertAction(IAction[Ship]):

    def make_halite_actions(self) -> Dict[Id, str]:

        return {self.obj.id: "CONVERT"}





class IdleShipyardAction(IAction[Shipyard]):

    """

    To make sure everyone has a proper action assigned

    """



    def make_halite_actions(self) -> Dict[Id, str]:

        return {}
%%writefile -a MyBot.py

class IMission(ABC, Generic[ObjType]):

    base_score: float = 0



    def __init__(

        self,

        *,

        context: Context,

        obj: ObjType,

        score: float,

        resolve_goal: Any,

        limit_num: Optional[Tuple[str, int]] = None,

        info: str = "",

    ):

        """

        resolve_id_goal_score is for the linear_sum_assign resolution.

        Each id and each goal will be assigned only once, such that sum(score) is maximized

        Use goal=None for unique dummy goals



        `info` is for debugging only; mission factories can leave notes, so that you can trace where missions came from

        """

        self.context = context

        self.obj = obj

        self.score = score

        self.limit_num = limit_num

        self.info = info



        self.resolve_id_goal_score = (self.obj.id, resolve_goal, self.score)



    @abstractmethod

    def make_actions(self) -> Sequence[IAction]:

        pass



    def __repr__(self):

        return (

            f"{self.__class__.__name__}({self.obj.id} {self.score:g}"

            + (f"; {self.info}" if self.info else "")

            + ")"

        )
%%writefile -a MyBot.py

class MoveMission(IMission[Ship]):

    def __init__(

        self,

        *,

        context: Context,

        ship: Ship,

        dest: Cell,

        prio_score: int = 0,

        score: float = 0,

        info: str = "",

        limit_num: Optional[Tuple[str, int]] = None,

    ):

        super().__init__(

            context=context,

            obj=ship,

            score=calc_prio_score(

                prio=prio_score, score=score, dist=calc_dist(ship.cell, dest)

            ),

            limit_num=limit_num,

            resolve_goal=dest.position,

            info=info,

        )

        self.dest = dest



    def make_actions(self) -> Sequence[IAction]:

        next_poses = pos_towards(self.obj.cell, self.dest)



        result = []



        board = self.obj.cell._board



        for pos in get_prox(self.obj.cell, 0, 1):

            cell = board[pos]

            if pos in next_poses:

                score = self._rescore_enemy(cell, self.score)

            else:

                raw_score = 0



                if (

                    len(next_poses) == 1 and calc_max_dist(self.obj.cell, cell) == 1

                ):  # prefer going sideways instead of completely opposite

                    raw_score += TINY_SCORE_STEP



                score = self._rescore_enemy(cell, raw_score)



            result.append(

                MoveAction(ship=self.obj, dest=cell, score=score, info=self.info)

            )



        return result



    def _rescore_enemy(self, cell: Cell, score: float) -> float:

        if (

            cell.position in self.context.state.enemy_halite

            and self.context.state.enemy_halite[cell.position] <= self.obj.halite

        ) or cell.position in self.context.state.enemy_shipyard_poses:

            return prio_add_score(score, diff_prio=-10)



        if any(

            self.context.state.enemy_halite[pos] <= self.obj.halite

            for pos in get_prox(cell, 1) & self.context.state.enemy_halite.keys()

        ):

            return prio_add_score(score, diff_prio=-9)



        return score



    def __repr__(self):

        return (

            f"Move({self.obj.id}, {format_pos(self.obj.position)}->{format_pos(self.dest.position)}"

            f": {self.score:g}; {self.info})"

        )
%%writefile -a MyBot.py

class FearlessMoveMission(MoveMission):

    def _rescore_enemy(self, cell: Cell, score: float) -> float:

        return score



    def __repr__(self):

        return f"FearlessMove({self.obj.id}->{format_pos(self.dest.position)}: {self.score:g})"

%%writefile -a MyBot.py

class ConvertMission(IMission[Ship]):

    def __init__(

        self,

        *,

        context: Context,

        ship: Ship,

        prio_score: int = 0,

        score: float = 0,

        resolve_goal=None,

        info: str = "",

    ):

        # Technically it is not checked whether a ship tries to convert on top of another shipyard!

        super().__init__(

            context=context,

            obj=ship,

            score=calc_prio_score(prio=prio_score, score=score),

            resolve_goal=resolve_goal,

            info=info,

        )



    def make_actions(self) -> List[IAction]:

        result: List[IAction] = []



        result.append(

            ConvertAction(

                obj=self.obj,

                score=self.score,

                pos=self.obj.position,

                halite_cost=max(

                    self.context.board.configuration.convert_cost - self.obj.halite, 0

                ),

                info=self.info,

            )

        )



        result.append(

            MoveAction(ship=self.obj, dest=self.obj.cell, score=0, info=self.info)

        )



        return result





class SpawnMission(IMission[Shipyard]):

    def __init__(

        self,

        *,

        context: Context,

        shipyard: Shipyard,

        prio_score: int = 0,

        score: float = 0,

        info: str = "",

        limit_num: Optional[Tuple[str, int]] = None,

    ):

        super().__init__(

            context=context,

            obj=shipyard,

            score=calc_prio_score(prio=prio_score, score=score),

            resolve_goal=shipyard.position,

            info=info,

            limit_num=limit_num,

        )



    def make_actions(self) -> List[IAction]:

        result: List[IAction] = []



        result.append(

            SpawnAction(

                obj=self.obj,

                score=self.score,

                pos=self.obj.position,

                halite_cost=self.context.board.configuration.spawn_cost,

                info=self.info,

            )

        )



        result.append(

            IdleShipyardAction(

                obj=self.obj, score=0, halite_cost=0, pos=None, info=self.info

            )

        )



        return result





class IdleShipMission(IMission[Ship]):

    """

    Mainly a fallback, when ships do not get any other mission assigned

    """



    def __init__(self, *, context: Context, ship: Ship):

        super().__init__(context=context, obj=ship, score=-1e-4, resolve_goal=None)



    def make_actions(self) -> Sequence[IAction]:

        return [

            MoveAction(

                ship=self.obj, dest=self.obj.cell, score=self.score, info=self.info

            )

        ]





class IdleShipyardMission(IMission[Shipyard]):

    def __init__(self, *, context: Context, shipyard: Shipyard, info: str = ""):

        super().__init__(

            context=context, obj=shipyard, score=-1e-4, resolve_goal=None, info=info

        )



    def make_actions(self) -> List[IAction]:

        return [

            IdleShipyardAction(

                obj=self.obj, score=self.score, halite_cost=0, pos=None, info=self.info

            )

        ]
%%writefile -a MyBot.py



IMissionFactory = Callable[[Context], Sequence[IMission]]





def random_ship_mission(ctx: Context) -> Sequence[IMission]:

    """

    Just an example. Not used here.

    """

    result = []



    for ship in ctx.board.current_player.ships:

        step = random.choice([Point(0, 1), Point(0, -1), Point(1, 0), Point(-1, 0)])

        dest = ship.cell.neighbor(step)

        result.append(MoveMission(context=ctx, ship=ship, dest=dest, info="Random"))



    return result
%%writefile -a MyBot.py

def mining_mission(ctx: Context) -> Sequence[IMission]:

    """

    Mine or return halite

    """

    result = []



    player = ctx.board.current_player

    for ship in player.ships:

        if any(

            ctx.state.enemy_halite[pos] <= ship.halite

            for pos in get_prox(ship.cell, 1) & ctx.state.enemy_halite.keys()

        ):  # being harassed

            result.extend(

                return_halite(

                    ctx,

                    ship,

                    ctx.config.PRIO_HARASSED_RETURN_HALITE,

                    info="Mine(harassed,return {})",

                )

            )

            continue



        mine_scores: np.ndarray = ctx.state.halite_array



        for shipyard in player.shipyards:

            if shipyard.position != ship.position:

                mine_scores[shipyard.position] = ship.halite

        # This overrides the score for shipyard positions

        # It needs adjustment



        for pos, score in np.ndenumerate(mine_scores):  # type: ignore

            if (

                np.isnan(score)  # type: ignore

                or score == 0

                or ctx.state.halite_array[pos] < ctx.state.min_mine_halite

                or any(

                    ctx.board[prox_enemy_pos].ship.halite <= ship.halite

                    for prox_enemy_pos in get_prox(ctx.board[pos], 0, 1)

                    & ctx.state.enemy_ship_poses

                )

            ):

                continue



            result.append(

                MoveMission(

                    context=ctx,

                    ship=ship,

                    dest=ctx.board[pos],

                    prio_score=ctx.config.PRIO_MINE,

                    score=score / 100,

                    info=f"Mine {format_pos(pos)}",

                )

            )



    return result





def return_halite(

    ctx: Context,

    ship: Ship,

    prio_score: int = 0,

    score: float = 0,

    limit_num: Optional[Tuple[str, int]] = None,

    info: str = "Return halite {}",

) -> Sequence[IMission[Ship]]:

    """

    Info should contain "{}" which will be filled with the destination position

    """

    player = ctx.board.current_player

    if not player.shipyards:

        return []



    closest_shipyard_cell = find_closest(

        ship.cell, [shipyard.cell for shipyard in player.shipyards]

    )



    assert closest_shipyard_cell is not None



    return [

        MoveMission(

            context=ctx,

            ship=ship,

            dest=closest_shipyard_cell,

            prio_score=prio_score,

            score=score,

            limit_num=limit_num,

            info=info.format(format_pos(closest_shipyard_cell.position)),

        )

    ]
%%writefile -a MyBot.py

def early_return(ctx: Context) -> Sequence[IMission[Ship]]:

    player = ctx.board.current_player



    min_spawn_halite = ctx.state.min_spawn_halite



    if not player.shipyards or player.halite >= min_spawn_halite:

        return []



    result: List[IMission[Ship]] = []



    for ship in player.ships:

        if ship.halite + player.halite >= min_spawn_halite:

            result.extend(

                return_halite(

                    ctx=ctx,

                    ship=ship,

                    prio_score=ctx.config.PRIO_EARLY_RETURN,

                    limit_num=("early_return", 1),

                    info="Early return {}",

                )

            )



    return result
%%writefile -a MyBot.py

halite_smooth_filter = smooth_filter = Filter(size=5, func=None)





def make_new_shipyard(ctx: Context) -> Sequence[IMission]:

    player = ctx.board.current_player



    if (

        not player.shipyards

        or len(player.shipyards)

        >= len(player.ships) // ctx.config.SHIPS_PER_SHIPYARD + 1

    ):

        return []



    shipyard_cells = [shipyard.cell for shipyard in player.shipyards]



    distant_ships = [

        ship

        for ship in player.ships

        if calc_dist(ship.cell, find_closest(ship.cell, shipyard_cells))  # type: ignore

        >= ctx.config.NEW_SHIPYARD_DIST

    ]



    if not distant_ships:

        return []



    shipyard_border_poses = set(

        itertools.chain.from_iterable(

            get_prox(shipyard_cell, ctx.config.NEW_SHIPYARD_DIST)

            for shipyard_cell in shipyard_cells

        )

    )



    shipyard_border_poses = set(

        border_pos

        for border_pos in shipyard_border_poses

        if min(

            calc_dist(ctx.board[border_pos], shipyard_cell)

            for shipyard_cell in shipyard_cells

        )

        >= ctx.config.NEW_SHIPYARD_DIST

    )



    if not shipyard_border_poses:

        return []



    smoothed_halite_array = halite_smooth_filter(ctx.state.halite_array)



    smoothed_halite_at_border = [

        smoothed_halite_array[pos] for pos in shipyard_border_poses

    ]



    min_convert_halite = mean(smoothed_halite_at_border)



    result: List[IMission] = []



    for ship in distant_ships:

        if smoothed_halite_array[ship.position] < min_convert_halite:

            continue



        result.append(

            ConvertMission(

                context=ctx,

                ship=ship,

                prio_score=ctx.config.PRIO_NEW_SHIPYARD,

                score=ship.halite / 100,

                resolve_goal="new_shipyard",

                info="New shipyard",

            )

        )



    return result
%%writefile -a MyBot.py

def destroyer(ctx: Context) -> Sequence[IMission]:

    """

    Target an enemy shipyard with an empty own ship

    If all ships are full, make one ship return

    """

    player = ctx.board.current_player



    if (

        not ctx.state.enemy_shipyard_poses

        or len(player.ships) < ctx.config.MIN_NUM_SHIPS_DESTROYER

    ):

        return []



    enemy_shipyard_cells = [ctx.board[pos] for pos in ctx.state.enemy_shipyard_poses]



    result: List[IMission[Ship]] = []



    empty_ships = [ship for ship in player.ships if ship.halite == 0]



    if empty_ships:

        for ship in empty_ships:

            closest_enemy_shipyard_cell = find_closest(ship.cell, enemy_shipyard_cells)



            assert closest_enemy_shipyard_cell is not None



            result.append(

                FearlessMoveMission(

                    context=ctx,

                    ship=ship,

                    dest=closest_enemy_shipyard_cell,

                    prio_score=ctx.config.PRIO_DESTROYER,

                    info=f"Destroyer(attack {format_pos(closest_enemy_shipyard_cell.position)})",

                )

            )

    elif player.shipyards:

        shipyard_cells = [shipyard.cell for shipyard in player.shipyards]

        for ship in player.ships:

            closest_shipyard_cell = find_closest(ship.cell, shipyard_cells)

            assert closest_shipyard_cell is not None



            result.append(

                MoveMission(

                    context=ctx,

                    ship=ship,

                    dest=closest_shipyard_cell,

                    prio_score=ctx.config.PRIO_DESTROYER,

                    score=ship.halite / 100,

                    info=f"Destroyer(return {format_pos(closest_shipyard_cell.position)})",

                )

            )



    return result

%%writefile -a MyBot.py

def final_convert_mission(ctx: Context) -> Sequence[IMission]:

    """

    Convert all ships with sufficient halite on final step

    """

    if ctx.board.step < ctx.board.configuration.episode_steps - 2:

        return []



    result: List[IMission[Ship]] = []



    for ship in ctx.board.current_player.ships:

        if ship.halite > ctx.board.configuration.convert_cost:

            result.append(

                ConvertMission(

                    context=ctx,

                    ship=ship,

                    prio_score=ctx.config.PRIO_FINAL_CONVERT,

                    info="Final convert",

                )

            )



    return result
%%writefile -a MyBot.py

def emergency_shipyard_mission(ctx: Context) -> Sequence[IMission]:

    """

    Immediately create a shipyard if no more shipyards left

    """

    player = ctx.board.current_player



    if (

        ctx.board.step == 0

        or len(player.shipyards) > 0

        or (

            len(player.ships) == 1

            and player.halite

            + player.ships[0].halite

            - ctx.board.configuration.convert_cost

            < ctx.board.configuration.spawn_cost

        )

    ):

        return []



    ships = [ship for ship in player.ships]



    convertable_ships = [

        ship

        for ship in ships

        if ship.halite + player.halite >= ctx.board.configuration.convert_cost

    ]



    if not convertable_ships:  # bad luck

        print("!!! Emergency convert: No convertable ships")

        return []



    if len(convertable_ships) == 1:

        return [

            ConvertMission(

                context=ctx,

                ship=convertable_ships[0],

                prio_score=ctx.config.PRIO_EMERGENCY_SHIPYARD,

            )

        ]



    result: List[IMission[Ship]] = []



    for ship in convertable_ships:

        result.append(

            ConvertMission(

                context=ctx,

                ship=ship,

                prio_score=ctx.config.PRIO_EMERGENCY_SHIPYARD,

                score=mean(

                    calc_dist(ship.cell, other_ship.cell)

                    for other_ship in ships

                    if other_ship != ship

                ),

                resolve_goal="emergency_convert",

                info="Emergency convert",

            )

        )



    return result

%%writefile -a MyBot.py

def first_ship_mission(ctx: Context) -> Sequence[IMission]:

    """

    What to do for the first ship

    """

    if ctx.board.step > 0:

        return []



    return [

        ConvertMission(

            context=ctx, ship=ship, prio_score=ctx.config.PRIO_FIRST, info="First ship"

        )

        for ship in ctx.board.current_player.ships

    ]
%%writefile -a MyBot.py

def shipyard_mission(ctx: Context) -> Sequence[IMission]:

    """

    Plain shipyard keeps spawning until some number of ships is reached

    """

    target_num_ships = max(

        ctx.config.MAX_NUM_SHIPS,

        len(ctx.state.enemy_ship_poses)

        - ctx.config.LESS_SHIPS_THAN_ENEMY,  # careful! uses number of all enemy ships; only for 1v1

    )



    player = ctx.board.current_player



    if len(player.ships) >= target_num_ships or (

        len(player.ships) > 0 and player.halite < ctx.state.min_spawn_halite

    ):

        return [

            IdleShipyardMission(context=ctx, shipyard=shipyard, info="Idle shipyard")

            for shipyard in ctx.board.current_player.shipyards

        ]



    num_ships_to_spawn = target_num_ships - len(player.ships)

    assert num_ships_to_spawn > 0



    return [

        SpawnMission(

            context=ctx,

            shipyard=shipyard,

            prio_score=ctx.config.PRIO_SPAWN,

            info="Plain spawn",

            limit_num=("spawn", num_ships_to_spawn),

        )

        for shipyard in ctx.board.current_player.shipyards

    ]
%%writefile -a MyBot.py

def idle_mission(ctx) -> Sequence[IMission]:

    """

    Fall back mission where shipyards do nothing and ships stand still with score 0

    Needed to guarantee resolution if no other mission gets chosen

    """

    player = ctx.board.current_player



    result: List[IMission] = []



    result.extend(

        MoveMission(context=ctx, ship=ship, dest=ctx.board[pos], info="Fallback idle")

        for ship in player.ships

        for pos in get_prox(ship.cell, 0, 1)

    )



    result.extend(

        IdleShipyardMission(context=ctx, shipyard=shipyard, info="Fallback idle")

        for shipyard in player.shipyards

    )



    return result
%%writefile -a MyBot.py

def solve_assign(obj_goal_penalties: List[Tuple[Any, Any, float]]) -> List[int]:

    if not obj_goal_penalties:

        return []



    # Rewrite non-conflict goals

    non_conflict_goals = ((object(), i) for i in itertools.count())  # dummy goals



    obj_goal_penalties = [

        (obj, goal if goal is not None else next(non_conflict_goals), score)

        for obj, goal, score in obj_goal_penalties

    ]



    # groupby by obj/goal to select only best scores for each obj/goal

    all_to_resolve_objs = defaultdict(list)



    for idx, obj_goal_penalty in enumerate(obj_goal_penalties):

        all_to_resolve_objs[obj_goal_penalty[:2]].append((obj_goal_penalty, idx))



    # from those with same obj/goal pick only the best

    best_to_resolve_objs = list(

        max(objs, key=lambda x: x[0][2]) for objs in all_to_resolve_objs.values()

    )



    best_obj_goal_penalties, best_penalty_objs = zip(*best_to_resolve_objs)



    matrix, obj_goal_penalty_map = _make_matrix(best_obj_goal_penalties)



    try:

        x_idxs, y_idxs = linear_sum_assignment(matrix, maximize=True)

    except ValueError as exc:

        goal_obj = defaultdict(list)

        for obj, goal, _ in best_obj_goal_penalties:

            goal_obj[goal].append(obj)

        raise ValueError(

            f"Assign solver failed with {exc} for {obj_goal_penalties}. "

            f"You may need to add actions which guarantee resolution, like letting bots stay on the spot.\n"

            f"Goal to obj:\n{pprint.pformat(goal_obj)}"

        )



    try:

        result = [

            best_penalty_objs[obj_goal_penalty_map[x_idx, y_idx]]

            for x_idx, y_idx in zip(x_idxs, y_idxs)

        ]

    except KeyError as exc:

        raise ValueError(

            f"Assignment solution could not be resolved for {exc}. "

            "You may need to add a stay on the spot move to the bot. Objects were:\n"

            + "\n".join(map(str, obj_goal_penalties))

        )



    assert is_unique(x_idxs), obj_goal_penalties

    assert is_unique(y_idxs), obj_goal_penalties



    return result



def _make_matrix(obj_goal_penalties):

    assert is_unique(obj[:2] for obj in obj_goal_penalties)



    xs = {el: i for i, el in enumerate(set(x[0] for x in obj_goal_penalties))}

    ys = {el: i for i, el in enumerate(set(x[1] for x in obj_goal_penalties))}



    result = np.full(shape=(len(xs), len(ys)), fill_value=np.NINF)  # type: ignore



    obj_goal_penalty_map = {}



    for i, (x, y, penalty) in enumerate(obj_goal_penalties):

        x_idx = xs[x]

        y_idx = ys[y]



        obj_goal_penalty_map[x_idx, y_idx] = i



        result[x_idx, y_idx] = penalty



    return result, obj_goal_penalty_map
%%writefile -a MyBot.py

def solve_max_sum(cost_scores: List[Tuple[float, float]], max_sum: float) -> List[int]:

    result = []



    idx_cost_scores = list(enumerate(cost_scores))



    with_costs = []



    for idx, (cost, score) in idx_cost_scores:

        assert cost >= 0

        if cost > 0:

            with_costs.append((idx, cost, score / cost))

        else:

            result.append(idx)



    # Simple greedy method

    sum_available = max_sum



    for idx, cost, _ in sorted(

        with_costs, key=itemgetter(2), reverse=True

    ):  # TODO do proper knapsack instead of greedy!

        if cost > sum_available:

            continue



        sum_available -= cost

        result.append(idx)



    sum_cost = sum(cost_scores[idx][0] for idx in result)

    assert sum_cost <= max_sum, (sum_cost, result)



    return result
%%writefile -a MyBot.py

class Agent:

    def __init__(self, config: Config, mission_factories: List[IMissionFactory]):

        self.config = config

        self.mission_factories = mission_factories

        self.memory: Dict[Any, Any] = {}



    def __call__(

        self, obs: Dict[str, Any], configuration: Dict[str, Any]

    ) -> Dict[Id, str]:

        try:

            # easier to re-run agent on downloaded replays; note that this might influence other bots too

            # random.seed(123)

            # np.random.seed(123)  # type: ignore

            board = Board(obs, configuration)



            print(f"Step {board.step}, Player Halite: {board.current_player.halite}")



            state = State.from_board(board)



            context = Context(

                board=board, config=self.config, state=state, memory=self.memory

            )



            missions = self.make_missions(board, context)



            num_limited_missions = self.num_limit_missions(missions)



            resolved_missions = self.resolve_missions(num_limited_missions)



            self.validate_final_missions(resolved_missions, context)



            actions = self.make_actions(resolved_missions)



            affordable_actions = self.select_affordable_actions(actions, context)



            resolved_actions = self.resolve_actions(affordable_actions)



            self.validate_final_actions(resolved_actions, context)



            halite_actions = self.make_halite_actions(resolved_actions)



            self.print_state(

                board=board,

                num_limited_missions=num_limited_missions,

                resolved_missions=resolved_missions,

                resolved_actions=resolved_actions,

            )



            return halite_actions

        except Exception:

            traceback.print_exc(

                file=sys.stderr

            )  # you can download logs from Kaggle and inspect what went wrong

            raise



    def print_state(

        self,

        *,

        board: Board,

        num_limited_missions: List[IMission],

        resolved_missions: List[IMission],

        resolved_actions: List[IAction],

    ) -> None:

        player = board.current_player



        print(

            "Resolved Missions: ",

            ", ".join(

                f"{mission_info}:{cnt}"

                for mission_info, cnt in Counter(

                    re.sub(r" *\[.+:.+\] *", "", mission.info)

                    if mission.info

                    else "?mission"

                    for mission in resolved_missions

                    if mission.info != "Idle shipyard"

                ).most_common()

            ),

        )



        print(f"{len(player.ships)} Ships, {len(player.shipyards)} Shipyards:")

        for action in sorted(resolved_actions, key=lambda a: a.obj.id):

            obj = action.obj

            best_obj_missions = sorted(

                [mission for mission in num_limited_missions if mission.obj == obj],

                key=attrgetter("score"),

                reverse=True,

            )[:3]



            if isinstance(obj, Ship):

                print(

                    f"Ship {obj.id} {format_pos(obj.position)} h{obj.halite}: {action} from {' '.join(map(str, best_obj_missions))}"

                )

            else:

                print(

                    f"Shipyard {obj.id} {format_pos(obj.position)}: {action} from {' '.join(map(str, best_obj_missions))}"

                )



        print("---")



    def make_missions(self, board: Board, context: Context) -> List[IMission]:

        missions: List[IMission] = []



        for mission_factory in self.mission_factories:

            new_missions = mission_factory(context)

            missions.extend(new_missions)



        return missions



    def num_limit_missions(self, missions: List[IMission]) -> List[IMission]:

        """

        Limits the number of missions with identical mission.limit_num

        to mission.limit_num[1] top scores of all missions

        """

        result = []

        limit_num_missions = defaultdict(list)



        for mission in missions:

            if mission.limit_num is not None:

                limit_num_missions[mission.limit_num].append(mission)

            else:

                result.append(mission)



        for (name, max_num), group_missions in limit_num_missions.items():

            if max_num < 0:

                raise ValueError(f"limit_num {name} had invalid max_num {max_num}")



            result.extend(

                sorted(group_missions, key=attrgetter("score"), reverse=True)[:max_num]

            )



        return result



    def resolve_missions(self, missions: List[IMission]) -> List[IMission]:

        resolution_idxs = solve_assign(

            [mission.resolve_id_goal_score for mission in missions]

        )



        result = [missions[idx] for idx in resolution_idxs]



        return result



    def validate_final_missions(

        self, missions: List[IMission], context: Context

    ) -> None:

        player = context.board.current_player

        cur_ids = set(ship.id for ship in player.ships) | set(

            shipyard.id for shipyard in player.shipyards

        )

        mission_ids = [mission.obj.id for mission in missions]



        missing_ids = cur_ids - set(mission_ids)

        if missing_ids:

            raise ValueError(

                f"Missing missions for ids: {', '.join(map(str, sorted(missing_ids)))}. You may need let ships stay on the spot to guarantee resolution."

            )



    def make_actions(self, missions: List[IMission]) -> List[IAction]:

        return list(

            itertools.chain.from_iterable(

                mission.make_actions() for mission in missions

            )

        )



    def select_affordable_actions(

        self, actions: List[IAction], context: Context

    ) -> List[IAction]:

        affordable_idxs = solve_max_sum(

            [(action.halite_cost, action.score) for action in actions],

            max_sum=context.board.current_player.halite,

        )



        result = [actions[idx] for idx in affordable_idxs]



        return result



    def resolve_actions(self, actions: List[IAction]) -> List[IAction]:

        resolution_idxs = solve_assign(

            [action.resolve_id_goal_score for action in actions]

        )



        result = [actions[idx] for idx in resolution_idxs]



        return result



    def validate_final_actions(self, actions: List[IAction], context: Context) -> None:

        player = context.board.current_player

        cur_ids = set(ship.id for ship in player.ships) | set(

            shipyard.id for shipyard in player.shipyards

        )

        action_ids = [action.obj.id for action in actions]



        if not is_unique(action_ids):

            raise ValueError(

                f"Non unique final action ids: {', '.join(f'{id}:{cnt}' for id, cnt in Counter(action_ids).most_common() if cnt > 1)}"

            )



        missing_ids = cur_ids - set(action_ids)

        if missing_ids:

            raise ValueError(

                f"Missing actions for ids: {', '.join(map(str, sorted(missing_ids)))}. You may need let ships stay on the spot to guarantee resolution."

            )



    def make_halite_actions(self, actions: List[IAction]) -> Dict[str, str]:

        result = {}

        for action in actions:

            result.update(action.make_halite_actions())

        return result

%%writefile -a MyBot.py

config = Config()

agent = Agent(

    config,

    [

        first_ship_mission,

        shipyard_mission,

        mining_mission,

        emergency_shipyard_mission,

        destroyer,

        final_convert_mission,

        make_new_shipyard,

        early_return,

        idle_mission,

    ],

)  # better always add idle_mission (with score 0) to guarantee resolution



def call_agent(obs, conf):

    return agent(obs, conf)
from kaggle_environments import make

from tqdm.auto import tqdm

from MyBot import call_agent



env = make("halite", configuration={"episodeSteps": 400})



env.reset()

trainer = env.train([None, call_agent])

obs = trainer.reset()



for _ in tqdm(range(env.configuration.episodeSteps - 1)):

    if env.done:

        break

    action = call_agent(obs, env.configuration)

    obs, reward, done, info = trainer.step(action)
env.render(mode="ipython", width=800, height=600)