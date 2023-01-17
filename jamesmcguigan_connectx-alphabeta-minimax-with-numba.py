%%writefile submission.py
%%writefile submission.py

#!/usr/bin/env python3

from __future__ import annotations



##### 

##### ./kaggle_compile.py games/connectx/agents/AlphaBetaAgent.py

##### 

##### 2020-06-24 12:38:03+01:00

##### 

##### ai-pacman	git@github.com:JamesMcGuigan/ai-pacman.git (fetch)

##### ai-pacman	git@github.com:JamesMcGuigan/ai-pacman.git (push)

##### ecosystem-research	git@github.com:JamesMcGuigan/ecosystem-research.git (fetch)

##### ecosystem-research	git@github.com:JamesMcGuigan/ecosystem-research.git (push)

##### kaggle-arc	git@github.com:JamesMcGuigan/kaggle-arc.git (fetch)

##### kaggle-arc	git@github.com:JamesMcGuigan/kaggle-arc.git (push)

##### origin	git@github.com:JamesMcGuigan/ai-games.git (fetch)

##### origin	git@github.com:JamesMcGuigan/ai-games.git (push)

##### udacity-artificial-intelligence	https://github.com/JamesMcGuigan/udacity-artificial-intelligence (fetch)

##### udacity-artificial-intelligence	https://github.com/JamesMcGuigan/udacity-artificial-intelligence (push)

##### 

#####   kaggle-arc        583a4c1 kaggle_compile.py -> submission.py + submission.csv

#####   knights-isolation 27d0916 3_Adversarial Search | udacity submit requires kaggle_compile.py

#####   master            f296c88 refactor: AlphaBetaAgent.agent -> AlphaBetaAgent.agent(kwargs)(observation, configuration)

#####   n-queens          ddb0d49 prolog | nqueens | print solution to nqueens.txt

##### * numba             88fa481 [ahead 3] ConnectX | AlphaBetaAgent | exit loop via raise TimeoutError

##### 

##### 88fa481f1803f20c88654ce21c5ce5dcea14aa95

##### 



#####

##### START util/tuplize.py

#####



from struct import Struct



import numpy as np







# noinspection PyTypeChecker

def tuplize(value):

    """

    Recursively cast to an immutable tuple that can be hashed



    >>> tuplize([])

    ()

    >>> tuplize({"a": 1})

    (('a', 1),)

    >>> tuplize({"a": 1})

    (('a', 1),)

    >>> tuplize(np.array([1,2,3]))

    (1, 2, 3)

    >>> tuplize(np.array([[1,2,3],[4,5,6]]))

    ((1, 2, 3), (4, 5, 6))

    >>> tuplize('string')

    'string'

    >>> tuplize(42)

    42

    """

    if isinstance(value, (list,tuple,set)): return tuple(tuplize(v) for v in value)

    if isinstance(value, np.ndarray):

        if len(value.shape) == 1: return tuple(value.tolist())

        else:                     return tuple(tuplize(v) for v in value.tolist())

    if isinstance(value, (dict,Struct)):    return tuple(tuplize(v) for v in value.items())

    return value





if __name__ == '__main__':

    # python3 -m doctest -v util/tuplize.py

    import doctest

    doctest.testmod()





#####

##### END   util/tuplize.py

#####



#####

##### START util/vendor/cached_property.py

#####



# -*- coding: utf-8 -*-

# Source: https://github.com/pydanny/cached-property/blob/master/cached_property.py



# Simplified function for performance

class cached_property(object):

    """

    A property that is only computed once per instance and then replaces itself

    with an ordinary attribute. Deleting the attribute resets the property.

    Source: https://github.com/bottlepy/bottle/commit/fa7733e075da0d790d809aa3d2f53071897e6f76

    """



    def __init__(self, func):

        self.__doc__ = getattr(func, "__doc__")

        self.func = func

        self.func_name = self.func.__name__



    def __get__(self, obj, cls):

        if obj is None: return self

        value = obj.__dict__[self.func_name] = self.func(obj)

        return value





#####

##### END   util/vendor/cached_property.py

#####



#####

##### START games/connectx/core/KaggleGame.py

#####





import time

from struct import Struct

from typing import Dict

from typing import List



# from util.tuplize import tuplize

# from util.vendor.cached_property import cached_property







class KaggleGame:

    """

    This is a generic baseclass wrapper around kaggle_environments

    def agent(observation, configuration):

        game = KaggleGame(observation, configuration)

        return random.choice(game.actions())

    """



    def __init__(self, observation, configuration, heuristic_class, verbose=True):

        self.time_start    = time.perf_counter()

        self.observation   = observation

        self.configuration = configuration

        self.verbose       = verbose

        self.player_id     = None

        self._hash         = None

        self.heuristic_class = heuristic_class





    def __hash__(self):

        """Return an id for caching purposes """

        self._hash = self._hash or hash(tuplize((self.observation, self.configuration)))

        # self._hash = self._hash or self.board.tobytes()

        return self._hash





    ### Result Methods



    @cached_property

    def actions( self ) -> List:

        """Return a list of valid actions"""

        raise NotImplementedError



    def result( self, action ) -> KaggleGame:

        """This returns the next KaggleGame after applying action"""

        observation = self.result_observation(self.observation, action)

        return self.__class__(observation, self.configuration, self.verbose)



    def result_observation( self, observation: Struct, action ) -> Dict:

        """This returns the next observation after applying action"""

        raise  NotImplementedError

        # return copy(self.observation)







    ### Heuristic Methods



    @cached_property

    def heuristic(self):

        """Delay resolution until after parentclass constructor has finished"""

        return self.heuristic_class(self)



    @cached_property

    def gameover( self ) -> bool:

        """Has the game reached a terminal game?"""

        if self.heuristic:

            return self.heuristic.gameover

        else:

            return len( self.actions ) == 0



    @cached_property

    def score( self ) -> float:

        return self.heuristic.score



    @cached_property

    def utility( self ) -> float:

        return self.heuristic.utility





#####

##### END   games/connectx/core/KaggleGame.py

#####



#####

##### START games/connectx/core/PersistentCacheAgent.py

#####



import atexit

import gzip

import math

import os

import pickle

import time

import zlib







class PersistentCacheAgent:

    persist = False

    cache   = {}

    verbose = True



    def __new__(cls, *args, **kwargs):

        # noinspection PyUnresolvedReferences

        for parentclass in cls.__mro__:  # https://stackoverflow.com/questions/2611892/how-to-get-the-parents-of-a-python-class

            if cls is parentclass: continue

            if cls.cache is getattr(parentclass, 'cache', None):

                cls.cache = {}  # create a new cls.cache for each class

                break

        instance = object.__new__(cls)

        return instance



    def __init__(self, *args, **kwargs):

        super().__init__()

        if not self.persist: return  # disable persistent caching

        self.load()

        self.autosave()



    def autosave( self ):

        # Autosave on Ctrl-C

        atexit.unregister(self.__class__.save)

        atexit.register(self.__class__.save)



    # def __del__(self):

    #     self.save()



    @classmethod

    def filename( cls ):

        return './.cache/' + cls.__name__ + '.zip.pickle'



    @classmethod

    def load( cls ):

        if not cls.persist: return  # disable persistent caching

        if cls.cache:       return  # skip loading if the file is already in class memory

        try:

            # class cache may be more upto date than the pickle file, so avoid race conditions with multiple instances

            filename   = cls.filename()

            start_time = time.perf_counter()

            with gzip.GzipFile(filename, 'rb') as file:  # reduce filesystem cache_size

                # print("loading: "+cls.file )

                data = pickle.load(file)

                cls.cache.update({ **data, **cls.cache })

                if cls.verbose:

                    print("loaded: {:40s} | {:4.1f}MB in {:4.1f}s | entries: {}".format(

                        filename,

                        os.path.getsize(filename)/1024/1024,

                        time.perf_counter() - start_time,

                        cls.cache_size(cls.cache),

                    ))

        except (IOError, TypeError, EOFError, zlib.error) as exception:

            pass



    @classmethod

    def save( cls ):

        if not cls.persist: return  # disable persistent caching

        # cls.load()                # update any new information from the file

        if cls.cache:

            filename = cls.filename()

            dirname  = os.path.dirname(filename)

            if not os.path.exists(dirname): os.mkdir(dirname)

            start_time = time.perf_counter()

            # print("saving: " + filename )

            with gzip.GzipFile(filename, 'wb') as file:  # reduce filesystem cache_size

                pickle.dump(cls.cache, file)

                if cls.verbose:

                    print("wrote:  {:40s} | {:4.1f}MB in {:4.1f}s | entries: {}".format(

                        filename,

                        os.path.getsize(filename)/1024/1024,

                        time.perf_counter() - start_time,

                        cls.cache_size(cls.cache),

                    ))



    @staticmethod

    def cache_size( data ):

        return sum([

            len(value) if isinstance(key, str) and isinstance(value, dict) else 1

            for key, value in data.items()

        ])



    @classmethod

    def reset( cls ):

        cls.cache = {}

        cls.save()





    ### Caching

    @classmethod

    def cache_function( cls, function, game, player_id, *args, **kwargs ):

        hash = (player_id, game)  # QUESTION: is player_id required for correct caching between games?

        if function.__name__ not in cls.cache:   cls.cache[function.__name__] = {}

        if hash in cls.cache[function.__name__]: return cls.cache[function.__name__][hash]



        score = function(game, *args, **kwargs)

        cls.cache[function.__name__][hash] = score

        return score



    @classmethod

    def cache_infinite( cls, function, game, player_id, *args, **kwargs ):

        # Don't cache heuristic values, only terminal states

        hash = (player_id, game)  # QUESTION: is player_id required for correct caching between games?

        if function.__name__ not in cls.cache:   cls.cache[function.__name__] = {}

        if hash in cls.cache[function.__name__]: return cls.cache[function.__name__][hash]



        score = function(game, player_id, *args, **kwargs)

        if abs(score) == math.inf: cls.cache[function.__name__][hash] = score

        return score





#####

##### END   games/connectx/core/PersistentCacheAgent.py

#####



#####

##### START games/connectx/core/ConnectX.py

#####



from copy import copy

from struct import Struct

from typing import Callable

from typing import List

from typing import Tuple

from typing import Union



import numpy as np



# from games.connectx.core.KaggleGame import KaggleGame

# from util.vendor.cached_property import cached_property







class ConnectX(KaggleGame):

    players = 2



    # observation   = {'mark': 1, 'board': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]}

    # configuration = {'columns': 7, 'rows': 6, 'inarow': 4, 'steps': 1000, 'timeout': 2}

    def __init__( self, observation, configuration, heuristic_class: Callable=None, verbose=True, **kwargs ):

        super().__init__(observation, configuration, heuristic_class, verbose)

        self.rows:      int = configuration.rows

        self.columns:   int = configuration.columns

        self.inarow:    int = configuration.inarow

        self.timeout:   int = configuration.timeout

        self.player_id: int = observation.mark

        self.board: np.ndarray = self.cast_board(observation.board)  # Don't modify observation.board

        self.board.setflags(write=False)  # WARN: https://stackoverflow.com/questions/5541324/immutable-numpy-array#comment109695639_5541452





    ### Magic Methods



    def __hash__(self):

        return hash(self.board.tobytes())



    def __eq__(self, other):

        if not isinstance(other, self.__class__): return False

        return self.board.tobytes() == other.board.tobytes()





    ### Utility Methods



    def cast_board( self, board: Union[np.ndarray,List[int]], copy=False ) -> np.ndarray:

        if isinstance(board, np.ndarray):

            if copy: return board.copy()

            else:    return board

        else:

            board = np.array(board, dtype=np.int8).reshape(self.rows, self.columns)

            return board









    ### Result Methods



    def result( self, action ) -> 'ConnectX':

        """This returns the next KaggleGame after applying action"""

        if not hasattr(self, '_results_cache'): self._results_cache = {}

        if action not in self._results_cache:

            observation = self.result_observation(self.observation, action)

            result      = self.__class__(observation, self.configuration, self.heuristic_class, self.verbose)

            self._results_cache[action] = result

        return self._results_cache[action]



    def result_observation( self, observation: Struct, action: int ) -> Struct:

        output = copy(observation)

        output.board = self.result_board(observation.board, action, observation.mark)

        output.mark  = 2 if observation.mark == 1 else 1

        return output



    def result_board( self, board: np.ndarray, action: int, mark: int ) -> np.ndarray:

        """This returns the next observation after applying an action"""

        next_board = self.cast_board(board, copy=True)

        coords     = self.get_coords(next_board, action)

        if None not in coords:

            next_board[coords] = mark

        return next_board



    def get_coords( self, board: np.ndarray, action: int ) -> Tuple[int,int]:

        col = action if 0 <= action < self.columns else None

        row = np.count_nonzero( board[:,col] == 0 ) - 1

        if row < 0: row = None

        return (row, col)



    @cached_property

    def actions(self) -> List[int]:

        # rows are counted from sky = 0; if the top row is empty we can play

        actions = np.nonzero(self.board[0,:] == 0)[0].tolist()   # BUGFIX: Kaggle wants List[int] not np.ndarray(int64)

        return list(actions)





#####

##### END   games/connectx/core/ConnectX.py

#####



#####

##### START games/connectx/core/Heuristic.py

#####



import math



# from games.connectx.core.KaggleGame import KaggleGame

# from games.connectx.core.PersistentCacheAgent import PersistentCacheAgent

# from util.vendor.cached_property import cached_property







class Heuristic(PersistentCacheAgent):

    """ Returns heuristic_class scores relative to the self.game.player_id"""

    cache = {}

    def __new__(cls, game: KaggleGame, *args, **kwargs):

        hash = game

        if hash not in cls.cache:

            cls.cache[hash] = object.__new__(cls)

        return cls.cache[hash]



    def __init__( self, game: KaggleGame, *args, **kwargs ):

        super().__init__(*args, **kwargs)

        self.game      = game

        self.player_id = game.player_id



    @cached_property

    def gameover( self ) -> bool:

        """Has the game reached a terminal game?"""

        return abs(self.utility) == math.inf



    @cached_property

    def score( self ) -> float:

        """Heuristic score"""

        raise NotImplementedError



    @cached_property

    def utility(self ) -> float:

        """ +inf for victory or -inf for loss else 0 """

        raise NotImplementedError





#####

##### END   games/connectx/core/Heuristic.py

#####



#####

##### START games/connectx/heuristics/LinesHeuristic.py

#####



import functools

import math

from dataclasses import dataclass

from enum import Enum

from enum import unique

from typing import FrozenSet

from typing import List

from typing import Set

from typing import Tuple

from typing import Union



import numpy as np

from fastcache import clru_cache

from numba import njit



# from games.connectx.core.ConnectX import ConnectX

# from games.connectx.core.Heuristic import Heuristic

# from util.vendor.cached_property import cached_property







# (1,0)  -> (-1,0)  = down -> up

# (0,1)  -> (0,-1)  = left -> right

# (1,1)  -> (-1,-1) = down+left -> up+right

# (-1,1) -> (1,-1)  = up+left   -> down+right

@unique

class Direction(Enum):

    UP_DOWN       = (1,0)

    LEFT_RIGhT    = (0,1)

    DIAGONAL_UP   = (1,1)

    DIAGONAL_DOWN = (1,-1)

Directions = frozenset( d.value for d in Direction.__members__.values() )





@functools.total_ordering

@dataclass(init=True, frozen=True)

class Line:

    game:          'ConnectX'

    cells:         FrozenSet[Tuple[int,int]]

    direction:     Direction

    mark:          int







    ### Factory Methods



    @classmethod

    @clru_cache(None)

    def line_from_position( cls, game: 'ConnectX', coord: Tuple[int, int], direction: Direction ) -> Union['Line', None]:

        # NOTE: This function doesn't improve with @jit

        mark = game.board[coord]

        if mark == 0: return None



        cells = { coord }

        for sign in [1, -1]:

            next = cls.next_coord(coord, direction, sign)

            while cls.is_valid_coord(next, game.rows, game.columns) and game.board[next] == mark:

                cells.add(next)

                next = cls.next_coord(next, direction, sign)



        return Line(

            game      = game,

            cells     = frozenset(cells),

            mark      = mark,

            direction = direction,

        )





    ### Magic Methods



    def __len__( self ):

        return len(self.cells)



    def __hash__(self):

        return hash((self.direction, self.cells))



    def __eq__(self, other):

        if not isinstance(other, self.__class__): return False

        return self.cells == other.cells and self.direction == other.direction



    def __lt__(self, other):

        return self.score < other.score



    def __repr__(self):

        args = {"mark": self.mark, "direction": self.direction, "cells": self.cells }

        return f"Line({args})"







    ### Navigation Methods



    @staticmethod

    @njit()

    def next_coord( coord: Tuple[int, int], direction: Direction, sign=1 ) -> Tuple[int,int]:

        """Use is_valid_coord to verify the coord is valid """

        return ( coord[0]+(direction[0]*sign), coord[1]+(direction[1]*sign) )



    @staticmethod

    @njit()

    def is_valid_coord( coord: Tuple[int, int], rows: int, columns: int ) -> bool:

        x,y  = coord

        if x < 0 or rows    <= x: return False

        if y < 0 or columns <= y: return False

        return True





    ### Heuristic Methods



    @cached_property

    # @njit() ### throws exceptions

    def gameover( self ) -> bool:

        return len(self) == self.game.inarow



    @njit()

    def utility( self, player_id: int ) -> float:

        if len(self) == self.game.inarow:

            if player_id == self.mark: return  math.inf

            else:                      return -math.inf

        return 0



    @cached_property

    def score( self ):

        # A line with zero liberties is dead

        # A line with two liberties is a potential double attack

        # A line of 2 with 2 liberties is worth more than a line of 3 with one liberty

        if len(self) == self.game.inarow: return math.inf

        if len(self.liberties) == 0:      return 0                         # line can't connect 4

        if len(self) + self.extension_length < self.game.inarow: return 0  # line can't connect 4

        score = ( len(self)**2 + self.extension_score ) * len(self.liberties)

        if len(self) == 1: score /= len(Directions)                                    # Discount duplicates

        return score





    @cached_property

    def extension_length( self ):

        return np.sum(list(map(len, self.extensions)))



    @cached_property

    def extension_score( self ):

        # less than 1 - ensure center col is played first

        return np.sum([ len(extension)**1.25 for extension in self.extensions ]) / ( self.game.inarow**2 )





    @cached_property

    def liberties( self ) -> Set[Tuple[int,int]]:

        ### Numba doesn't like this syntax

        cells = {

            self.next_coord(coord, self.direction, sign)

            for coord in self.cells

            for sign in [1, -1]

        }

        cells = {

            coord

            for coord in cells

            if  self.is_valid_coord(coord, self.game.rows, self.game.columns)

                and self.game.board[coord] == 0

        }

        return cells



    ### BUG: Numba optimized code returns zero scores

    # @cached_property

    # def liberties( self ) -> Set[Tuple[int,int]]:

    #     return self._liberties(

    #         direction=self.direction,

    #         cells=tuple(self.cells),

    #         board=self.game.board,

    #         rows=self.game.rows,

    #         columns=self.game.columns,

    #         is_valid_coord=self.is_valid_coord,

    #         next_coord=self.next_coord,

    #     )

    # @staticmethod

    # @njit(parallel=True)

    # def _liberties( direction, cells, board: np.ndarray, rows: int, columns: int, is_valid_coord, next_coord ) -> Set[Tuple[int,int]]:

    #     coords = set()

    #     for sign in [1, -1]:

    #         for coord in cells:

    #             next = next_coord(coord, direction, sign)

    #             coords.add(next)

    #     output = set()

    #     for coord in coords:

    #         if is_valid_coord(coord, rows, columns) and board[coord] == 0:

    #             output.add(coord)

    #     return output





    @cached_property

    def extensions( self ) -> List[FrozenSet[Tuple[int,int]]]:

        extensions = []

        for next in self.liberties:

            extension = { next }

            for sign in [1,-1]:

                while len(extension) + len(self) < self.game.inarow:

                    next = self.next_coord(next, self.direction, sign)

                    if next in self.cells:                                               break

                    if not self.is_valid_coord(next, self.game.rows, self.game.columns): break

                    if self.game.board[next] not in (0, self.mark):                      break

                    extension.add(next)

            if len(extension):

                extensions.append(frozenset(extension))

        return extensions



    ### BUG: Numba optimized code returns zero scores

    # @cached_property

    # def extensions( self ) -> List[Set[Tuple[int,int]]]:

    #     return self._extensions(

    #         length_self=len(self),

    #         liberties=self.liberties,

    #         cells=self.cells,

    #         mark=self.mark,

    #         direction=self.direction,

    #         board=self.game.board,

    #         inarow=self.game.inarow,

    #         rows=self.game.rows,

    #         columns=self.game.columns,

    #         next_coord=self.next_coord,

    #         is_valid_coord=self.is_valid_coord

    #     )

    # @staticmethod

    # @njit(parallel=True)

    # def _extensions( length_self, liberties, cells, mark, direction, board, inarow, rows, columns, next_coord, is_valid_coord ) -> List[Set[Tuple[int,int]]]:

    #     extensions = []

    #     for next in liberties:

    #         extension = { next }

    #         for sign in [1,-1]:

    #             while len(extension) + length_self < inarow:

    #                 next = next_coord(next, direction, sign)

    #                 if next in cells:                           break

    #                 if not is_valid_coord(next, rows, columns): break

    #                 if board[next] not in (0, mark):            break

    #                 extension.add(next)

    #         if len(extension):

    #             extensions.append(set(extension))

    #     return extensions







class LinesHeuristic(Heuristic):

    ### Heuristic Methods - relative to the current self.player_id

    ## Heuristic Methods



    cache = {}

    def __new__(cls, game: ConnectX, *args, **kwargs):

        hash = frozenset(( game.board.tobytes(), np.fliplr(game.board).tobytes() ))

        if hash not in cls.cache:

            cls.cache[hash] = object.__new__(cls)

        return cls.cache[hash]



    def __init__(self, game: ConnectX):

        super().__init__(game)

        self.game      = game

        self.board     = game.board

        self.player_id = game.player_id



    @cached_property

    def lines(self) -> List['Line']:

        lines = set()

        for (row,col) in zip(*np.where(self.board != 0)):

            if self.board[row,col] == 0: continue

            lines |= {

                Line.line_from_position(self.game, (row, col), direction)

                for direction in Directions

            }

        lines = { line for line in lines if line.score != 0 }

        return sorted(lines, reverse=True, key=len)



    @cached_property

    def gameover( self ) -> bool:

        """Has the game reached a terminal game?"""

        if len( self.game.actions ) == 0:                    return True

        if np.any([ line.gameover for line in self.lines ]): return True

        return False





    @cached_property

    def score( self ) -> float:

        """Heuristic score"""

        # mark is the next player to move - calculate score from perspective of player who just moved

        hero_score    = np.sum([ line.score for line in self.lines if line.mark != self.player_id ])

        villain_score = np.sum([ line.score for line in self.lines if line.mark == self.player_id ])

        return hero_score - villain_score



    @cached_property

    def utility(self) -> float:

        """ +inf for victory or -inf for loss else 0 - calculated from the perspective of the player who made the previous move"""

        for line in self.lines:

            if len(line) == 4:

                # mark is the next player to move - calculate score from perspective of player who just moved

                return math.inf if line.mark != self.player_id else -math.inf

            else:

                break  # self.lines is sorted by length

        return 0





#####

##### END   games/connectx/heuristics/LinesHeuristic.py

#####



#####

##### START games/connectx/agents/AlphaBetaAgent.py

#####



import math

import random

import time

from queue import LifoQueue

from struct import Struct

from typing import Callable



# from games.connectx.core.ConnectX import ConnectX

# from games.connectx.core.KaggleGame import KaggleGame

# from games.connectx.core.PersistentCacheAgent import PersistentCacheAgent

# from games.connectx.heuristics.LinesHeuristic import LinesHeuristic







class AlphaBetaAgent(PersistentCacheAgent):

    heuristic_class = LinesHeuristic

    defaults = {

        "verbose_depth":    True,

        "search_max_depth": 100 # if "pytest" not in sys.modules else 3,

    }



    def __init__( self, game: ConnectX, *args, **kwargs ):

        super().__init__(*args, **kwargs)

        self.kwargs    = { **self.defaults, **kwargs }

        self.game      = game

        self.player_id = game.observation.mark

        self.queue     = LifoQueue()

        self.verbose_depth    = self.kwargs.get('verbose_depth')

        self.search_max_depth = self.kwargs.get('search_max_depth')





    ### Public Interface



    def get_action( self, endtime: float ) -> int:

        action = self.iterative_deepening_search(endtime=endtime)

        return int(action)







    ### Search Functions



    def iterative_deepening_search( self, endtime=0.0 ) -> int:

        # The real trick with iterative deepening is caching, which allows us to out-depth the default minimax Agent

        if self.verbose_depth: print('\n'+ self.__class__.__name__.ljust(20) +' | depth:', end=' ', flush=True)

        best_action = random.choice(self.game.actions)

        try:

            for depth in range(1, self.search_max_depth+1):

                action, score = self.alphabeta(self.game, depth=depth, endtime=endtime)

                if endtime and time.perf_counter() >= endtime: break  # ignore results on timeout



                best_action = action

                if self.verbose_depth: print(depth, end=' ', flush=True)

                if abs(score) == math.inf:

                    if self.verbose_depth: print(score, end=' ', flush=True)

                    break  # terminate iterative deepening on inescapable victory condition

        except TimeoutError:

            pass  # This is the fastest way to exit a loop: https://www.kaggle.com/c/connectx/discussion/158190

        return int(best_action)

        # if self.verbose_depth: print( depth, type(action), action, int((time.perf_counter() - time_start) * 1000), 'ms' )





    def alphabeta( self, game, depth, endtime=0.0 ):

        scores = []

        best_action = random.choice(game.actions)

        best_score  = -math.inf

        for action in game.actions:

            result = game.result(action)

            score  = self.alphabeta_min_value(result, player_id=self.player_id, depth=depth-1, endtime=endtime)

            if endtime and time.perf_counter() >= endtime: raise TimeoutError

            if score > best_score:

                best_score  = score

                best_action = action

            scores.append(score)  # for debugging



        # action, score = max(zip(game.actions, scores), key=itemgetter(1))

        return best_action, best_score  # This is slightly quicker for timeout purposes





    def alphabeta_min_value( self, game: KaggleGame, player_id: int, depth: int, alpha=-math.inf, beta=math.inf, endtime=0.0):

        return self.cache_infinite(self._alphabeta_min_value, game, player_id, depth, alpha, beta, endtime)

    def _alphabeta_min_value( self, game: KaggleGame, player_id, depth: int, alpha=-math.inf, beta=math.inf, endtime=0.0 ):

        sign = 1 if player_id != game.player_id else -1

        if game.gameover:  return sign * game.heuristic.utility  # score relative to previous player who made the move

        if depth == 0:     return sign * game.heuristic.score

        scores = []

        score  = math.inf

        for action in game.actions:

            result    = game.result(action)

            score     = min(score, self.alphabeta_max_value(result, player_id, depth-1, alpha, beta, endtime))

            if endtime and time.perf_counter() >= endtime: raise TimeoutError

            if score <= alpha: return score

            beta      = min(beta,score)

            scores.append(score)  # for debugging

        return score



    def alphabeta_max_value( self, game: KaggleGame, player_id: int, depth, alpha=-math.inf, beta=math.inf, endtime=0.0  ):

        return self.cache_infinite(self._alphabeta_max_value, game, player_id, depth, alpha, beta, endtime)

    def _alphabeta_max_value( self, game: KaggleGame, player_id: int, depth, alpha=-math.inf, beta=math.inf, endtime=0.0  ):

        sign = 1 if player_id != game.player_id else -1

        if game.gameover:  return sign * game.heuristic.utility  # score relative to previous player who made the move

        if depth == 0:     return sign * game.heuristic.score

        scores = []

        score  = -math.inf

        for action in game.actions:

            result    = game.result(action)

            score     = max(score, self.alphabeta_min_value(result, player_id, depth-1, alpha, beta, endtime))

            if endtime and time.perf_counter() >= endtime: raise TimeoutError

            if score >= beta: return score

            alpha     = max(alpha, score)

            scores.append(score)  # for debugging

        return score







    ### Exported Interface



    # observation   = {'mark': 1, 'board': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]}

    # configuration = {'columns': 7, 'rows': 6, 'inarow': 4, 'steps': 1000, 'timeout': 2}

    @classmethod

    def agent(cls, **kwargs) -> Callable[[Struct, Struct],int]:

        heuristic_class = kwargs.get('heuristic_class', cls.heuristic_class)



        def kaggle_agent(observation: Struct, configuration: Struct):

            # Leave a small amount of time to return an answer - was 1.1, but try 0.25 now we exiting loop via exception

            endtime = time.perf_counter() + configuration.timeout - 0.75

            game    = ConnectX(observation, configuration, heuristic_class, **kwargs)

            agent   = cls(game, **kwargs)

            action  = agent.get_action(endtime)

            # print(endtime - time.perf_counter(), 's')  # min -0.001315439000000751 s

            return int(action)

        return kaggle_agent







# The last function defined in the file run by Kaggle in submission.csv

def agent(observation, configuration) -> int:

    return AlphaBetaAgent.agent()(observation, configuration)





#####

##### END   games/connectx/agents/AlphaBetaAgent.py

#####



##### 

##### ./kaggle_compile.py games/connectx/agents/AlphaBetaAgent.py

##### 

##### 2020-06-24 12:38:03+01:00

##### 

##### ai-pacman	git@github.com:JamesMcGuigan/ai-pacman.git (fetch)

##### ai-pacman	git@github.com:JamesMcGuigan/ai-pacman.git (push)

##### ecosystem-research	git@github.com:JamesMcGuigan/ecosystem-research.git (fetch)

##### ecosystem-research	git@github.com:JamesMcGuigan/ecosystem-research.git (push)

##### kaggle-arc	git@github.com:JamesMcGuigan/kaggle-arc.git (fetch)

##### kaggle-arc	git@github.com:JamesMcGuigan/kaggle-arc.git (push)

##### origin	git@github.com:JamesMcGuigan/ai-games.git (fetch)

##### origin	git@github.com:JamesMcGuigan/ai-games.git (push)

##### udacity-artificial-intelligence	https://github.com/JamesMcGuigan/udacity-artificial-intelligence (fetch)

##### udacity-artificial-intelligence	https://github.com/JamesMcGuigan/udacity-artificial-intelligence (push)

##### 

#####   kaggle-arc        583a4c1 kaggle_compile.py -> submission.py + submission.csv

#####   knights-isolation 27d0916 3_Adversarial Search | udacity submit requires kaggle_compile.py

#####   master            f296c88 refactor: AlphaBetaAgent.agent -> AlphaBetaAgent.agent(kwargs)(observation, configuration)

#####   n-queens          ddb0d49 prolog | nqueens | print solution to nqueens.txt

##### * numba             88fa481 [ahead 3] ConnectX | AlphaBetaAgent | exit loop via raise TimeoutError

##### 

##### 88fa481f1803f20c88654ce21c5ce5dcea14aa95

##### 
%run submission.py
from kaggle_environments import evaluate, make, utils



%load_ext autoreload

%autoreload 2
### Play against yourself without an ERROR or INVALID.

### Note: The first episode in the competition will run this to weed out erroneous agents.



env = make("connectx", debug=True)

env.run(["/kaggle/working/submission.py", "/kaggle/working/submission.py"])

print("\nEXCELLENT SUBMISSION!" if env.toJSON()["statuses"] == ["DONE", "DONE"] else "MAYBE BAD SUBMISSION?")

env.render(mode="ipython", width=500, height=450)
env = make("connectx", debug=True)

env.run(["/kaggle/working/submission.py", "negamax"])

print("\nEXCELLENT SUBMISSION!" if env.toJSON()["statuses"] == ["DONE", "DONE"] else "MAYBE BAD SUBMISSION?")

env.render(mode="ipython", width=500, height=450)

env = make("connectx", debug=True)

env.play([None, "/kaggle/working/submission.py"], width=500, height=450)