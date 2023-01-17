import base64

import gzip

import os

import re

import time

from typing import Any

from typing import Union



import dill

import humanize





# _base64_file__test_base64_static_import = """

# H4sIAPx9LF8C/2tgri1k0IjgYGBgKCxNLS7JzM8rZIwtZNLwZvBm8mYEkjAI4jFB2KkRbED1iXnF

# 5alFhczeWqV6AEGfwmBHAAAA

# """





def base64_file_varname(filename: str) -> str:

    # ../data/AntColonyTreeSearchNode.dill.zip.base64 -> _base64_file__AntColonyTreeSearchNode__dill__zip__base64

    varname = re.sub(r'^.*/',   '',   filename)  # remove directories

    varname = re.sub(r'[.\W]+', '__', varname)   # convert dots and non-ascii to __

    varname = f"_base64_file__{varname}"

    return varname





def base64_file_var_wrap(base64_data: Union[str,bytes], varname: str) -> str:

    return f'{varname} = """\n{base64_data.strip()}\n"""'                    # add varname = """\n\n""" wrapper





def base64_file_var_unwrap(base64_data: str) -> str:

    output = base64_data.strip()

    output = re.sub(r'^\w+ = """|"""$', '', output)  # remove varname = """ """ wrapper

    output = output.strip()

    return output





def base64_file_encode(data: Any) -> str:

    encoded = dill.dumps(data)

    encoded = gzip.compress(encoded)

    encoded = base64.encodebytes(encoded).decode('utf8').strip()

    return encoded





def base64_file_decode(encoded: str) -> Any:

    data = base64.b64decode(encoded)

    data = gzip.decompress(data)

    data = dill.loads(data)

    return data





def base64_file_save(data: Any, filename: str, vebose=True) -> float:

    """

        Saves a base64 encoded version of data into filename, with a varname wrapper for importing via kaggle_compile.py

        # Doesn't create/update global variable.

        Returns filesize in bytes

    """

    varname    = base64_file_varname(filename)

    start_time = time.perf_counter()

    try:

        os.makedirs(os.path.dirname(filename), exist_ok=True)

        with open(filename, 'wb') as file:

            encoded = base64_file_encode(data)

            output  = base64_file_var_wrap(encoded, varname)

            output  = output.encode('utf8')

            file.write(output)

            file.close()

        if varname in globals(): globals()[varname] = encoded  # globals not shared between modules, but update for saftey



        filesize = os.path.getsize(filename)

        if vebose:

            time_taken = time.perf_counter() - start_time

            print(f"base64_file_save(): {filename:40s} | {humanize.naturalsize(filesize)} in {time_taken:4.1f}s")

        return filesize

    except Exception as exception:

        pass

    return 0.0





def base64_file_load(filename: str, vebose=True) -> Union[Any,None]:

    """

        Performs a lookup to see if the global variable for this file alread exists

        If not, reads the base64 encoded file from filename, with an optional varname wrapper

        # Doesn't create/update global variable.

        Returns decoded data

    """

    varname    = base64_file_varname(filename)

    start_time = time.perf_counter()

    try:

        # Hard-coding PyTorch weights into a script - https://www.kaggle.com/c/connectx/discussion/126678

        encoded = None



        if varname in globals():

            encoded = globals()[varname]



        if encoded is None and os.path.exists(filename):

            with open(filename, 'rb') as file:

                encoded = file.read().decode('utf8')

                encoded = base64_file_var_unwrap(encoded)

                # globals()[varname] = encoded  # globals are not shared between modules



        if encoded is not None:

            data = base64_file_decode(encoded)



            if vebose:

                filesize = os.path.getsize(filename)

                time_taken = time.perf_counter() - start_time

                print(f"base64_file_load(): {filename:40s} | {humanize.naturalsize(filesize)} in {time_taken:4.1f}s")

            return data

    except Exception as exception:

        print(f'base64_file_load({filename}): Exception:', exception)

    return None

from pytest import fixture



from util.base64_file import *





@fixture

def data():

    return { "question": [ 0,2,1,0,0,0,0, 0,2,1,2,0,0,0 ], "answer": 42 }





def test_base64_file_varname():

    input    = './data/MontyCarloNode.pickle.zip.base64'

    expected = '_base64_file__MontyCarloNode__pickle__zip__base64'

    actual   = base64_file_varname(input)

    assert actual == expected





def test_base64_wrap_unwrap(data):

    varname   = base64_file_varname('test')

    input     = base64.encodebytes(dill.dumps(data)).decode('utf8').strip()

    wrapped   = base64_file_var_wrap(input, varname)

    unwrapped = base64_file_var_unwrap(wrapped)



    assert isinstance(input,   str)

    assert isinstance(wrapped, str)

    assert isinstance(unwrapped, str)

    assert varname     in wrapped

    assert varname not in unwrapped

    assert input != wrapped

    assert input == unwrapped





def test_base64_save_load(data):

    assert data == data



    filename = '/tmp/test_base64_save_load'

    if os.path.exists(filename): os.remove(filename)

    assert not os.path.exists(filename)



    loaded   = base64_file_load(filename)

    assert loaded is None



    varname  = base64_file_varname(filename)

    filesize = base64_file_save(data, filename)

    loaded   = base64_file_load(filename)



    assert os.path.exists(filename)

    assert filesize < 1024  # less than 1kb

    assert data == loaded



    # assert varname in globals()          # globals are not shared between modules

    # assert output == globals()[varname]  # globals are not shared between modules





def test_base64_static_import(data):

    assert data == data



    filename = '/tmp/test_base64_static_import'

    if os.path.exists(filename): os.remove(filename)

    assert not os.path.exists(filename)





    varname  = base64_file_varname(filename)

    filesize = base64_file_save(data, filename)

    loaded   = base64_file_load(filename)



    if varname in globals(): del globals()[varname]  # globals are not shared between modules

    contents = open(filename, 'r').read()

    exec(contents, globals())

    assert varname in globals()

    encoded = globals()[varname]

    decoded = base64_file_decode(encoded)



    assert varname in contents

    assert data == loaded == decoded
# Source: https://github.com/JamesMcGuigan/ai-games/blob/master/games/connectx/core/PersistentCacheAgent.py



import atexit

import math



# from util.base64_file import base64_file_load

# from util.base64_file import base64_file_save





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

        return f'./data/{cls.__name__}_base64'



    @classmethod

    def load( cls ):

        if not cls.persist: return  # disable persistent caching

        if cls.cache:       return  # skip loading if the file is already in class memory, empty dict is False

        filename = cls.filename()

        loaded   = base64_file_load(filename, vebose=cls.verbose)

        if loaded:  # cls.cache should not be set to None

            cls.cache = loaded



    @classmethod

    def save( cls ):

        if not cls.persist: return  # disable persistent caching

        # cls.load()                # update any new information from the file

        if cls.cache:

            filename = cls.filename()

            base64_file_save(cls.cache, filename, vebose=cls.verbose)





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
