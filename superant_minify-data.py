import numpy as np

import pandas as pd

import sys

import json

from collections import namedtuple



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
dtypes = {

    "title": "category",

    "event_count": "int16",

    "event_code": "int16",

    "game_time": "int32",

    "title": "category",

    "type": "category",

    "world": "category",

}



D = pd.read_csv("/kaggle/input/data-science-bowl-2019/train.csv", dtype=dtypes, parse_dates=["timestamp"])



D.info(memory_usage='deep')
def factorize(dd):

    id_maps={}



    for col, dtype in [("event_id", "int16"), ("game_session", "int32"), ("installation_id", "int16")]:

        factors, id_map = pd.factorize(dd[col])

        dd[col]=factors.astype(dtype)

        

        if col in ["event_id"]:

            id_maps[col]=id_map

    

    return id_maps



id_maps = factorize(D)   # the map for event_id might be needed later

D.head()
namedtuple_types={}





def parse_event_data(text):

    result = json.loads(text)

    for col in ["event_code", "event_count", "game_time"]:

        if col in result:

            del result[col]

            

    if "identifier" in result:

        result["identifier"]=tuple(result["identifier"].split(","))

        

    result = recur_str_intern(result)

    

    return result





def recur_str_intern(obj):

    if isinstance(obj, str):

        return sys.intern(obj)

    

    if isinstance(obj, dict):

        keys = frozenset(obj.keys())

        sorted_keys = sorted(keys)

        

        if keys not in namedtuple_types:

            namedtuple_types[keys]=namedtuple(f"EventData{len(namedtuple_types)}", sorted_keys)

            

        cur_type=namedtuple_types[keys]

            

        return cur_type(*[recur_str_intern(obj[k]) for k in sorted_keys])

    

    if isinstance(obj, list):

        return tuple(recur_str_intern(x) for x in obj)  # convert to tuple anyway

    

    if isinstance(obj, tuple):

        return tuple(recur_str_intern(x) for x in obj)

    

    return obj



for dd in [D]:

    dd["event_data"] = dd["event_data"].apply(parse_event_data)

    

D.head()   # Unfortunately Pandas does not display namedtuples right
D.info(memory_usage='deep')  # doubt that it takes everything into account, but let's see