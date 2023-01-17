import pandas as pd

import numpy as np
raw_data = {"name": ['Bulbasaur', 'Charmander','Squirtle','Caterpie'],

            "evolution": ['Ivysaur','Charmeleon','Wartortle','Metapod'],

            "type": ['grass', 'fire', 'water', 'bug'],

            "hp": [45, 39, 44, 45],

            "pokedex": ['yes', 'no','yes','no']                        

            }
pokemon = pd.Series(raw_data)

pokemon
pokemon1 = pd.DataFrame.from_dict(raw_data)

pokemon1 = pokemon1[["name", "type", "hp", "evolution", "pokedex"]]

pokemon1
place = ["Hills","Volcano","Lakes","park"]

pokemon1["Place"] = place

pokemon1
pokemon1.dtypes