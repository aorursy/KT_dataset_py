import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import json
# data is stored in a tab seperated file 

data = pd.read_csv('../input/dota2_yasp_v1.txt', sep='\t', index_col=0)
data.head()
# the column 'item' holds the original json string that the other columns are derived from.

example = json.loads(data.loc[1, 'item'])

print(example.keys())
# Most of the information is in the players list

# it contains one dict for each of the ten players

print('number of players: ', len(example['players']))

print(example['players'][0].keys())
# What I mostly knew about dota before looking at this data is people are sometimes

# rude in chat. Lets see if there is any evidence of this.



pd.DataFrame(example['chat'])
