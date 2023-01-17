import numpy  as np 

import pandas as pd 



import os

import glob

from   pathlib import Path

from   functools import reduce

from   typing import List, Tuple, Dict, NamedTuple



import collections





# import os

# for dirname, _, filenames in os.walk('/kaggle/input'):

#     for filename in filenames:

#         print(os.path.join(dirname, filename))
root_dir    = '/kaggle'

project_dir = 'nfl-big-data-bowl-2021'

data_dir    = Path(root_dir)/f'input/{project_dir}'

export_dir  = Path(root_dir)/'working/exports' # 'Save and Run All'

scratch_dir = Path(root_dir)/'temp/exports'    # 'Save and Run All'

!ls /kaggle/input/nfl-big-data-bowl-2021
df_games    = pd.read_csv(Path(data_dir)/'games.csv')

df_players  = pd.read_csv(Path(data_dir)/'players.csv')

df_plays    = pd.read_csv(Path(data_dir)/'plays.csv')

df_games.shape, df_players.shape, df_plays.shape
file_names   = sorted( glob.glob(str(Path(data_dir)/'*')) )

weekly_files = [file for file in file_names if Path(file).parts[-1].startswith('week')]

weekly_data  = [pd.read_csv(file, parse_dates=['time']) for file in weekly_files]

df_weekly    = pd.concat(weekly_data, axis=0).sort_values(by='time')

df_weekly.shape