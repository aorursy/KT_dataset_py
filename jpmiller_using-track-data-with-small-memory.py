import sys

import numpy as np

import pandas as pd

pd.options.display.float_format = '{:,.2f}'.format

from tqdm.notebook import tqdm



import skmem #utility script
csize = 4_000_000 #set this to fit your situation

chunker = pd.read_csv('../input/nfl-playing-surface-analytics/PlayerTrackData.csv',

                      chunksize=csize)

track_list = []

mr = skmem.MemReducer()

for chunk in tqdm(chunker, total = int(80_000_000/csize)):

    chunk['PlayKey'] = chunk.PlayKey.fillna('0-0-0')

    id_array = chunk.PlayKey.str.split('-', expand=True).to_numpy()

    chunk['PlayerKey'] = id_array[:,0]

    chunk['GameID'] = id_array[:,1]

    chunk['PlayKey'] = id_array[:,2]

    chunk['event'] = chunk.event.fillna('none')

    floaters = chunk.select_dtypes('float').columns.tolist()

    chunk = mr.fit_transform(chunk, float_cols=floaters) #float downcast is optional

    track_list.append(chunk)
tracks = pd.concat(track_list)

tracks['event'] = tracks.event.astype('category') #retype after concat

col_order = [9,10,0,1,2,3,4,5,7,6,8]

tracks = tracks[[tracks.columns[idx] for idx in col_order]]

display(tracks.dtypes, tracks.head())
tracks.to_parquet('InjuryRecord.parq')