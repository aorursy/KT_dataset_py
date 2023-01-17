import datetime as dt
import os
import re

import numpy as np
import pandas as pd
from tqdm import tqdm

NFL_DATA_DIR = '../input'
NGS_DIR = '.'

YARD = 0.9144
MPH = 1.609344
SPEED_LIMIT = 13
MAX_SPEED = 11
SMOOTHING_FACTOR = 3
start = dt.datetime.now()
NGS_BY_PLAY = {}
ngs_filenames = [f for f in os.listdir(NFL_DATA_DIR) if f.startswith('NGS')]
for f in ngs_filenames[:]:
    ngs = pd.read_csv(os.path.join(NFL_DATA_DIR, f), parse_dates=['Time'], low_memory=False)
    ngs['PlayKey'] = ngs['GameKey'].apply(str) + '_' + ngs['PlayID'].apply(str)
    for playkey, df in tqdm(ngs.groupby('PlayKey')):
        play_filename = os.path.join(NGS_DIR, f'{playkey}.csv')
        NGS_BY_PLAY[playkey] = df[['GSISID', 'Time', 'x', 'y', 'dis', 'o', 'dir', 'Event']]

def get_plays():
    data = pd.read_csv(os.path.join(NFL_DATA_DIR, 'play_information.csv'), parse_dates=['Game_Date'])
    data.columns = [col.replace('_', '') for col in data.columns]
    data['PlayKey'] = data['GameKey'].apply(str) + '_' + data['PlayID'].apply(str)
    data = data.drop(['PlayID', 'PlayType'], axis=1)
    data = data.sort_values(['GameKey', 'Quarter', 'GameClock'])
    data['PlayType'] = 'Punt'
    return data

def get_ngs(playkey):
    ngs = NGS_BY_PLAY[playkey].copy()
    ngs['t'] = (ngs.Time - ngs.Time.min()) / np.timedelta64(1, 's')
    ngs = ngs.sort_values(by='t')
    return ngs

def calculate_speed_and_acceleration(ngs, smoothing_factor=5):
    speed = ngs.pivot('t', 'GSISID', 'dis') * YARD
    speed = speed.fillna(0)
    speed = speed.rolling(smoothing_factor).mean() * 10
    acc = speed.clip(0, MAX_SPEED).diff(smoothing_factor) * 10. / smoothing_factor
    return speed, acc

def collect_ngs_player_stats():
    plays = get_plays()
    result = []
    for playkey in tqdm(plays.PlayKey.values):
        try:
            ngs = get_ngs(playkey)

            speed, acc = calculate_speed_and_acceleration(ngs, SMOOTHING_FACTOR)
            max_speed = speed.max(axis=0).reset_index().rename(columns={0: 'MaxSpeed'})
            min_acceleration = acc.min(axis=0).reset_index().rename(columns={0: 'MinAcceleration'})
            
            collision_coords = pd.DataFrame([[c, acc[c].argmin()] for c in acc.columns],
                                            columns=['GSISID', 't'])
            collision_coords = collision_coords.merge(ngs[['GSISID', 't', 'x', 'y']],
                                                      how='left', on=['GSISID', 't'])
            collision_coords['x'] = collision_coords['x'] - 10
            collision_coords.columns = ['GSISID', 'CollisionTime', 'CollisionX', 'CollisionY']

            stats = pd.merge(max_speed, min_acceleration, on='GSISID')
            stats = stats.merge(collision_coords, on='GSISID', how='left')
            stats['PlayKey'] = playkey
            result.append(stats)
        except Exception as e:
            print(e)
    return pd.concat(result)
player_ngs = collect_ngs_player_stats()
print(player_ngs.shape)
print(player_ngs.count())
print(player_ngs.nunique())
player_ngs.to_csv('player_ngs.csv', index=False)
def get_video_review():
    data = pd.read_csv(os.path.join(NFL_DATA_DIR, 'video_review.csv'))
    data.columns = [col.replace('_', '') for col in data.columns]
    data['PlayKey'] = data['GameKey'].apply(str) + '_' + data['PlayID'].apply(str)

    footage = pd.read_csv(os.path.join(NFL_DATA_DIR, 'video_footage-injury.csv'))
    footage['PlayKey'] = footage['gamekey'].apply(str) + '_' + footage['playid'].apply(str)

    footage = footage.rename(columns={'PREVIEW LINK (5000K)': 'VideoLink'})
    data = data.merge(footage[['PlayKey', 'VideoLink', 'PlayDescription']], how='left', on=['PlayKey'])
    return data
videos = get_video_review()
for playkey in videos.PlayKey.values:
    try:
        NGS_BY_PLAY[playkey].to_csv(f'ngs_{playkey}.csv', index=False)
    except Exception as e:
        print(e)
end = dt.datetime.now()
print('Latest run {}.\nTotal time {}s'.format(end, (end - start).seconds))