# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

from collections import defaultdict
import itertools

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from pathlib import Path

data_dir = Path('../input')

import os
os.listdir(data_dir)
# Read a file that contains the CityID for every Tourney slot. This is painstakingly created by hand based on information at ncaa.com.

tourney_cities = pd.read_csv(data_dir / '2018-ncaa-mens-supplement/NCAATourneyCities.csv')
tourney_cities.sample(10)
def _traverse_bracket(seed, path_to_final, slots):
    # recursively traverse the bracket (I'm always proud when a recursion works
    # the first time) and append all slots from round 1 to national final
    # for a seed to a list
    slot = pd.concat([slots.Slot[slots.StrongSeed == seed],
                      slots.Slot[slots.WeakSeed == seed]]).values[0]
    path_to_final.append(slot)
    if slot == 'R6CH':
        return
    _traverse_bracket(slot, path_to_final, slots)


def get_slot_dict(season, data_dir=Path('input')):
    seeds = pd.read_csv(data_dir / 'NCAATourneySeeds.csv')
    slots = pd.read_csv(data_dir / 'NCAATourneySlots.csv')

    idx = seeds.Season == season
    seeds = seeds[idx]
    idx = slots.Season == season
    slots = slots[idx]
    # slot_teams is a dict with tourney slots as keys
    # holding a list of all teams that could possibly
    # go through this slot
    slot_teams = defaultdict(list)
    for ii, row in seeds.iterrows():
        ptf = []
        _traverse_bracket(row.Seed, ptf, slots)
        for slot in ptf:
            slot_teams[slot].append(row.TeamID)
    
    # for every t1_t2 combination get all hypothetical slots
    pair_slot = defaultdict(list)
    for k, v in slot_teams.items():
        for t1, t2 in itertools.product(v, v):
            if t1 >= t2:
                continue
            pair_slot[f'{t1}_{t2}'].append(k)

    # and then keep only the first one where they can meet
    play_in_slots = {'W11', 'W16', 'Y16', 'Z11'}
    for k, v in pair_slot.items():
        v.sort()
        pair_slot[k] = v[-1] if v[-1] in play_in_slots else v[0]
    return pair_slot
pair_slot = get_slot_dict(2018, data_dir / 'mens-machine-learning-competition-2018')
# Test some pairings for the 2018 season
# W and Z meet only in the final (Z01 == 1314, W12 == 1293)
assert pair_slot['1293_1314'] == 'R6CH'
# W01 and W02 meet in R4W1
assert pair_slot['1345_1437'] == 'R4W1'
# Y and Z meet in R5YZ
assert pair_slot['1243_1344'] == 'R5YZ'
# Seeds 1 and 16 meet in first round
assert pair_slot['1300_1462'] == 'R1Z1'
# Taken form the starter kernel
def get_year_t1_t2(ID):
    """Return a tuple with ints `year`, `team1` and `team2`."""
    return (int(x) for x in ID.split('_'))

# Enter that pair/slot information to the sample submission
sample_sub = pd.read_csv(data_dir / 'mens-machine-learning-competition-2018' / 'SampleSubmissionStage2.csv')
sample_sub = pd.concat([pd.DataFrame(columns=['CityID'], dtype=int), sample_sub])
for ii, row in sample_sub.iterrows():
    year, t1, t2 = get_year_t1_t2(row.ID)
    slot = pair_slot[f'{t1}_{t2}']
    game_city_id = tourney_cities.loc[tourney_cities.Slot == slot, 'CityID'].values[0]
    sample_sub.loc[ii, 'CityID'] = game_city_id
    
sample_sub.sample(10)
sample_sub.to_csv('SampleSubStage2CityID.csv')
