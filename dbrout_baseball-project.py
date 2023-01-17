import numpy as np 

import pandas as pd 

import seaborn as sns

import matplotlib.pyplot as plt
pitches = pd.read_csv('../input/pitches.csv')

pitches.rename(columns={'ab_id':'atbat_id'}, inplace=True)

pitches.head(10)
atbats = pd.read_csv('../input/atbats.csv')

atbats.rename(columns={'ab_id':'atbat_id'}, inplace=True)

atbats.head(10)
player_name = pd.read_csv('../input/player_names.csv')

player_name.rename(columns={'id':'batter_id'}, inplace=True)

player_name.head(10)
#your code here