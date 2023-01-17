import os

import numpy as np

import pandas as pd
INPUT_DIR = '../input'

def load_data(input_dir):

    return [pd.read_csv(os.path.join(input_dir, file_), index_col=0) for file_ in os.listdir(input_dir)]

        

serves, ball_bounces, rallies, points, events = load_data(INPUT_DIR)
# events

events.head()
# points

points.head()
# rallies

rallies.head()
# serves

serves.head()
# ball bounces

ball_bounces.head()
points[['rallyid','winner']].groupby('winner').count()
points.groupby(['server']).size()