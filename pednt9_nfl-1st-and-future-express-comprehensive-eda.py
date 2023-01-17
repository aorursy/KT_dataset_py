import pandas as pd

import matplotlib.pylab as plt

import seaborn as sns

import matplotlib.patches as patches

import pandas_profiling

import warnings

warnings.filterwarnings('ignore')



from time import time
# Load the data files

playlist = pd.read_csv('../input/nfl-playing-surface-analytics/PlayList.csv')

injuries = pd.read_csv('../input/nfl-playing-surface-analytics/InjuryRecord.csv')

tracking = pd.read_csv('../input/nfl-playing-surface-analytics/PlayerTrackData.csv', nrows=int(1e6)) # load only a fraction of the data
# Make report for playlist

pandas_profiling.ProfileReport(playlist)
# Make report for injuries

pandas_profiling.ProfileReport(injuries)
# Make report for tracking data

pandas_profiling.ProfileReport(tracking)