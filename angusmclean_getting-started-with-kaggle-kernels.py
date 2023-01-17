!ls /kaggle/input/music-classification/kaggle
import numpy as np

import pandas as pd
ROOT_DIR = '/kaggle/input/music-classification/kaggle/'



## Read the training data labels csv file

dfLabels = pd.read_csv(ROOT_DIR + 'labels.csv')

dfLabels
## Read individual song file

dfSong = pd.read_csv(ROOT_DIR + 'training/' + dfLabels.id[0], header=None)

dfSong
## Display wave form of song

# Aesthetics :

# - Select just the start of song, also apply some smoothing so its easier to see

# - Each column (ie song channel) is different color



dfSong[:200].rolling(5).mean().plot.line(figsize=(16,8), title="Song Wave Form (by Channel)")
## Plot distribution of values for each channel

# Aesthetics : 

# - Based on above graph, values typically fall between -100 and 100, trim bounds



dfSong.plot.density(xlim=(-100,100), figsize=(12, 8), title="Distribution of values for each channel")