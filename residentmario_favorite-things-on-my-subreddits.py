import os

import numpy as np

import pandas as pd



np.array(sorted(os.listdir("../input/top-things/top-things/reddits")))
pd.set_option('max_colwidth',120)

pd.read_csv("../input/top-things/top-things/reddits/t/tennis.csv").name.to_frame()
pd.read_csv("../input/top-things/top-things/reddits/f/Fitness.csv").name.head(10).to_frame()
pd.read_csv("../input/top-things/top-things/reddits/m/malefashionadvice.csv").name.head(10).to_frame()
pd.read_csv("../input/top-things/top-things/reddits/p/programming.csv").name.head(10).to_frame()
pd.read_csv("../input/top-things/top-things/reddits/t/The_Donald.csv").name.head(10).to_frame()