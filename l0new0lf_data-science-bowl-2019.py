import numpy as np

import pandas as pd

import os
base = "/kaggle/input/data-science-bowl-2019/"

os.listdir("/kaggle/input/data-science-bowl-2019/")
pd.read_csv(base + 'train.csv').head()
pd.read_csv(base + "specs.csv").head() 