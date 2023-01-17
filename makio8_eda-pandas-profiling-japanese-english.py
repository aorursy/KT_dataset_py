import pandas as pd

import numpy as np



import matplotlib.pyplot as plt

import seaborn as sns

sns.set()



pd.options.display.max_columns = 100
import pandas_profiling as pdp
train = pd.read_csv("../input/learn-together/train.csv")

test = pd.read_csv("../input/learn-together/test.csv")
pdp.ProfileReport(train)