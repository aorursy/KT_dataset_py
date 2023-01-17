import pandas as pd

import numpy as np

from scipy.stats import ttest_ind
df = pd.read_csv("../input/cereal.csv")
df.isnull()