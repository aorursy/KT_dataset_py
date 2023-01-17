import numpy as np
import pandas as pd

# %matplotlib inline
import matplotlib.pyplot as plt
import matplotlib.cm as cm

import tensorflow as tf



data = pd.read_csv('../input/train.csv')
print(data[0:5])