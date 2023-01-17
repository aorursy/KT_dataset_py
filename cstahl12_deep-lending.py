import numpy as np

import pandas as pd
from keras.models import Sequential

from keras.optimizers import RMSprop, Adam

from keras.layers import Dense, Activation, Dropout
data = pd.read_csv('../input/loan.csv', index_col=0, low_memory=False)

data.head()