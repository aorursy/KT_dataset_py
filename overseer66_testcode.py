import pandas as pd

import numpy as np



from keras.models import Sequential

from keras.layers import Dense, Activation, Dropout

from keras.layers import LSTM

from keras.optimizers import RMSprop
data = pd.read_csv("../input/train.csv")
D = data['Sequence'].values

D = [x.split(',') for x in D]
D[8]
[len(x) for x in D]