import numpy as np
import pandas as pd
states = pd.read_csv('../input/population-of-provinces-and-states-for-covid19/population.csv')
train = pd.read_csv('../input/covid19-global-forecasting-week-3/train.csv')
test  = pd.read_csv('../input/covid19-global-forecasting-week-3/test.csv')
train_states = set(train.Province_State) - set([np.nan])
added_states = set(states.State)
assert train_states <= added_states
print('alright!')