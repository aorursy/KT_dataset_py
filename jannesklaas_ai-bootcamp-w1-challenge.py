import pandas as pd

import numpy as np
df = pd.read_csv('../input/W1data.csv')

df.head()
# Get labels

y = df[['Cultivar 1', 'Cultivar 2', 'Cultivar 3']].values

# Get inputs

X = df.drop(['Cultivar 1', 'Cultivar 2', 'Cultivar 3'], axis = 1)

X.shape, y.shape # Print shapes just to check