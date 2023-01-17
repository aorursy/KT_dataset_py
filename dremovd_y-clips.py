import pandas as pd

import numpy as np



train = pd.read_hdf('../input/train.h5')
from collections import Counter



Counter(train['y']).most_common(n = 4)