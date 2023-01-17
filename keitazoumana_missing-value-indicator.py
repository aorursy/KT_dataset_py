import pandas as pd

import numpy as np
from sklearn.impute import MissingIndicator
features = [[4,2,1],

            [24,12,6],

            [8,-1, 2],

            [28,14,7],

            [32,16,-1],

            [600,300,150],

            [-1,-1,1]]
'''

Instanciate the missing indicator by telling it 

that our missing values are represented by '-1'

'''

indicator = MissingIndicator(missing_values=-1)
mark_missing_values_only = indicator.fit_transform(features)

mark_missing_values_only
indicator.features_