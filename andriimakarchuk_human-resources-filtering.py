import numpy as np

import pandas as pd
data = pd.read_csv("../input/human-resources-data-set/HRDataset_v13.csv").dropna()
cols = list( pd.Series(data.columns).unique() )

colsToDrop = list( set(data.columns)-set(cols) )



data = data.drop( labels=colsToDrop, axis=1 )
print( list(data.columns) )