import numpy as np

import pandas as pd

df = pd.read_csv('../input/crime_homicide_subset.csv')



for line in df.values:

    print(line)