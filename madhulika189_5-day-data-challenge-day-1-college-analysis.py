import pandas as pd

import numpy as np
Region_Data = pd.read_csv('../input/salaries-by-region.csv',encoding='utf-8')

# Checking first 5 rows of the dataset

Region_Data.head(n=5)
Region_Data.describe()