

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



pd.set_option('max_rows',160)



df = pd.read_csv('/kaggle/input/cost-of-living/cost-of-living.csv',index_col=[0]).T

df[['Toyota Corolla 1.6l 97kW Comfort (Or Equivalent New Car)']].sort_values(by='Toyota Corolla 1.6l 97kW Comfort (Or Equivalent New Car)',ascending=False)