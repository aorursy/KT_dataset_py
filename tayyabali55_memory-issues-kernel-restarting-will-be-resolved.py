import dask
import dask.dataframe as dd
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
%%time
train = dd.read_csv("../input/train.csv",  dtype={'acoustic_data': np.int16, 'time_to_failure': np.float64})
train.head()