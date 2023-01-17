import dask
import dask.dataframe as dd
import dask.array as da
import pandas as pd
import numpy as np
from dask.multiprocessing import get
from multiprocessing import cpu_count
nCores= cpu_count()
nCores
# Numpy array
import numpy as np
np.arange(100)
# Dask array
import dask.array as da
da.arange(100, chunks=5).compute()
x= da.arange(100,chunks=(5,))
x.visualize("dask.svg")
x=da.arange(10, chunks=(5,))
x.visualize("dask.svg")
x.chunks
# Numpy array

x= np.arange(1000)

# Dask array
y= da.from_array(x, chunks=(100))

type(x), type(y)
y.mean().compute()
x= da.arange(10, chunks=(5))
x
x.compute()
x= da.arange(11,chunks= (5))
x.compute()
x.chunks
x= da.arange(11, chunks=(5))
x.sum().compute()
x= da.arange(11, chunks=(5))
x.mean().compute()
import os
os.chdir("../input/experimental-datasets/DataFiles/FIFA/")
import dask.dataframe as dd

import pandas as pd
%time df1= pd.read_csv("fifa19data.csv")
import dask.dataframe as dd
%time df= dd.read_csv("fifa19data.csv")
df.dtypes