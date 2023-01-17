import pandas as pd
import numpy as np
from pandas import Series, DataFrame
dframe=pd.read_csv('../input/lec25.csv')
dframe
#to not use data as index and columns
dframe=pd.read_csv('../input/lec25.csv',header=None)
dframe
dframe=pd.read_table('../input/lec25.csv',sep=',',header=None)
dframe
pd.read_csv('../input/lec25.csv',header=None,nrows=2)

dframe.to_csv('mytextdata_out.csv')

import sys
dframe.to_csv(sys.stdout)
dframe.to_csv(sys.stdout, sep='_')
dframe.to_csv(sys.stdout, columns=[1,2,3])







