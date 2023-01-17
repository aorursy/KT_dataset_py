import numpy as np 

import pandas as pd
data = pd.read_hdf('../input/train.h5')
t20_size = ((data.technical_20 != 0) & (~data.technical_20.isnull())).sum()

t30_size = ((data.technical_30 != 0) & (~data.technical_30.isnull())).sum()

other = data.shape[0] - t20_size - t30_size
t20_rows = (data.technical_20 != 0) & (~data.technical_20.isnull())

t30_rows = (data.technical_30 != 0) & (~data.technical_30.isnull())

joined_rows = t20_rows & t30_rows

other_rows = (~t20_rows) & (~t30_rows)
print ('Full dataset ->', data.shape[0],'\n','technical_20 has value ->', t20_size,'\n',

       'technical_30 has a value ->', t30_size,'\n',       

      'both technical_20 and technical_30 does not have a value->',other,'\n',

      'both technical_20 and technical_30 has a value->',joined_rows.sum(),'\n')
data.loc[t20_rows, 'technical_20'].describe()
data.loc[t30_rows, 'technical_20'].describe()
data.loc[t30_rows, 'technical_30'].describe()
data.loc[t20_rows, 'technical_30'].describe()
print ('y mean when technical_20 nonzero -> ','{:.4}'.format(data.loc[t20_rows,'y'].mean()),'\n',

       'y mean in residual data set -> ','{:.4}'.format(data.loc[other_rows,'y'].mean()),'\n',

      'y mean when technical_30 nonzero -> ','{:.4}'.format(data.loc[t30_rows,'y'].mean()))