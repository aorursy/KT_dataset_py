import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
data = pd.read_hdf('../input/train.h5')
pd.set_option('display.max_columns', 150)

pd.set_option('display.max_rows', 100)
data.technical_16.describe()
t16 = data.loc[(data.id == 288) & (data.technical_16 != 0.0)  & (~data.technical_16.isnull()) ,['timestamp', 'technical_16']]

ax = t16.plot(use_index=False)
ax=t16.technical_16.plot(use_index=False)
t16 = data.loc[(data.id == 1201) & (data.technical_16 != 0.0)  & (~data.technical_16.isnull()) ,['timestamp', 'technical_16']]

ax=t16.technical_16.plot(use_index=False)
ax = t16.plot(use_index=False)