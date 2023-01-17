# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
import pandas as pd
df = pd.read_csv('../input/parks.csv', index_col=['Park Code'])
df.head(3)
df.iloc[2]
df.loc['BADL']
df.loc[['BADL', 'ARCH', 'ACAD']]
df.iloc[[2,0,1]]
df[:3]
df[3:6]
df['State'].head(3)
df.State.head(3)
df.columns = [col.replace(' ', '_').lower() for col in df.columns]
print(df.columns)