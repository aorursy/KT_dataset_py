# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
!pip install clustergrammer2
from clustergrammer2 import net
import numpy as np

import pandas as pd



# generate random matrix

num_rows = 500

num_cols = 10

np.random.seed(seed=100)

mat = np.random.rand(num_rows, num_cols)



# make row and col labels

rows = range(num_rows)

cols = range(num_cols)

rows = [str(i) for i in rows]

cols = [str(i) for i in cols]



# make dataframe 

df = pd.DataFrame(data=mat, columns=cols, index=rows)
df.head()
net.load_df(df)

net.widget()
d