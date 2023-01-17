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
# Hint: Try two approaches, one using `.value_counts()` and the other using `.groupby()`.

# Hint: You'll need to pass a list of column names to `.groupby()`. 

# Hint: You'll need to first create and store a grouped DataFrame with longitude/latitude coordinates, and then compute and store the count (group size) as a new column. 
