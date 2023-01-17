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
#import files from local machine/Kaggle
import os
import pandas as pd
import numpy as np
from pandas import Series, DataFrame
cars=pd.read_csv("../input/mtcars.csv")
#To view the first 5 columns in the csv, use the .head() method
cars.head()
#To set column names to certain values, use the cars.columns=['col1',.....]
#To group a daraframe by values use the groupby() method using a certain column
#Mean displays the mean of all the values having the same value as 4,6,8 which are the values on which we are grouping
cars_groups=cars.groupby(cars['cyl'])
cars_groups.mean()
