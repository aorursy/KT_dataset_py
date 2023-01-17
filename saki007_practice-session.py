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
import numpy as np
import pandas as pd
import sys
import matplotlib.pyplot as plt
import seaborn as sns

#Creating a Series by passing a list of values
s = pd.Series([1,3,5,np.nan,6,8]) 
print(s)

dates = pd.date_range('20190101',periods=8)
print(dates)
#Creating a DataFrame 
df = pd.DataFrame(np.random.randn(8,4),index=dates,columns=list('ABCD'))
print(df)