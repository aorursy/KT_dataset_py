# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import pandas_profiling as p
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
df=pd.read_csv("../input/StudentsPerformance.csv")  #Reading Data
#lets see first how data is 
df.head()
df.shape
#Now lets check is there is any missing values in data
df.isnull().sum()
df.dtypes
df.describe()
p.ProfileReport(df)
df.hist(figsize=(10,10))
