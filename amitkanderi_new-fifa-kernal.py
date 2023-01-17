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
# saving file path in variable for easier access
data_filepath='../input/data.csv'
# Reading data and storing it in Data Frame
fifa_data=pd.read_csv(data_filepath)
#Print data frame content
fifa_data.head()
# Checking Info of the Data
fifa_data.info()
# As there are some NaN Data as well. 
fifa_data.describe()
# Shape of Dataset
fifa_data.shape
fifa_data.nunique()
# Now check for NaN values
fifa_data.isnull().any()