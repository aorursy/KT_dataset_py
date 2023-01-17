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
data=pd.read_csv('../input/Pokemon.csv')
data.columns
data.head()
data.tail()
print ("Minimum speed :",data.Speed.min())
print ("Maximum speed :",data.Speed.max())
# shape gives number of rows and columns in a tuble
data.shape
# info gives data type like dataframe, number of sample or row, number of feature or column, feature types and memory usage
data.info()