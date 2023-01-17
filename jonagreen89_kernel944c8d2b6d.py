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
stations = pd.read_csv('../input/stations.csv')
m2001 = pd.read_csv('../input/csvs_per_year/csvs_per_year/madrid_2001.csv')
stations.sort_values('elevation', ascending=False)  
# df.groupby('elevation')
df = pd.merge(m2001, stations, left_index=True, right_index=True, how='inner')
df.sort_values('NOx', ascending=False) 
# need to research some of the fields