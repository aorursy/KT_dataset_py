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
#read data into df
explore = pd.read_csv("../input/2016 School Explorer.csv")
scores = pd.read_csv("../input/D5 SHSAT Registrations and Testers.csv")

explore.shape, scores.shape
#quickly look at the data
explore.sample(10)
scores.head()
#look for missing data
Missing_value_count = explore.isnull().sum().sort_values(ascending=False)
Missing_value_count[0:25]
explore = explore[explore['New?'].notna()]
explore = explore.drop(columns = Missing_value_count[0:2].index)

