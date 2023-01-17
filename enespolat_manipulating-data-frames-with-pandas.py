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
# read data
data = pd.read_csv('../input/pokemon.csv')
data= data.set_index("#")
data.head()
# indexing using square brackets
data["HP"][1]
# using column attribute and row label
data.HP[1]
# using loc accessor
data.loc[1,["HP"]]
# Selecting only some columns
data[["HP","Attack"]]
# Difference between selecting columns: series and dataframes
print(type(data["HP"]))     # series
print(type(data[["HP"]]))   # data frames
# Slicing and indexing series
data.loc[1:10,"HP":"Defense"]   # 10 and "Defense" are inclusive
# Reverse slicing 
data.loc[10:1:-1,"HP":"Defense"] 
# From something to end
data.loc[1:10,"Speed":] 
# Creating boolean series
boolean = data.HP > 200
data[boolean]