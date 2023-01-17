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
data = pd.read_csv('../input/Suicides in India 2001-2012.csv')
columns = data.columns

print(columns)

#'State', 'Year', 'Type_code', 'Type', 'Gender', 'Age_group', 'Total'
#shape of dataset

data.shape
#extract state columns from data

state_col = data.State



#Get state name 

state = state_col.unique()

state
#lenth of state array

len(state)
type_col      = data.Type

type_code_col = data.Type_code



#unique value of Type and Type_code



Type = type_col.unique()

Type_code = type_code_col.unique()



#print the lenth of Type and Type_code

print(len(Type))

print(len(Type_code))
#first print type an type_code columns and see what actually this are

print(Type)

print(Type_code)
import matplotlib.pyplot as plt