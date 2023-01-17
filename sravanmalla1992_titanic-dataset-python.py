# Ignore warnings

import warnings

warnings.filterwarnings('ignore')



# Handle table-like data and matrices

import numpy as np

import pandas as pd
# To import train dataset from file

# get titanic & test csv files as a DataFrame

train = pd.read_csv("../input/train.csv")

test  = pd.read_csv("../input/test.csv")



full = train.append( test , ignore_index = True )

titanic = full[ :891 ]



del train , test



print ('Datasets:' , 'full:' , full.shape , 'titanic:' , titanic.shape)
titanic.info()

titanic.describe()
titanic.isnull()