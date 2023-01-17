#os-operating system
import os

os.getcwd()
import os
import numpy as np
import pandas as pd
from IPython.display import Image
from matplotlib import pyplot as plt
import pandas as pd

data = pd.read_csv('../input/plant-1-generation-data/Plant_1_Generation_Data.csv')
data.info()
data.head()
data.head(10)
type(data)
type(data['SOURCE_KEY'])
data['SOURCE_KEY'].head()
#numpy array
type(data['SOURCE_KEY'].values)
# basic python/numpy data type - str, int, float...
type(data['SOURCE_KEY'].values[0])
len(data)
len(data) == len(data['SOURCE_KEY'])
#explore dataset
data.info()
data.describe()
data.count()
data.isnull()

df = pd.DataFrame(data,columns=['DC_POWER'])
df
df.max()
df.min()
df.mean()
df = pd.DataFrame(data,columns=['DC_POWER','SOURCE_KEY'])
df

df.max()
data['DC_POWER'].mean()
