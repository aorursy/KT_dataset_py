import pandas as pd

data0 = pd.read_csv('../input/housing/anscombe.csv')

data = pd.read_csv('../input/housing/housing.csv')
#check the datasets

data.info()
data.head()
data.tail()
data.describe()
data['ocean_proximity'].value_counts()
import matplotlib as plt

data.hist(bins=80,figsize=(20,20))