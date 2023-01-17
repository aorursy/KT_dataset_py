import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
pd.__version__


import math
x = float('nan')
math.isnan(x)
True
%matplotlib inline

crime = pd.read_csv("../input/train.csv")

crime.head()
crime['Category'].head()
crime.dtypes
category = crime['Category'].head()
category.head()

property_crime = category[['VANDALISM']]
property_crime.head()
category.dtype
