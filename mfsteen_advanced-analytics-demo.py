12.13 * 15.9
import numpy as np 

import pandas as pd 

import matplotlib.pyplot as plt

%matplotlib inline
numOfDays = 252

numOfTrials = 10

changes = pd.DataFrame(np.random.normal(loc = 0.0, scale = 1.0, size = (numOfDays, numOfTrials)))
changes.head()