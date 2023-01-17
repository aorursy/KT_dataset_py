import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns; sns.set()

from sklearn.preprocessing import StandardScaler

import os
# print(os.listdir("../input"))

names = [ 'date','time','open','high','low','close' ]
train = pd.read_csv("../input/FX_USDJPY_201701till01312259_Train - USDJPY_201701till01312259 (1).csv", names=names)
test = pd.read_csv("../input/FX_USDJPY_201701312300_Test - USDJPY_201701312300 (1).csv", names=names)

train.head()
plot_data = train
plot_data.plot()