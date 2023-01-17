import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
print(os.listdir("../input"))
cgm = pd.read_csv('../input/Chinese_grandmaster.csv')
cgm.sort_values(by=['gold count','highest rank'],ascending=[False,True])
cgm.city.value_counts()