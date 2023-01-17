import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import matplotlib.pyplot as plt # plotting

import seaborn as sns # fancier plotting

%matplotlib inline
price = pd.read_csv("../input/DJIA_table.csv")
y= price['Close']

x= price['Date']

plt.scatter(x,y)