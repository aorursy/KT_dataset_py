from matplotlib import pyplot

import matplotlib.pyplot as plt 

from pandas import read_csv

data = read_csv("../input/pima-indians-diabetes-database/diabetes.csv")

data.hist(figsize = (10,10), color = "#5F9EA0")

plt.show() 

plt.savefig('fig_1.png')
data.plot(kind='density', subplots=True, layout=(3,3), sharex=False , figsize =(10,10))

plt.show()
plt.savefig('fig_2.png')
data.plot(kind='box', subplots=True, layout=(3,3), sharex=False,sharey=False ,figsize =(10,10))

plt.show()