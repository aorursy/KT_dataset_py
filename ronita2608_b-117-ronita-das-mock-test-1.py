import pandas as pd
data=pd.read_csv('../input/iris/Iris.csv')
data.columns
data.iloc[0:10]#selecting 6 cloumns and 10 rows
import numpy as np

import random as rn

import matplotlib.pyplot as plt
ht=np.array([])

for i in range(50):

    ht=np.append(ht,[rn.randint(50,70)])
ht[10]=172

ht[11]=172

ht[22]=2

ht[49]=2
plt.boxplot(ht)

plt.title('Height')

plt.xlabel('X-axis')

plt.ylabel('Y-axis')

plt.show()
data.loc[data.SepalWidthCm.isnull()]
#since no nan or null value is found
data#total no. of observations