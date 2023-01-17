# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input/"))



# Any results you write to the current directory are saved as output.
df = pd.read_csv("../input/energyWork_MachineL.csv",sep = ";")
df.head(10)
print(plt.style.available)

plt.style.use('ggplot')

df.info()
plt.scatter(df.energy,df.work)

plt.xlabel("energy")

plt.ylabel("work")

plt.show()
plt.scatter(df.energy,df.work)

plt.xlabel("energy")

plt.ylabel("work")



from sklearn.linear_model import LinearRegression



linear_reg = LinearRegression()

x = df.energy.values.reshape(-1,1)

y = df.work.values.reshape(-1,1)



linear_reg.fit(x,y)



y_head = linear_reg.predict(x)



plt.plot(x, y_head, color = "green")



plt.show()
from sklearn.metrics import r2_score



print("r_square score: ", r2_score(y,y_head))