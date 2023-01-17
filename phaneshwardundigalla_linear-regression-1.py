# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

import numpy as np
import pandas as pd
df=pd.read_csv("../input/heights-and-weights/data.csv")
df.keys()
x=df.Height
y=df.Weight
x=df.iloc[:,0:1].values
y=df.iloc[:,1].values
m=1.2
c=2.1
y=m*x+c
from sklearn.linear_model import LinearRegression
MachineBrain=LinearRegression()
MachineBrain.fit(x,y)


y_predict=m*x+c

import matplotlib.pyplot as plt
plt.scatter(x,y)
plt.plot(x,y_predict,c="red")
