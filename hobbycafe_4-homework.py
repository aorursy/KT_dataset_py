# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
data=pd.read_csv("../input/ArduinoSensorValues.csv")
data
data1 = data.loc[:,["gravity_x","gravity_y","gravity_z"]]
data1.plot()
plt.show()
data1.plot(subplots = True)
plt.show()
data1.plot(kind= "scatter",x="gravity_x",y="gravity_y")
plt.show()
data1.plot(kind = "hist",y="gravity_y",bins=10,range=(-10,10),density=True)
plt.show()
fig, axes = plt.subplots(nrows=2,ncols=1)
data1.plot(kind = "hist",y="gravity_y",bins=30,range=(-10,10),density=True,ax = axes[0])
data1.plot(kind = "hist",y="gravity_y",bins=30,range=(-10,10),density=True,ax = axes[1],cumulative=True)
plt.savefig("graph.png")
plt.show()
data.describe()
import warnings
warnings.filterwarnings("ignore")
data2 = data.head()
date_list = ["1992-01-10","1992-02-10","1992-03-10","1993-03-15","1993-03-16"]
datetime_object = pd.to_datetime(date_list)
data2["date"] = datetime_object
data2= data2.set_index("date")
data2
print(data2.loc["1993-03-16"])
print(data2.loc["1992-03-10":"1993-03-16"])
data2.resample("A").mean()
data2.resample("M").mean()
data2.resample("M").first().interpolate("linear")
data2.resample("M").mean().interpolate("linear")