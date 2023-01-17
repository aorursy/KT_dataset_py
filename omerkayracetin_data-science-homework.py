# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
data=pd.read_csv("../input/diamonds.csv")
data.columns=['Unnamed', 'carat', 'cut', 'color', 'clarity', 'depth', 'table',
       'price', 'x', 'y', 'z']
data
data.info()
data.describe()
data.corr()
f,ax=plt.subplots(figsize=(15,15))
sns.heatmap(data.corr(),annot=True,fmt=".1f",linewidths=0.5,ax=ax)
plt.show()
data.columns
d_colored=data[data.color == "D"]
e_colored=data[data.color == "E"]
f_colored=data[data.color == "F"]
g_colored=data[data.color == "G"]
h_colored=data[data.color == "H"]
i_colored=data[data.color == "I"]
j_colored=data[data.color == "J"]
d_colored.carat.plot(alpha=0.8,color="red",linewidth=0.3,grid=True)
plt.xlabel("diamonds")
plt.ylabel("carat")
plt.title("d colored diamonds' carat line plot")
plt.show()

e_colored.carat.plot(alpha=0.8,color="blue",linewidth=0.3,grid=True)
plt.xlabel("diamonds")
plt.ylabel("carat")
plt.title("e colored diamonds' carat line plot")
plt.show()

f_colored.carat.plot(alpha=0.8,color="black",linewidth=0.3,grid=True)
plt.xlabel("diamonds")
plt.ylabel("carat")
plt.title("f colored diamonds' carat line plot")
plt.show()

g_colored.carat.plot(alpha=0.8,color="purple",linewidth=0.3,grid=True)
plt.xlabel("diamonds")
plt.ylabel("carat")
plt.title("g colored diamonds' carat line plot")
plt.show()

h_colored.carat.plot(alpha=0.8,color="grey",linewidth=0.3,grid=True)
plt.xlabel("diamonds")
plt.ylabel("carat")
plt.title("h colored diamonds' carat line plot")
plt.show()

i_colored.carat.plot(alpha=0.8,color="pink",linewidth=0.3,grid=True)
plt.xlabel("diamonds")
plt.ylabel("carat")
plt.title("i colored diamonds' carat line plot")
plt.show()

j_colored.carat.plot(alpha=0.8,color="green",linewidth=0.3,grid=True)
plt.xlabel("diamonds")
plt.ylabel("carat")
plt.title("j colored diamonds' carat line plot")
plt.show()

data.plot(kind="scatter",x="price",y="carat",grid=True,color="r",alpha=0.8)
plt.xlabel("Price")
plt.ylabel("Carat")
plt.title("Price-Carat Scatter Plot")
plt.show()
data.carat.plot(kind="hist",bins=50,grid=True,alpha=0.9)
plt.title("Carat Frequency")
plt.xlabel("carat")
plt.show()
fair=data[data.cut=="Fair"]
premium=data[data.cut=="Premium"]
fair.price.plot(kind="line",label="fair cut",color="red",linewidth=0.5,grid=True,alpha=0.6)
premium.price.plot(kind="line",label="premum cut",color="blue",linewidth=0.5,grid=True,alpha=0.6)
plt.legend()
plt.xlabel("diamonds")
plt.ylabel("prices")
plt.title("fair-premium cut diamonds' prices")
plt.show()
fair.carat.plot(kind="line",label="fair cut",color="red",linewidth=0.5,grid=True,alpha=0.6)
premium.carat.plot(kind="line",label="premium cut",color="blue",linewidth=0.5,grid=True,alpha=0.6)
plt.legend()
plt.xlabel("diamonds")
plt.ylabel("carats")
plt.title("fair-premium cut diamonds' carats")
plt.show()
