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
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
import pandas as pd
veri=pd.read_csv("../input/glass.csv")

veri.columns #names of parameters
veri.info() # data types of samples
veri.head(10) #first 10 sample

veri.corr() #Correlation of parameters

f,ax = plt.subplots(figsize=(16,16))  #Correlation map
sns.heatmap(veri.corr(), annot=True,linewidths=2,fmt='.2f',ax=ax)
#Scatter Plot
veri.plot(kind="scatter",x="Type",y="Mg",color="orange")
plt.xlabel=("Type")
plt.ylabel=("Mg")
plt.show()
#Line Plot
veri.Ba.plot(kind="line",grid=True,color="red",linestyle=":",linewidth=1,label="Ba")
veri.Al.plot(kind="line",grid=True,color="blue",linestyle="--",linewidth=1,label="Al")
plt.xlabel=("Samples")
plt.ylabel=("Values")
plt.legend(loc="upper left")
plt.show()
#histogram
veri.Type.plot(kind="hist",color="blue",bins=7,figsize=(10,10))
plt.show()
filtr=veri.Al>1.36
print(veri[filtr]) #filtr for Al values over than 1.36
