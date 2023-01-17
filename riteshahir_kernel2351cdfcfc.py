
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as pt
import os


dataset=pd.read_excel("../input/temp.xls")
dataset=dataset.set_index('YEAR')
dataset.head()

min_months=[]
min_months1=[]
min_months = dataset.loc[:,'JAN':'DEC'].min(axis=1)
min_months1=min_months.values



y=min_months1
x=dataset.index
print(x)
print(y)
pt.plot(x,y)

z=[]
for i in range(len(dataset.index)):
    z.append(13)
pt.plot(x,z)
pt.show()
#work pendding

#from collections import count 
#from numpy import count
#d= {}
#for i in set(y):
 #   d[i] =y.count(i)
