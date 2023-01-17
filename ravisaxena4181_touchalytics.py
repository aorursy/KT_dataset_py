# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

#print(os.listdir("../input/data.csv"))



data=pd.read_csv("../input/data.csv")

data.head(10)



# Any results you write to the current directory are saved as output.
''' assigning the data headers'''

data.columns=["phoneID","userID","documentID","timeMS","action","phoneOri","x","y","pressure","areaCovered","fingerOri"]

data.head()
#data.describe()

userdata= data.groupby('userID').mean()

userdata

'''select a few records to analyse trend'''

dataselected=data.head(2000)

dataselected
#plt.scatter(dataselected.x,dataselected.y,color ='purple')

plt.plot(dataselected.x,color ='red')

#plt.xlabel ('y')

#plt.ylabel ('x')

plt.plot(dataselected.y,color ='blue')



plt.show()

print("Combined x,y graph")

plt.scatter(dataselected.x,dataselected.y,color ='purple')

plt.xlabel ('x')

plt.ylabel ('y')



plt.show()
'''Seperated graphs for each attribute data analysis'''

plt.figure(figsize=(20,10))

plt.subplots_adjust(hspace= 0.2, wspace= 0.2)

plt.subplot(2,2,1)

plt.title ('X Axis of Screen')

plt.plot(dataselected.x,'r')

plt.subplot(2,2,2)

plt.title ('Y Axis of Screen')

plt.plot(dataselected.y, 'b')

plt.subplot(2,2,3)

plt.title ('Pressure')

plt.plot(dataselected.pressure, 'g')

plt.subplot(2,2,4)

plt.title ('Action')

plt.plot(dataselected.action,'y')

plt.show()