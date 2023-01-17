# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
data=pd.read_csv('../input/fifa19/data.csv')
#Data Frames from dictionary



team=["Chelsea" , "Liverpool" , "M.City" , "M.United"]

ordinary=[1,2,3,4]

list_label=["team","ordinary"]

list_value=[team,ordinary]

zipped=list(zip(list_label , list_value))

print("zipped")

print(zipped)

data_dict=dict(zipped)

print("data_dict")

print(data_dict)

df=pd.DataFrame(data_dict)

print("df")

print(df)
df["league"]=["Premier_league" , " Europe_league","world_league","FA CUP"]

print(df)
#Visual exploraty  DATA ANALYSIS



data1=data.loc[:,["ID","Age"]]

data1.plot()

#it is confusing
# when grap is confusing , you can use the subplots

#SUBPLOTS

data1.plot(subplots=True)

plt.show()
#SCATTER PLOT 



data1.plot(kind='scatter' , x="ID" , y="Age")

plt.show()
#hist plot



data1.plot(kind='hist' , y='Age' , bins=50 , range=(0,250),normed=True )

plt.show()
fig , axes = plt.subplots(nrows=2 , ncols=1)

data1.plot(kind='hist' , y='Age' , bins=50 , range=(0,250), normed=True , ax=axes[0])

data1.plot(kind='hist' , y='Age' , bins=50 , range=(0,250), normed=True , ax=axes[1] , cumulative=True)

plt.savefig("graph.png")

plt.show