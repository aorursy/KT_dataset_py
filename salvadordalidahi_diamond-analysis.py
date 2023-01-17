# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt #Görselleştirme

import seaborn as sns #Görselleştirme 

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
data = pd.read_csv("../input/diamonds.csv") 
data.info()
data.head() #receive first 5 value

#data.drop(["Sütun Adı"]) , axis = 1 vertical drop , inplace = True update data after drop

data.drop(["Unnamed: 0"] , axis = 1 , inplace = True)
data.columns #give column name
data.describe() #numerical statistics
data.loc[(data["x"] == 0) | (data["y"] == 0) | (data["z"] == 0)]



#Notice that instead of using '&' in the code above, '|' because if we used 'and (&)' it would show data where all are 0
#just use len()

len(data[(data["x"] == 0) | (data["y"] == 0) | (data["z"] == 0)])

data = data[np.logical_and(np.logical_and((data["x"] != 0) , (data["y"] != 0))  , (data["z"] != 0))] 

#drop with three condition

#we use NumPy Logical_and method

data.info()
data.describe()
data.corr()
f,ax = plt.subplots(figsize = (12,12))

sns.heatmap(data.corr() , annot = True ,linewidths = 2 , fmt = ".2f" , ax = ax)

plt.show()
sns.jointplot(x="carat" , y = "price", data=data , height = 5 , kind = "reg")

plt.show()
sns.catplot(x = "cut" ,kind = "count", data = data , aspect = 3)

plt.show()
sns.catplot(x = "cut" , y = "price" , data = data ,aspect = 3)
sns.catplot(x = "color" , kind = "count" , data = data , aspect = 3)

plt.show()
sns.catplot(x = "color" , y = "price" , data = data , kind = "violin" , aspect = 3)

plt.show()