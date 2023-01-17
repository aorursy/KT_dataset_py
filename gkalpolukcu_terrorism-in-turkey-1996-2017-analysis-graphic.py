# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns  # visualization tool



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
data = pd.read_csv("/kaggle/input/terrorism-in-turkey-19962017/TableOfTurkey.csv")

data.head(5)
data.info()
data.columns
data.corr()
f,ax = plt.subplots(figsize=(14, 14))

sns.heatmap(data.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)

plt.show()
data.columns

plt.scatter(data.Year,data.Killed, color="red", alpha=0.5)
# Line Plot

# color = color, label = label, linewidth = width of line, alpha = opacity, grid = grid, linestyle = sytle of line

data.Year.plot(kind = 'line', color = 'g',label = 'Year',linewidth=1,alpha = 0.5,grid = True,linestyle = ':')

data.Wounded.plot(color = 'b',label = 'Wounded',linewidth=1, alpha = 0.5,grid = True,linestyle = '-.')

plt.legend(loc='upper right')     # legend = puts label into plot

plt.xlabel('x axis')              # label = name of label

plt.ylabel('y axis')

plt.title('Line Plot')            # title = title of plot

plt.show()
data.plot(kind="scatter", x="Year", y="Wounded", alpha=0.5,color="blue")

plt.xlabel('Year')

plt.ylabel('Wounded')
data.Year.plot(kind="hist", bins=50,figsize=(12,12), color="green")

plt.show()
data.Year.plot(kind = 'hist',bins = 50)
dictionary = {'Ankara' : 'Çankaya', 'İzmir' : 'Karşıyaka'}

print(dictionary.keys())

print(dictionary.values())
dictionary['İstanbul'] = "Taksim"

print(dictionary)

dictionary['Ankara'] = "Keçiören"

print(dictionary)

del dictionary['İstanbul']

print(dictionary)

print('Ankara' in dictionary)

dictionary.clear()

print(dictionary)
#del dictionary
series = data["Wounded"]

print(type(series))

data_frame = data[["Wounded"]]

print(type(data_frame))
print(7 > 3)

print(7!=3)

# Boolean operators

print(True and False)

print(True or False)
x = data['Year']>2000

data[x]
data[np.logical_and(data['Year']>2000, data['Target_type']=="Military")]
data[(data['Year']>2000) & (data['Wounded']==5.0)]
i=0

while i != 10:

    print("i is : ", i)

    i +=1

print(i, "is equal to 10")
list = [2,4,6,8]

for i in list:

    print("i is : ", i)

print('')



for index, value in enumerate(list):

    print(index," : ",value)

print('')  



dictionary = {'Ankara' : 'Çankaya', 'İzmir' : 'Karşıyaka'}

for key,value in dictionary.items():

    print(key," : ",value)

print("")    



for index,value in data[['Wounded']][0:1].iterrows():

    print(index," : ",value)