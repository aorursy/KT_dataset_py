# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns  # visualization tool



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))





import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
data = pd.read_csv("../input/iris/Iris.csv")

data.head()
data.info()

data.corr()
#correlation map

f,ax = plt.subplots(figsize=(10, 10))

sns.heatmap(data.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)

plt.show()
data.head(10)
data.columns
data.SepalLengthCm.plot(kind = 'line', color = 'g',label = 'SepalLengthCm',linewidth=1,alpha = 0.5,grid = True,linestyle = ':')

data.PetalLengthCm.plot(color = 'r',label = 'Species',linewidth=1, alpha = 0.5,grid = True,linestyle = '-.')

plt.legend(loc='upper right')     # legend = puts label into plot

plt.xlabel('x axis')              # label = name of label

plt.ylabel('y axis')

plt.title('Line Plot')            # title = title of plot

plt.show()
# x = SepalLengthCm, y = PetalLengthCm

data.plot(kind='scatter', x='SepalLengthCm', y='Species',alpha = 0.5,color = 'red')

plt.xlabel('SepalLengthCm')              # label = name of label

plt.ylabel('Species')

plt.title('SepalLengthCm Species Scatter Plot')            # title = title of plot
data.SepalLengthCm.plot(kind = 'hist',bins = 50,figsize = (12,12))

plt.show()
data.SepalLengthCm.plot(kind = 'hist',bins = 50)

plt.clf()
x = data['SepalLengthCm']>4     # SepalLengthCm 4'ten büyük olanları al

data[x]
data_frame = data[['SepalLengthCm']]

print(type(data_frame))
data[np.logical_and(data['PetalLengthCm']>6, data['SepalLengthCm']>7 )]
data[(data['PetalLengthCm']>6) & (data['SepalLengthCm']>7)]