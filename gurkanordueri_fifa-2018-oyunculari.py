# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns 



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
data = pd.read_csv("/kaggle/input/fifa-18-demo-player-dataset/PlayerPersonalData.csv")

data.info()
data.corr()
data.columns
data.head(20)
data.Overall.plot(kind = 'line', color = 'y',label = 'Overall',linewidth=1,alpha = 0.5,grid = True,linestyle = ':')

data.Potential.plot(color = 'g',label = 'Potential',linewidth=1, alpha = 0.5,grid = True,linestyle = '-.')



plt.legend(loc='upper right')     

plt.xlabel('x axis')              

plt.ylabel('y axis')

plt.title('Line Plot')            

plt.show()
data.plot(kind='scatter', x='Overall', y='Age',alpha = 0.5,color = 'red')

plt.xlabel('Overall')              

plt.ylabel('Age')

plt.title('Age Overall Scatter Plot')            

plt.show()
data.Potential	.plot(kind = 'hist',bins = 70,figsize = (12,12))

plt.show()
x = data["Overall"] > 85

print(x)

data[x]
data[np.logical_and(data['Age']<24, data['Overall']>80 )]