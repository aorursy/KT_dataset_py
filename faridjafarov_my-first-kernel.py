import numpy as np 

import pandas as pd 

import matplotlib.pyplot as plt

import seaborn as sns



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



data = pd.read_csv('/kaggle/input/fifa19/data.csv')
data.info()
data.head(10)
data.columns
# correlation map

f,ax = plt.subplots(figsize = (18,18))

sns.heatmap(data.corr(),annot = True,linewidths =.3,fmt = '.1f',ax = ax)

plt.show()
data.Age.plot(figsize = (15,15),kind = 'line',color = 'red',alpha = 0.5,linewidth = 1,linestyle = ':',grid = True,label = 'Age')

data.Overall.plot(kind = 'line',color = 'green',alpha = 0.5,linewidth = 1,linestyle = '-.',grid = True,label = 'Overall')

plt.legend(loc = 'upper right')

plt.xlabel('x axis')

plt.ylabel('y axis')

plt.title('Age Overall Line Plot')

plt.show()
data.plot(kind = 'scatter',x = "Age",y = 'Overall',alpha = 0.5,color = 'red')

plt.xlabel('Age')

plt.ylabel('Overall')

plt.title('Age Overall Scatter Plot')

plt.show()
data.Age.plot(kind = 'hist',bins = 55)

plt.title('Overall Histogram')

plt.show()
# filtering

# age is upper than 30 and overall is upper than 90  

filter1 = data.Age > 30

filter2 = data.Overall > 90

data1 = data[filter1 & filter2]

data1
# filtering

# overall is upper than 85 and plays in FC Barcelona

filter3 = data.Overall > 85

filter4 = data.Club == 'FC Barcelona'

data2 = data[filter3 & filter4]

data2