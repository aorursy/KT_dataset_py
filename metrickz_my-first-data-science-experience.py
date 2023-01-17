# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns  # visualization tool



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
data = pd.read_csv("../input/data.csv")

data.info()
data.columns
data.corr()

data.head(10)
#usage of line plot for visualization

data.Overall.plot(kind = 'line',color ='black',label = 'Overall',linewidth=1,alpha = 0.5,grid = True,linestyle = '-')

data.Potential.plot(color ='yellow',label = 'Potential',linewidth=1,alpha = 0.5,grid = True,linestyle = ':')



plt.legend(loc = 'upper right')

plt.xlabel("x axis")

plt.xlabel("y axis")

plt.title('Line Plot')            # title = title of plot

plt.show()
"""as you can see, the correlation is between overall and potential is almost directly proportional (by the way the value of correlation is 0.660939 )"""

data.plot(kind = 'scatter',x ='Overall',y  ='Potential',alpha = 0.4,color = 'yellow')

plt.xlabel("Overall")

plt.ylabel("Potential")

plt.title('Overall and Special Scatter Plot')
data.Overall.plot(kind = 'hist',bins = 40,figsize = (10,10),fontsize = '10')

plt.show()
data1 = data["Name"].head()

data2 = data["Club"].head()



conc_data_col = pd.concat([data1,data2],axis = 1)

conc_data_col
data.head()
position = data.GKDiving



data['position'] = ["Striker" if i<10.0 else "Midfielder" if (i>=10.0 and i<35.0) else "Defense" if (i>=35.0 and i<70.0) else "GoalKeeper" for i in position]
data1 = data['Name'].head(10)

data2 = data['position'].head(10)



conc_data_row = pd.concat([data1,data2],axis=1)

conc_data_row
data['position'].value_counts()
new_data = data.head()



melted_data = pd.melt(frame=new_data,id_vars='Name',value_vars=['Overall','position'])

melted_data
#pivoting 

melted_data.pivot(index='Name',columns='variable',values='value')
