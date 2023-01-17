# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
data=pd.read_csv('../input/FIFA 2018 Statistics.csv')
data.info()
data.head(10)
data.corr()
#correlation map
f,ax = plt.subplots(figsize=(18, 18))
sns.heatmap(data.corr(), annot=True, linewidths=.5, fmt= '.2f',ax=ax)
plt.show()
data.columns
data.tail(10)
#Line Plot
data['Goal Scored'].plot(kind = 'line', color = 'g',label = 'Goal Scored',linewidth=1,alpha = 1,grid = True,linestyle = '-')
data['On-Target'].plot(color="r",linewidth=1,alpha=1,grid=True,linestyle="-.")
plt.legend()
plt.xlabel('Matches')
plt.ylabel('Y label')
plt.title('line Plot')
plt.show()






#Scatter Plot
data.plot(kind='scatter',x='Attempts',y='Goal Scored',alpha=0.5,color='g')
plt.title('Goals and Attempts scatter plot')
# Histogram
# bins = number of bar in figure
data['Yellow Card'].plot(kind='hist',bins=15,figsize=(8,8))


data.keys()
x=data['Goal Scored']>=4
data[x]
data[np.logical_and(data['Corners']>3,data['Goal Scored']>3)]
