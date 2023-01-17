# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))
from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
import matplotlib.pyplot as plt
import seaborn as sns
data = pd.read_csv('../input/creditcard.csv')
data.info()
data.corr()
#showing correlation map
f,ax = plt.subplots(figsize=(18, 18))#size of boxes
sns.heatmap(data.corr(), annot = True, linewidths = 5, fmt = '.1f', ax=ax)#coloring map
plt.show()
data.head(10)
data.columns
data.V10.plot(kind = 'line', color = 'g',label ='V10', linewidth = 1, alpha = 0.5, grid = True, linestyle =':')
data.V20.plot(kind = 'line', color = 'b', label = 'V20', linewidth =1, alpha = 0.5, grid = True, linestyle = '-')
plt.legend('upper right')
plt.xlabel = ('x axis')
plt.ylabel = ('y axis')
plt.title = ('Line Plot')
plt.show()
data.plot(kind = 'scatter',x = 'V10', y = 'V20',alpha = 0.5, color = 'red')
plt.xlabel = 'V10'
plt.ylabel = 'V20'
plt.title = 'V10-V20 Scatter Plot'
data.V10.plot(kind = 'hist', bins = 50, alpha = 0.5, figsize = (30,30))
plt.show()
series = data['Amount']
data_frame = data[['Amount']]
x = data['Amount']<100
data[x]