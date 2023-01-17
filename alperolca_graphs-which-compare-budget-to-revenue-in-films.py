# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns  # visualization tool

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
# data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
data = pd.read_csv("../input/tmdb_5000_movies.csv")
data.info()
data.corr()
data.budget.plot(kind ='line',color = 'r',grid = True,alpha = 0.5,linewidth = 1,label ='Budget',linestyle = '--')
data.revenue.plot(color = 'b',grid = True,alpha = 0.5,label ='Revenue',linewidth = 1,linestyle = '-')
plt.legend()
plt.xlabel('a the budget which makes for  a film that director devotes  .')
plt.ylabel('the money which earns from the film.')
plt.title('Budget Compare Revenue')
plt.show()
data.revenue.plot(kind ='hist',color = 'r',bins=50,figsize=(10,5))
data.budget.plot(kind = 'hist',color = 'black',bins =20,figsize =(10,5))

plt.ylabel("Revenue")
plt.xlabel('Budget')
plt.show()
