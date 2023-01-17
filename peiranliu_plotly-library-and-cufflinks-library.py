# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import pandas as pd

import numpy as np

from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

init_notebook_mode(connected=True)    #THIS LINE IS MOST IMPORTANT AS THIS WILL DISPLAY PLOT ON 

#NOTEBOOK WHILE KERNEL IS RUNNING

%matplotlib inline
from plotly import __version__

from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot



print(__version__) # requires version >= 1.9.0
#Version is 3.7.0 which is good
import cufflinks as cf
init_notebook_mode(connected=True)
df = pd.DataFrame(np.random.randn(100,4),columns='A B C D'.split())
df.head()

#Create the fake data
df2 = pd.DataFrame({'Category':['A','B','C'],'Values':[32,43,50]})
df2
df.iplot(kind='scatter',x='A',y='B',mode='markers',size=10)
df2.iplot(kind='bar',x='Category',y='Values')
df.count().iplot(kind='bar')
df.iplot(kind='box')
df3 = pd.DataFrame({'x':[1,2,3,4,5],'y':[10,20,30,20,10],'z':[5,4,3,2,1]})

df3.iplot(kind='surface',colorscale='rdylbu')
df[['A','B']].iplot(kind='spread')
df['A'].iplot(kind='hist',bins=25)
df.iplot(kind='bubble',x='A',y='B',size='C')
df.scatter_matrix()