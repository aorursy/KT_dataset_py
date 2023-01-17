# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
from plotly import __version__

from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

import cufflinks as cf

init_notebook_mode(connected=True)

cf.go_offline()
df = pd.DataFrame(np.random.randn(100,4), columns = 'A B C D'.split())
df
df2 = pd.DataFrame({'Category': ['A', 'B', 'C'], 'Values': [32, 10, 23]})
df2
df.iplot()
df.iplot(kind = 'scatter', x = 'A', y = 'B', mode = 'markers')
df2.iplot(kind = 'bar', x = 'Category', y = 'Values')
df.sum().iplot(kind = 'bar')
df.iplot(kind = 'box')
df3 = pd.DataFrame({'x':[1,2,3,4,5], 'y':[10,20,30,20,10], 'z':[500,400,300,200,100]})
df3.iplot(kind='surface')
df['A'].iplot(kind='hist')