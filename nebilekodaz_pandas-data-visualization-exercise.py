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
import matplotlib.pyplot as plt
df3 = pd.read_csv("../input/df3")
%matplotlib inline
df3.head()
df3.info()
df3.plot.scatter(x='a',y='b',c='red',s=50,figsize=(12,3))
df3['a'].plot.hist()
plt.style.use('ggplot')
df3['a'].plot.hist(bins=20,alpha=0.5)
df3[['a','b']].plot.box()
df3['d'].plot.kde(alpha=0.5)
df3['d'].plot.density(alpha=0.5,lw=5,ls='--')
df3.ix[0:30].plot.area(alpha=0.5)
df3.ix[0:30].plot.area(alpha=0.5)
plt.legend(loc='center left', bbox_to_anchor=(1, 0.75))
