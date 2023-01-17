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
#The two most popular data viz lib are matplotlib and Seaborn
import pandas as pd
import numpy as np
from numpy.random import randn
from pandas import Series, DataFrame
import matplotlib.pyplot as plt
from matplotlib import rcParams
import seaborn as sb
#Line chart shows changes of one or more variable with regards to another variable, usually time.
#Bars represent a certain value against another attribute my measuring the length of the bar i.e. they represent observation counts within categories
#Pie charts represent several attributes as pie slices
#There are 2 methods to plot building: a. Functionally using functions b.Object oriented using plots and plot elements
#We use inline so that the viz appear on the notebook and not a separate tab. Image size can be specified using rcParams. Seaborn can support multiple styles. We will use "whitegid"
%matplotlib inline
rcParams['figure.figsize']=5,4
sb.set_style('whitegrid')
#Define a set of values for x and y
x=range(1,10)
y=[1,2,3,4,0,4,3,2,1]
plt.plot(x,y)
