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
#TimeSeries data can help predict and forecast from historical data

#1. Constant Time Series: Average does not change over time

#2. Trended Time Series: Average increases over time

#3. Untrended Seasonal TS: Has a uniform increase and decrease but the average doesn't change

#4. Trended Seasonal TS
import numpy as np

import pandas as pd

import seaborn as sb

import matplotlib.pyplot as plt

from pylab import rcParams

from pandas import Series, DataFrame

from numpy.random import randn
#params for data viz

%matplotlib inline

rcParams['figure.figsize']=5,4

sb.set_style('whitegrid')
#Simplest TS plot
