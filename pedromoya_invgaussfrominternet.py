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
# All credits from https://www.pythond.com/23015/como-se-ven-todas-las-distribuciones-disponibles-en-scipy-stats.html

# © 2017 Desarrollo de Python
import io 

import numpy as np 

import pandas as pd 

import scipy.stats as stats 

import matplotlib 

import matplotlib.pyplot as plt

matplotlib.rcParams['figure.figsize'] = (16.0, 14.0) 

matplotlib.style.use('ggplot') 

DISTRIBUTIONS = [stats.recipinvgauss(mu=0.63, loc=0.0, scale=1.0)]
r = stats.invgauss.rvs(mu=0.5, size=1000, scale=1, loc=0)

pd.Series(r).hist(bins=32, density=True) 
r = stats.invgauss.rvs(mu=2.5, size=1000, scale=1, loc=0)

pd.Series(r).hist(bins=32, density=True) 
r = stats.invgauss.rvs(mu=0.5, size=1000, scale=1, loc=0)

pd.Series(r).hist(bins=32, density=True)
r = stats.invgauss.rvs(mu=0.5, size=1000, scale=0.5, loc=0.5)

pd.Series(r).hist(bins=32, density=True)
r = stats.invgauss.rvs(mu=0.5, size=1000, scale=0.5, loc=4.5)

pd.Series(r).hist(bins=32, density=True)
r = stats.invgauss.rvs(mu=0.20, size=1000, scale=1, loc=0)

pd.Series(r).hist(bins=32, density=True)
r = stats.invgauss.rvs(mu=0.20, size=1000, scale=1, loc=0)

pd.Series(r).hist(bins=32, density=True)
r = stats.invgauss.rvs(mu=0.20, size=1000, scale=1, loc=0)

pd.Series(r).hist(bins=32, density=True)
r = stats.invgauss.rvs(mu=0.20, size=1000, scale=0.5, loc=0)

pd.Series(r).hist(bins=32, density=True)
r = stats.invgauss.rvs(mu=0.15, size=1000, scale=2, loc=0)

pd.Series(r).hist(bins=32, density=True)