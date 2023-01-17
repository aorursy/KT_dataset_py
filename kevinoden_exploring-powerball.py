# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt # Collection of functions for scientific and publication-ready visualization
import plotly.offline as py     # Open source library for composing, editing, and sharing interactive data visualization 
from matplotlib import pyplot
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.tools as tls

from collections import Counter
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
# Load data file
df = pd.read_csv('../input/powerball_draw_order.csv')
# initial view of data
df.head()
df.tail()
# limit the data to the relevent columns  PB1,,,PB5 and the Powerball
select_columns= ["PB1","PB2","PB3","PB4","PB5","Powerball"]
pbData = df[select_columns]
pbData.head()
pbData.describe()
pbData.hist(bins=50, figsize=(20,15))
plt.show()
#utilizing the 'value_counts' method and taking the top five occurences
pbData["PB1"].value_counts()[0:5]
## pbData["PB1"].value_counts()
## expected counts  =   target_index * 1/69
##
## now perform a chi-square test to see if "in aggregate" the first draw appears random
## what this does not test is if there is a single number which is statistically significant
## 
import scipy.stats as stats
expected_ct = 286*1/69  #sample length * probability of 

chi_squared_stat =  sum( (pbData["PB1"].value_counts() - expected_ct)**2/ expected_ct)
crit = stats.chi2.ppf(q = 0.95, # Find the critical value for 95% confidence*
                      df = 68)   # Df = number of variable categories - 1 = k-1

print("Critical value")
print(crit)

p_value = 1 - stats.chi2.cdf(x=chi_squared_stat,  # Find the p-value
                             df=68)
print("P value")
print(p_value)

print("Chi_squared_stat")
print(chi_squared_stat)
# First at the 95% level
n= 286     # number of draws
p1 = 1/69  # probability of a paticular number being drawn

chi_squared_stat =  (pbData["PB1"].value_counts().get_values()[0] - n*p1)**2/ (n*p1*(1-p1))
crit = stats.chi2.ppf(q = 0.95, # Find the critical value for 95% confidence*
                      df = 1)   # Df = number of variable categories - 1

print("Critical value")
print(crit)

p_value = 1 - stats.chi2.cdf(x=chi_squared_stat,  # Find the p-value
                             df=1)
print("P value")
print(p_value)

print("Chi_squared_stat")
print(chi_squared_stat)
# second at the 99% level
n= 286     # number of draws
p1 = 1/69  # probability of a paticular number being drawn

chi_squared_stat =  (pbData["PB1"].value_counts().get_values()[0] - n*p1)**2/ (n*p1*(1-p1))
crit = stats.chi2.ppf(q = 0.99, # Find the critical value for 95% confidence*
                      df = 1)   # Df = number of variable categories - 1

print("Critical value")
print(crit)

p_value = 1 - stats.chi2.cdf(x=chi_squared_stat,  # Find the p-value
                             df=1)
print("P value")
print(p_value)

print("Chi_squared_stat")
print(chi_squared_stat)