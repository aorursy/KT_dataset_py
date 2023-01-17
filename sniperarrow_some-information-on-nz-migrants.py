import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import csv

from sklearn import preprocessing

import matplotlib

matplotlib.style.use('ggplot')

%matplotlib inline

import math

import matplotlib as mpl

import plotly

import colorsys

plt.style.use('seaborn-talk')

from __future__ import division

import pylab

import plotly.plotly as py

import plotly.graph_objs as go

from matplotlib import colors as mcolors
df = pd.read_csv('../input/migration_nz.csv')

df.describe()
df = pd.read_csv('../input/migration_nz.csv')

df.head()
df.info()
from matplotlib import pyplot as plt

 

plt.figure(figsize=(8,6))

plt.plot(df['Year'], df['Value'], 'o')

plt.xlabel('Year')

plt.ylabel('Number of Migrants')

plt.show()
pd.pivot_table(df,index=["Citizenship", "Measure"], values=["Value"], aggfunc=np.sum)
pd.pivot_table(df,index=["Country","Citizenship", "Year"],

               values=["Value"],aggfunc=[np.sum],fill_value=0)