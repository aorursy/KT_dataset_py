# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
# Read input files
busBrkDly = pd.read_csv('../input/bus-breakdown-and-delays.csv')
#socrata = pd.read_json('../input/socrata_metadata.json',typ='frame')

# Describe Numerical data
busBrkDly.describe()
# Describe Categorical data
busBrkDly.describe(include='O')
# Count School Year through countplot.
plt.figure(figsize=(20,5))
ax=sns.countplot(x='School_Year', hue = 'School_Year',data= busBrkDly  ,linewidth=5,edgecolor=sns.color_palette("dark", 3))
plt.title('School Year?');
plt.figure(figsize=(20,10))
ax=sns.countplot(x='Run_Type',data= busBrkDly  ,linewidth=5,edgecolor=sns.color_palette("dark", 3))
plt.title('Run Type?');
plt.figure(figsize=(20,10))
ax = sns.countplot(x='Reason',hue='School_Year',data=busBrkDly, linewidth=2, edgecolor=sns.color_palette("dark",4))

# import plotly
import plotly.plotly as py
import plotly.graph_objs as go
# these two lines are what allow your code to show up in a notebook!
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode()

data = [go.Scatter(x=busBrkDly.School_Year, y=busBrkDly.How_Long_Delayed)]

