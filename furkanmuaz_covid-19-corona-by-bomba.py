# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import missingno as msno

import matplotlib.pyplot as plt

import seaborn as sns

import matplotlib_venn as venn

from math import pi

#from pandas.tools.plotting import parallel_coordinates

import plotly.graph_objs as go

#import plotly.plotly as py

from plotly.offline import init_notebook_mode, iplot

init_notebook_mode(connected=True)

import warnings

warnings.filterwarnings("ignore")



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
covid = pd.read_csv("../input/coronavirus-2019ncov/covid-19-all.csv")
covid
msno.matrix(covid)

plt.show()
covid['Country/Region'].head()
covid.columns
plt.figure(figsize=(15,10))

parallel_coordinates(covid,'Country/Region', colormap=plt.get_cmap("Set1"))

plt.title("Hangi ulkede ne kadar vaka var")

plt.xlabel("Ulkeler")

plt.ylabel("vaka")

plt.savefig('praph.png')

plt.show()
corr = covid.iloc[:,0:4].corr()

corr
import networkx as nx



# Transform it in a links data frame (3 cloumns only)

links = corr.stack().reset_index()

links.columns = ['var1', 'var2', 'value']



# correlation

threshold = -1 # sınır



# Keep only correlation over a threshold and remove self correlation ( cor (A,A)=1)

links_filtered = links.loc [ (links ['value'] >= threshold) & (links ['var1'] != links ['var2'])]



# Build your graph

G=nx.from_pandas_dataframe(links_filtered, 'var1', 'var2')



# Plot the network

nx.draw_circular(G, with_labels=True, node_color='orange', node_size=300, edge_color='red', linewidths=1, font_size=10)
links
from matplotlib_venn import venn2

Latitude = covid.iloc[:,0]

Longitude = covid.iloc[:,1]

venn2(subsets = (len(Latitude)-15, len(Latitude)-15, 15), set_labels = ('Latitude','Longitude'))

plt.show()
covid.columns