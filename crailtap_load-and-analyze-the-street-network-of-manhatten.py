# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 

import geoplotlib

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
#First import a library which can load the graph, e.g., networkx or osmnx 

import networkx as nx



#load the graph

G = nx.read_graphml('../input/manhatten.graphml')



#plot some first information about the graph:

nx.info(G)
nx.draw(G, pos=nx.spring_layout(G), node_size=0.01, width=0.1)