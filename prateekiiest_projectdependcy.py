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
import graphviz
d = graphviz.Digraph(format = 'png',
             edge_attr={'color':'brown', 
                         'style':'filled'},
              graph_attr={'rankdir':'LR', 
                          'bgcolor':'transparent'},
              node_attr={'fontsize':'31', 
                         'shape':'sqaure', 
                         'color':'black',
                        'style':'filled',
                        'fillcolor':'antiquewhite'})
lines = tuple(open("../input/dependencies.log.gz.txt", 'r'))

lines
for i in range(len(lines)):
    p = lines[i].split(":")
    d.node(p[0], **{'width':'2', 'height':'2'})
    for j in p[1].split(","):
        
        d.edge(p[0],j)
    
graphviz.Source(d)

