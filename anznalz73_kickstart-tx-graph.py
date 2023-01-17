# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



# https://github.com/FelixChop/MediumArticles/blob/master/Graph_analysis_Python.ipynb



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import networkx as nx



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.



df = pd.read_csv("/kaggle/input/mytxgraphcsv/export-0x2921722258114c1304afb682d2a55974de10a0bc.csv")



column_edge = 'Txhash'

column_ID = 'From'

column_TO = 'To'



G = nx.from_pandas_edgelist(df, source=column_ID, target=column_TO, edge_attr=column_edge)



#G.add_nodes_from(nodes_for_adding=df.Txhash.tolist())



nx.draw(G)