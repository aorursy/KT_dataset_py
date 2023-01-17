# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
aliases = pd.read_csv("../input/Aliases.csv")
email_receivers = pd.read_csv("../input/EmailReceivers.csv")
emails = pd.read_csv("../input/Emails.csv")
persons = pd.read_csv("../input/Persons.csv")
emails['RawText'][1]
import networkx as nx
import matplotlib.pyplot as plt
%matplotlib inline
# Extract the tos and the froms
tos = emails['MetadataTo']
froms = emails['MetadataFrom']
graph = nx.DiGraph()
for _to in tos:
    for _from in froms:
        if _to==';H':
            _to = 'H'
        if _from==';H':
            _from = 'H'
        if (_from,_to) in graph.edges():
            graph[_from][_to]['weight'] += 1
        else:
            graph.add_edge(_from,_to,weight=1)
graph.edges()
