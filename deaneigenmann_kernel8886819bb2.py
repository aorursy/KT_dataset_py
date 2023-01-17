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
## variables
num_peers = 1
req_freq = 1 # times per minutes
node_num_reqs_per_day = 1440 * req_freq
## Quick Calculation for how bandwidth in and out consumption of lookups to find a node

from math import log10

haystack = 100 # number of nodes
find_node_packet = 72
nodes_packet = 2762

num_roundtrips = log10(haystack)

outbound = find_node_packet * num_roundtrips
inbound = nodes_packet * num_roundtrips

print("outbound: % 2db" %(outbound))
print("inbound: % 2db" %(inbound))
## variables
num_peers = 1
req_freq = 1                                    # req/minutes
node_num_reqs_per_day = 1440 * req_freq         # req/day
band_FINDNODE = 216                             # bytes
band_NODES = 8286                               # bytes

band_per_node_per_day = (num_peers * node_num_reqs_per_day) * (band_FINDNODE + band_NODES)
print("KB per day per node: {:,}".format(band_per_node_per_day / 1000))