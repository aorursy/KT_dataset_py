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
import matplotlib.pyplot as plt
base_dir = "/kaggle/input/elliptic-data-set/elliptic_bitcoin_dataset/elliptic_bitcoin_dataset/"

feature_file = "elliptic_txs_features.csv"

edgelist_file = "elliptic_txs_edgelist.csv"

classes_file = "elliptic_txs_classes.csv"
feature_data = pd.read_csv(base_dir + feature_file,header=None)

edge_data = pd.read_csv(base_dir + edgelist_file)

class_data = pd.read_csv(base_dir + classes_file)
feature_data.shape, edge_data.shape, class_data.shape
feature_data.head(2)
feature_data.columns = [str(x) for x in np.arange(0,167)]

feature_data.columns
edge_data.head()
class_data.head()
g = pd.DataFrame(class_data.groupby(["class"]).count()["txId"] / class_data.shape[0] * 100).reset_index()

plt.bar(g["class"],g["txId"])

plt.show()
feature_data.iloc[:,1:].describe()
a = pd.merge(left=edge_data,right=class_data,left_on="txId1",right_on="txId",how="left").rename(columns={"class" : "txId1_class"}).drop(columns=["txId"])

edge_data = pd.merge(left=a,right=class_data,left_on="txId2",right_on="txId",how="left").rename(columns={"class" : "txId2_class"}).drop(columns=["txId"])
edge_data.head()
a = pd.merge(left=edge_data,right=feature_data[["0","1"]],left_on="txId1",right_on="0",how="left").rename(columns={"1" : "txId1_timestep"}).drop(columns=["0"])

edge_data = pd.merge(left=a,right=feature_data[["0","1"]],left_on="txId2",right_on="0",how="left").rename(columns={"1" : "txId2_timestep"}).drop(columns=["0"])
edge_data.head()
np.where(edge_data.txId1_timestep != edge_data.txId2_timestep)
edge_data = edge_data.assign(class_comb = edge_data.txId1_class + "_" + edge_data.txId2_class)

edge_data.head()
edge_data.groupby(["class_comb"]).agg({"txId1" : "count",

                                     "txId2" : lambda x : len(np.unique(x))/edge_data.shape[0] * 100}).reset_index()
source_set = set(edge_data.txId1.values)

dest_set = set(edge_data.txId2.values)
len(source_set - dest_set),len(source_set.intersection(dest_set)), len(dest_set - source_set)
len(source_set - dest_set) + len(source_set.intersection(dest_set))+ len(dest_set - source_set)
only_source = source_set - dest_set

only_dest = dest_set - source_set

common_nodes = source_set.intersection(dest_set)
'''

loop_list = []

for s in only_source:

    df = edge_data[edge_data.txId1 == s]

    if df.txId2.isin(common_nodes):

'''