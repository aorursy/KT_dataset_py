# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

filePaths = []

pathCount = 0

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        path = os.path.join(dirname, filename)

        print(str(pathCount), ".", path)

        pathCount += 1

        filePaths.append(path)



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import warnings

warnings.filterwarnings('ignore')

def viewInfo(table, name="Table"):

    col = table.index.tolist()

    row = table.columns.tolist()

    colLenght = len(col)

    rowLenght = len(row)

    print('[------{0}------]\n[*] Row : {1} | Col {2}'.format(name, rowLenght, colLenght))

    print(row, '\n')
# Read phanphoi data
import pandas as pd



factories = pd.read_csv(filePaths[8], index_col=0)

viewInfo(factories, "Factories")

factories
warehouses = pd.read_csv(filePaths[7])

viewInfo(warehouses, "Ware Houses")

warehouses
cost = pd.read_csv(filePaths[3], index_col=0)

viewInfo(cost, "Cost")

cost
trans = pd.read_csv(filePaths[9], index_col=0)

viewInfo(trans, "Trans")            

trans.head()
join_data_1 = pd.merge(trans, cost, left_on=["ToFC", "FromWH"], right_on=["FCID", "WHID"], how="left")

viewInfo(trans, "Trans")

viewInfo(cost, "Cost")

viewInfo(join_data, "Join Data")

join_data_1.head()
join_data_2 = pd.merge(join_data_1, factories, left_on="ToFC", right_on="FCID", how="left")

viewInfo(join_data_1, "Join 1")

viewInfo(factories, "Factories")

viewInfo(join_data_2, "Join 2")

join_data_2.head()
join_data = pd.merge(join_data_2, warehouses, left_on="FromWH", right_on="WHID", how="left")

join_data = join_data[["TransactionDate","Quantity","Cost","ToFC","FCName","FCDemand","FromWH","WHName","WHSupply","WHRegion"]]

viewInfo(join_data)

join_data.head()
kanto = join_data.loc[join_data["WHRegion"]=="??????"]

viewInfo(kanto, "Kanto Region")

kanto.head()
tohoku = join_data.loc[join_data["WHRegion"]=="??????"]

tohoku.head()
print("---????????????---")

print("???????????????????????????: " + str(kanto["Cost"].sum()) + "??????")

print("???????????????????????????: " + str(tohoku["Cost"].sum()) + "??????")

print("---?????????????????????---")

print("????????????????????????????????????: " + str(kanto["Quantity"].sum()) + "???")

print("????????????????????????????????????: " + str(tohoku["Quantity"].sum()) + "???")

print("---???????????????????????????????????????---")

tmp = (kanto["Cost"].sum() / kanto["Quantity"].sum()) * 10000

print("??????????????????????????????????????????????????????: " + str(int(tmp)) + "???")

tmp = (tohoku["Cost"].sum() / tohoku["Quantity"].sum()) * 10000

print("??????????????????????????????????????????????????????: " + str(int(tmp)) + "???")
print(cost.shape)

print(cost.loc[cost["FCID"]=="FC00001"])

print(factories)

cost_chk = pd.merge(cost, factories, on="FCID", how="left")

print("???????????????????????????????????????" + str(cost_chk["Cost"].loc[cost_chk["FCRegion"]=="??????"].mean()) + "??????")

print("???????????????????????????????????????" + str(cost_chk["Cost"].loc[cost_chk["FCRegion"]=="??????"].mean()) + "??????")
import networkx as nx

import matplotlib.pyplot as plt



# ????????????????????????????????????

G=nx.Graph()



# ???????????????

G.add_node("nodeA")

G.add_node("nodeB")

G.add_node("nodeC")



# ????????????

G.add_edge("nodeA","nodeB")

G.add_edge("nodeA","nodeC")

G.add_edge("nodeB","nodeC")



# ???????????????

pos={}

pos["nodeA"]=(0,0)

pos["nodeB"]=(1,1)

pos["nodeC"]=(0,1)



# ??????

nx.draw(G,pos)



# ??????

plt.show()
import networkx as nx

import matplotlib.pyplot as plt



# ???????????????????????????????????????

G=nx.Graph()



# ???????????????

G.add_node("nodeA")

G.add_node("nodeB")

G.add_node("nodeC")

G.add_node("nodeD")



# ????????????

G.add_edge("nodeA","nodeB")

G.add_edge("nodeA","nodeC")

G.add_edge("nodeB","nodeC")

G.add_edge("nodeA","nodeD")



# ???????????????

pos={}

pos["nodeA"]=(0,0)

pos["nodeB"]=(1,1)

pos["nodeC"]=(0,1)

pos["nodeD"]=(1,0)



# ??????

nx.draw(G,pos, with_labels=True)



# ??????

plt.show()
df_w = pd.read_csv(filePaths[10])

df_p = pd.read_csv(filePaths[0])

viewInfo(df_w, "Network Weight")

viewInfo(df_p, "Network Position")

print("---Position---")

print(df_p)

print("---Weight---")

print(df_w)
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import networkx as nx



# ?????????????????????????????????

size = 10

edge_weights = []

for i in range(len(df_w)):

    for j in range(len(df_w.columns)):

        edge_weights.append(df_w.iloc[i][j]*size)

print("[+] Edge :", len(edge_weights))



# ????????????????????????????????????

G = nx.Graph()



# ???????????????

for i in range(len(df_w.columns)):

    G.add_node(df_w.columns[i])



# ????????????

for i in range(len(df_w.columns)):

    for j in range(len(df_w.columns)):

        G.add_edge(df_w.columns[i],df_w.columns[j])



# ???????????????

pos = {}

for i in range(len(df_w.columns)):

    node = df_w.columns[i]

    pos[node] = (df_p[node][0],df_p[node][1])



# ??????

nx.draw(G, pos, with_labels=True,font_size=16, node_size = 1000, node_color='k', font_color='w', width=edge_weights)



# ??????

plt.show()
import pandas as pd



# ?????????????????????

df_tr = pd.read_csv(filePaths[11], index_col="??????")

df_pos = pd.read_csv(filePaths[6])

print(df_tr)

print(df_pos)
import pandas as pd

import matplotlib.pyplot as plt

import networkx as nx





# ????????????????????????????????????

G = nx.Graph()



# ???????????????

for i in range(len(df_pos.columns)):

    G.add_node(df_pos.columns[i])



# ????????????&?????????????????????????????????

num_pre = 0

edge_weights = []

size = 0.1

for i in range(len(df_pos.columns)):

    for j in range(len(df_pos.columns)):

        if not (i==j):

            # ????????????

            G.add_edge(df_pos.columns[i],df_pos.columns[j])

            # ???????????????????????????

            if num_pre<len(G.edges):

                num_pre = len(G.edges)

                weight = 0

                if (df_pos.columns[i] in df_tr.columns)and(df_pos.columns[j] in df_tr.index):

                    if df_tr[df_pos.columns[i]][df_pos.columns[j]]:

                        weight = df_tr[df_pos.columns[i]][df_pos.columns[j]]*size

                elif(df_pos.columns[j] in df_tr.columns)and(df_pos.columns[i] in df_tr.index):

                    if df_tr[df_pos.columns[j]][df_pos.columns[i]]:

                        weight = df_tr[df_pos.columns[j]][df_pos.columns[i]]*size

                edge_weights.append(weight)

                



# ???????????????

pos = {}

for i in range(len(df_pos.columns)):

    node = df_pos.columns[i]

    pos[node] = (df_pos[node][0],df_pos[node][1])

    

# ??????

nx.draw(G, pos, with_labels=True,font_size=16, node_size = 1000, node_color='k', font_color='w', width=edge_weights)



# ??????

plt.show()
df_tr = pd.read_csv(filePaths[11], index_col="??????")

df_tc = pd.read_csv(filePaths[5], index_col="??????")

print("---Route---")

print(df_tr)

print("\n---Cost---")

print(df_tc)
# ?????????????????????

def trans_cost(df_tr,df_tc):

    cost = 0

    for i in range(len(df_tc.index)):

        for j in range(len(df_tr.columns)):

            cost += df_tr.iloc[i][j]*df_tc.iloc[i][j]

    return cost



print("??????????????????:"+str(trans_cost(df_tr,df_tc)))
df_demand = pd.read_csv(filePaths[2])

df_supply = pd.read_csv(filePaths[1])

print("---Route---")

print(df_tr)

print("---Demand---")

print(df_demand)

print("\n---Supply---")

print(df_supply)
import pandas as pd



# ?????????????????????

print("---- F??????????????????????????????????????? ---")

# ????????????????????????

for i in range(len(df_demand.columns)):

    temp_sum = sum(df_tr[df_demand.columns[i]])

    print(str(df_demand.columns[i])+"???????????????:"+str(temp_sum)+" (?????????:"+str(df_demand.iloc[0][i])+")")

    if temp_sum>=df_demand.iloc[0][i]:

        print("????????????????????????????????????")

    else:

        print("????????????????????????????????????????????????????????????????????????????????????")



print("\n---- ????????????????????????????????????????????? ---")

# ????????????????????????

for i in range(len(df_supply.columns)):

    temp_sum = sum(df_tr.loc[df_supply.columns[i]])

    print(str(df_supply.columns[i])+"??????????????????:"+str(temp_sum)+" (????????????:"+str(df_supply.iloc[0][i])+")")

    if temp_sum<=df_supply.iloc[0][i]:

        print("?????????????????????????????????")

    else:

        print("????????????????????????????????????????????????????????????????????????????????????")
# ?????????????????????

df_tr_new = pd.read_csv(filePaths[4], index_col="??????")

print("---Trans Route New---")

print(df_tr_new)

print("\n---Demand---")

print(df_demand)

print("\n---Supply---")

print(df_supply)
import pandas as pd

import numpy as np



# ??????????????????????????? 

print("??????????????????(?????????):"+str(trans_cost(df_tr_new,df_tc)))



# ????????????????????????

# ?????????

def condition_demand(df_tr,df_demand):

    flag = np.zeros(len(df_demand.columns))

    for i in range(len(df_demand.columns)):

        temp_sum = sum(df_tr[df_demand.columns[i]])

        if (temp_sum>=df_demand.iloc[0][i]):

            flag[i] = 1

    return flag

            

# ?????????

def condition_supply(df_tr,df_supply):

    flag = np.zeros(len(df_supply.columns))

    for i in range(len(df_supply.columns)):

        temp_sum = sum(df_tr.loc[df_supply.columns[i]])

        if temp_sum<=df_supply.iloc[0][i]:

            flag[i] = 1

    return flag



print("????????????????????????:"+str(condition_demand(df_tr_new,df_demand)))

print("????????????????????????:"+str(condition_supply(df_tr_new,df_supply)))